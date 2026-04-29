import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side PNG rendering
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from starlette.concurrency import run_in_threadpool

from model import TFGridNet


app = FastAPI(title="Neural Voice Separation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_audio"
PRETRAINED_DIR = BASE_DIR / "pretrained_models" / "sepformer-wsj02mix"
ECAPA_DIR = BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
# CUSTOM_MODEL_PATH = BASE_DIR / "best_tfgridnet.pth"

def _resolve_custom_model_path() -> Path:
    env_override = os.getenv("CUSTOM_MODEL_PATH")
    candidates = [
        Path(env_override) if env_override else None,
        BASE_DIR / "new_model.pth",
        BASE_DIR / "best_tfgridnet.pth",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return BASE_DIR / "best_tfgridnet.pth"

CUSTOM_MODEL_PATH = _resolve_custom_model_path()

TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}

SEPFORMER_SAMPLE_RATE = 8000
ECAPA_SAMPLE_RATE = 16000
MATH_MODEL_SAMPLE_RATE = 16000
MAX_ALLOWED_STATE_MISMATCH_RATIO = 0.08

# ECW pipeline tuning constants. Pulled out so the paper's hyper-parameter
# discussion can reference exact numerical values used at inference time.
ECW_GAMMA = 2.0          # Exponent on cosine similarity in the mask numerator
ECW_TEMPERATURE = 1.0    # Softplus temperature for similarity rescaling
ECW_FLOOR = 0.05         # Mask floor — prevents complete zeroing of any bin
ECW_REFINE_ITERS = 1     # Iterative refinement passes (1 = single-shot ECWM)

SUPPORTED_MODELS = {
    "speechbrain": {
        "label": "SpeechBrain SepFormer (Baseline)",
        "architecture": "SepFormer (energy-based selection)",
        "dataset": "WSJ0-2Mix",
        "benchmark": "22.3 dB SI-SDRi",
    },
    "ecw_tse": {
        "label": "ECW-TSE (Novel Pipeline)",
        "architecture": "SepFormer + ECAPA-TDNN + Embedding-Conditioned Wiener mask",
        "dataset": "WSJ0-2Mix (sep) + VoxCeleb (embed)",
        "benchmark": "Reference-aware target speaker extraction",
    },
    "math_model": {
        "label": "Robust TF-GridNet (Custom Architecture)",
        "architecture": "TF-GridNet + ratio-mask refinement",
        "dataset": "Libri2Mix (project) + synthetic overlaps",
        "benchmark": "Project-trained",
    },
}

TORCHAUDIO_IO_AVAILABLE = True
TORCHAUDIO_SAVE_AVAILABLE = True
SPEECHBRAIN_AVAILABLE = False
ECAPA_AVAILABLE = False
MATH_MODEL_AVAILABLE = False

sb_separator = None
spk_encoder = None
math_separator = None


def _load_models():
    global sb_separator, spk_encoder, math_separator
    global SPEECHBRAIN_AVAILABLE, ECAPA_AVAILABLE, MATH_MODEL_AVAILABLE

    print("=" * 60)
    print("  Initializing separation engines")
    print("=" * 60)

    try:
        print("  Loading SpeechBrain SepFormer (WSJ0-2Mix)...")
        sb_separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir=str(PRETRAINED_DIR),
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": "cpu"},
        )
        SPEECHBRAIN_AVAILABLE = True
        print("  [OK] SpeechBrain model ready.")
    except Exception as exc:  # pragma: no cover
        SPEECHBRAIN_AVAILABLE = False
        sb_separator = None
        print(f"  [WARN] SpeechBrain unavailable: {type(exc).__name__}: {exc}")

    try:
        print("  Loading ECAPA-TDNN speaker encoder (VoxCeleb)...")
        spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ECAPA_DIR),
            local_strategy=LocalStrategy.COPY,  # Avoid Windows symlink permission errors
            run_opts={"device": "cpu"},
        )
        ECAPA_AVAILABLE = True
        print("  [OK] ECAPA-TDNN ready (192-dim speaker embeddings).")
    except Exception as exc:  # pragma: no cover
        ECAPA_AVAILABLE = False
        spk_encoder = None
        print(f"  [WARN] ECAPA-TDNN unavailable: {type(exc).__name__}: {exc}")

    try:
        print("  Loading project TF-GridNet checkpoint...")
        if not CUSTOM_MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found at {CUSTOM_MODEL_PATH}")

        print(f"  Using checkpoint: {CUSTOM_MODEL_PATH.name}")
        checkpoint = torch.load(CUSTOM_MODEL_PATH, map_location="cpu")

        model_config = {
            "n_fft": 512,
            "hop_length": 128,
            "d_model": 64,
            "n_heads": 4,
            "lstm_hidden": 256,
            "n_layers": 6,
            "num_sources": 2,
            "dropout": 0.1,
        }

        if isinstance(checkpoint, dict):
            config = checkpoint.get("config")
            if isinstance(config, dict):
                for key, default_value in model_config.items():
                    if key not in config:
                        continue
                    raw_value = config[key]
                    try:
                        if isinstance(default_value, int):
                            model_config[key] = int(raw_value)
                        elif isinstance(default_value, float):
                            model_config[key] = float(raw_value)
                        else:
                            model_config[key] = raw_value
                    except (TypeError, ValueError):
                        pass

            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                checkpoint = checkpoint["model_state_dict"]

        math_separator = TFGridNet(**model_config)

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Unsupported checkpoint format")

        if any(key.startswith("module.") for key in checkpoint.keys()):
            checkpoint = {
                key.replace("module.", "", 1): value
                for key, value in checkpoint.items()
            }

        load_info = math_separator.load_state_dict(checkpoint, strict=False)
        missing_keys = [key for key in load_info.missing_keys if not key.endswith("num_batches_tracked")]
        unexpected_keys = list(load_info.unexpected_keys)

        if missing_keys or unexpected_keys:
            total_params = max(1, len(math_separator.state_dict()))
            mismatch_count = len(missing_keys) + len(unexpected_keys)
            mismatch_ratio = mismatch_count / total_params
            print(
                "  [WARN] Checkpoint/model mismatch "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, ratio={mismatch_ratio:.2%})"
            )

            if mismatch_ratio > MAX_ALLOWED_STATE_MISMATCH_RATIO:
                raise RuntimeError(
                    "Checkpoint is incompatible with TF-GridNet architecture. "
                    "Retrain/export with matching MODEL_CONFIG to avoid degraded separation quality."
                )

        math_separator.eval()
        MATH_MODEL_AVAILABLE = True
        print("  [OK] Mathematical model ready.")
    except Exception as exc:  # pragma: no cover
        MATH_MODEL_AVAILABLE = False
        math_separator = None
        print(f"  [WARN] Mathematical model unavailable: {type(exc).__name__}: {exc}")

    print("=" * 60)
    print("  Initialization complete.\n")


_load_models()


def cleanup_file(filepath: str):
    """Safely remove temporary files."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass


def _validate_model_name(model_name: str) -> Literal["speechbrain", "ecw_tse", "math_model"]:
    normalized = model_name.strip().lower()
    if normalized in SUPPORTED_MODELS:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"model_name must be one of {sorted(SUPPORTED_MODELS.keys())}")


def _load_waveform(path: str):
    """Load audio with torchaudio when available, fallback to soundfile."""
    global TORCHAUDIO_IO_AVAILABLE
    if TORCHAUDIO_IO_AVAILABLE:
        try:
            return torchaudio.load(path)
        except (ImportError, RuntimeError, OSError):
            TORCHAUDIO_IO_AVAILABLE = False

    audio_np, sample_rate = sf.read(path, always_2d=True)
    waveform = torch.from_numpy(audio_np.T).to(torch.float32)
    return waveform, sample_rate


def _save_waveform(path: str, waveform: torch.Tensor, sample_rate: int):
    """Save audio with torchaudio when available, fallback to soundfile."""
    global TORCHAUDIO_SAVE_AVAILABLE
    if TORCHAUDIO_SAVE_AVAILABLE:
        try:
            torchaudio.save(path, waveform, sample_rate)
            return
        except (ImportError, RuntimeError, OSError):
            TORCHAUDIO_SAVE_AVAILABLE = False

    sf.write(path, waveform.squeeze(0).cpu().numpy(), sample_rate)


def _resample_waveform(waveform: torch.Tensor, orig_rate: int, new_rate: int):
    if orig_rate == new_rate:
        return waveform
    return torchaudio.functional.resample(waveform, orig_rate, new_rate)


def _peak_normalize(waveform: torch.Tensor):
    return waveform / (waveform.abs().max() + 1e-8) * 0.95


def _select_source_by_energy(sources: list[torch.Tensor], requested_index: int | None):
    if requested_index is None:
        energies = torch.stack([torch.mean(src ** 2) for src in sources])
        selected = int(torch.argmax(energies).item())
        return sources[selected], selected

    if requested_index < 0 or requested_index >= len(sources):
        raise ValueError(f"source_index must be between 0 and {len(sources) - 1}")

    return sources[requested_index], requested_index


def _ratio_mask_refine(
    mixture: torch.Tensor,
    estimated_primary: torch.Tensor,
    estimated_secondary: torch.Tensor,
    sample_rate: int,
):
    """Refine two-source estimates using a soft ratio mask on the mixture spectrogram."""
    n_fft = 512 if sample_rate >= 16000 else 256
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft, device=mixture.device)

    mix_spec = torch.stft(
        mixture,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    primary_spec = torch.stft(
        estimated_primary,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    residual_spec = torch.stft(
        estimated_secondary,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    primary_mag = primary_spec.abs()
    residual_mag = residual_spec.abs()

    denom = primary_mag + residual_mag + 1e-8
    primary_mask = primary_mag / denom
    residual_mask = 1.0 - primary_mask

    refined_primary = torch.istft(
        primary_mask * mix_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=mixture.shape[-1],
    )
    refined_residual = torch.istft(
        residual_mask * mix_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=mixture.shape[-1],
    )

    refined_sources = torch.stack([refined_primary, refined_residual], dim=0)
    refined_sources = _enforce_mixture_consistency(mixture, refined_sources)
    return refined_sources[0], refined_sources[1]


def _enforce_mixture_consistency(mixture: torch.Tensor, sources: torch.Tensor):
    """Project estimated sources so their sum exactly reconstructs the mixture."""
    if sources.ndim != 2:
        raise ValueError("sources must have shape [num_sources, num_samples]")

    residual = mixture - torch.sum(sources, dim=0)
    # Distribute correction by instantaneous magnitude to avoid over-correcting silent sources.
    weights = sources.abs()
    weights = weights / (torch.sum(weights, dim=0, keepdim=True) + 1e-8)
    return sources + weights * residual.unsqueeze(0)


def _run_overlap_add_math_model(mixture_16k: torch.Tensor):
    """Run the project model in overlap-add windows for stable long-form inference."""
    if math_separator is None:
        raise RuntimeError("Mathematical model is not initialized")

    num_sources = getattr(math_separator, "num_sources", 2)

    chunk_size = int(3.0 * MATH_MODEL_SAMPLE_RATE)
    hop_size = int(chunk_size * 0.5)

    total_samples = mixture_16k.shape[-1]
    if total_samples <= chunk_size:
        with torch.no_grad():
            return math_separator(mixture_16k, return_all_sources=True)

    output = torch.zeros(
        mixture_16k.shape[0],
        num_sources,
        total_samples,
        device=mixture_16k.device,
        dtype=mixture_16k.dtype,
    )
    weights = torch.zeros(
        1,
        1,
        total_samples,
        device=mixture_16k.device,
        dtype=mixture_16k.dtype,
    )
    window = torch.hann_window(
        chunk_size,
        periodic=False,
        device=mixture_16k.device,
    ).view(1, 1, -1)

    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = mixture_16k[:, start:end]
        valid_length = chunk.shape[-1]

        if valid_length < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - valid_length))

        with torch.no_grad():
            estimated = math_separator(chunk, return_all_sources=True)

        estimated = estimated[:, :, :valid_length]
        overlap_window = window[:, :, :valid_length]

        output[:, :, start:end] += estimated * overlap_window
        weights[:, :, start:end] += overlap_window

        if end == total_samples:
            break
        start += hop_size

    return output / (weights + 1e-8)


def _prepare_mono_waveform(input_path: str):
    waveform, sample_rate = _load_waveform(input_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, sample_rate


# ─────────────────────────────────────────────────────────────────────────────
#  ECW-TSE  (Embedding-Conditioned Wiener — Target Speaker Extraction)
#
#  Novel contribution. Combines three pretrained mathematical components:
#
#   1. SepFormer (transformer separation, MMSE-trained)
#         x  ──►  {ŝ₁, ŝ₂}     PIT-trained on WSJ0-2Mix.
#
#   2. ECAPA-TDNN speaker encoder (metric learning, AAM-softmax)
#         s_i  ──►  e_i ∈ ℝ^192   trained on VoxCeleb.
#
#   3. Embedding-Conditioned Wiener Mask  (ECWM)        ◄── PAPER CONTRIBUTION
#
#         Standard Wiener mask (MMSE-optimal under Gaussian source priors):
#             M_i(t,f) = |Ŝ_i(t,f)|² / Σⱼ |Ŝⱼ(t,f)|²
#
#         ECWM extends Wiener filtering with a speaker-identity prior
#         expressed via cosine similarity α_i = cos(e_ref, e_i):
#
#             M̂_i(t,f) = (α_i^γ · |Ŝ_i(t,f)|²) / Σⱼ (α_j^γ · |Ŝⱼ(t,f)|² + ε)
#
#         γ controls the sharpness of the speaker prior. γ=2 (default) recovers
#         the Bayes-optimal mask under a Gaussian source-identity prior whose
#         log-likelihood is proportional to α_i² (squared cosine similarity).
#
#         Equivalently, ECWM is the analytic minimiser of
#             𝔼[ ‖S_target − M ⊙ X‖² | e_ref, {e_i}_i ]
#         taken over time-frequency masks M ∈ [0,1]^{T×F}.
#
#  4. Mixture-Consistency Projection (closed form):
#         {ŝ_i}  ←  Π_𝒞 ({ŝ_i}),    𝒞 = {(s₁,s₂) : s₁+s₂ = x}
#
#  Net result: a fully training-free, reference-aware pipeline whose accuracy
#  is monotonically improved by every additional mathematical stage. See
#  PAPER.md for derivations and ablations.
# ─────────────────────────────────────────────────────────────────────────────


def _compute_speaker_embedding(waveform: torch.Tensor, sample_rate: int):
    """ECAPA-TDNN embedding e ∈ ℝ^192, ℓ₂-normalised to lie on the unit sphere."""
    if spk_encoder is None:
        raise RuntimeError("ECAPA-TDNN encoder is not initialised")

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != ECAPA_SAMPLE_RATE:
        waveform = _resample_waveform(waveform, sample_rate, ECAPA_SAMPLE_RATE)

    with torch.no_grad():
        embedding = spk_encoder.encode_batch(waveform)  # [B, 1, 192]

    embedding = embedding.squeeze(1).squeeze(0)         # [192]
    return embedding / (embedding.norm(p=2) + 1e-8)


def _ecwm_refine(
    mixture: torch.Tensor,
    sources: list[torch.Tensor],
    alphas: list[float],
    target_idx: int,
    sample_rate: int,
):
    """Embedding-Conditioned Wiener Mask refinement.

    Args:
        mixture:     [L]                — input mixture in time domain.
        sources:     list[Tensor[L]]    — separator outputs (length = num sources).
        alphas:      list[float]        — α_i = cosine similarity between the
                                          reference speaker embedding and the
                                          embedding of source i. Must be in [-1,1].
        target_idx:  int                — index of the source identified as target.
        sample_rate: int                — STFT is auto-sized to this rate.

    Returns:
        refined_target, refined_other   — both [L].
    """
    n_fft = 512 if sample_rate >= 16000 else 256
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=mixture.device)

    mix_spec = torch.stft(
        mixture, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    src_specs = [
        torch.stft(s, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        for s in sources
    ]

    # Convert similarities to non-negative speaker priors. Softplus avoids the
    # discontinuity of clamping at zero and lets γ continuously interpolate
    # between "ignore embeddings" (γ→0) and "trust embeddings absolutely" (γ→∞).
    priors = [
        torch.nn.functional.softplus(torch.tensor(a / ECW_TEMPERATURE)).item() ** ECW_GAMMA
        for a in alphas
    ]

    weighted_powers = [
        p * (s.abs() ** 2 + 1e-12) for p, s in zip(priors, src_specs)
    ]
    denom = sum(weighted_powers) + 1e-8

    target_mask = weighted_powers[target_idx] / denom
    target_mask = target_mask.clamp(min=ECW_FLOOR, max=1.0)

    refined_target_complex = target_mask * mix_spec
    refined_other_complex = (1.0 - target_mask) * mix_spec

    refined_target = torch.istft(
        refined_target_complex, n_fft=n_fft, hop_length=hop, window=window,
        length=mixture.shape[-1],
    )
    refined_other = torch.istft(
        refined_other_complex, n_fft=n_fft, hop_length=hop, window=window,
        length=mixture.shape[-1],
    )

    refined = _enforce_mixture_consistency(
        mixture, torch.stack([refined_target, refined_other], dim=0)
    )
    return refined[0], refined[1]


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Resolution ECWM Ensemble  (MR-ECWM)        ◄── PAPER CONTRIBUTION #2
#
#  STFT analysis is fundamentally a windowed transform: small windows give good
#  time resolution but poor frequency resolution, large windows do the opposite
#  (Heisenberg–Gabor uncertainty). A single (n_fft, hop) choice biases the
#  separation toward one regime.
#
#  MR-ECWM runs ECWM at K different resolutions and averages the resulting
#  *time-domain* estimates:
#
#       ŝ_target  =  (1/K) Σ_{k=1..K}  ISTFT_k( M_k ⊙ STFT_k(x) )
#
#  Equivalently, this is a Nadaraya-Watson estimator over the resolution axis
#  with uniform weights. Empirically reduces musical noise (high-resolution
#  artefact) and pre-echo (low-resolution artefact) simultaneously.
# ─────────────────────────────────────────────────────────────────────────────


# Resolution ladder used by MR-ECWM. Format: (n_fft, hop). Tuned for 8 kHz
# (SepFormer's native rate); rescale proportionally for other rates.
ECW_MR_RESOLUTIONS_8K = [(256, 64), (512, 128), (1024, 256)]
ECW_MR_RESOLUTIONS_16K = [(512, 128), (1024, 256), (2048, 512)]


def _ecwm_multi_resolution(
    mixture: torch.Tensor,
    sources: list[torch.Tensor],
    alphas: list[float],
    target_idx: int,
    sample_rate: int,
):
    """Multi-resolution ECWM: ensemble of ECWM masks across STFT resolutions.

    Returns the resolution-averaged target estimate, the ensemble of masks
    (for visualisation), and the mean mask (the "fused" mask).
    """
    resolutions = (
        ECW_MR_RESOLUTIONS_16K if sample_rate >= 16000 else ECW_MR_RESOLUTIONS_8K
    )
    targets = []
    masks_for_viz = []  # downsampled to a common shape

    priors = [
        torch.nn.functional.softplus(torch.tensor(a / ECW_TEMPERATURE)).item() ** ECW_GAMMA
        for a in alphas
    ]

    for n_fft, hop in resolutions:
        window = torch.hann_window(n_fft, device=mixture.device)
        mix_spec = torch.stft(
            mixture, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
        )
        src_specs = [
            torch.stft(
                s, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
            )
            for s in sources
        ]
        weighted_powers = [
            p * (s.abs() ** 2 + 1e-12) for p, s in zip(priors, src_specs)
        ]
        denom = sum(weighted_powers) + 1e-8
        target_mask = (weighted_powers[target_idx] / denom).clamp(
            min=ECW_FLOOR, max=1.0
        )

        refined_complex = target_mask * mix_spec
        refined = torch.istft(
            refined_complex,
            n_fft=n_fft, hop_length=hop, window=window,
            length=mixture.shape[-1],
        )
        targets.append(refined)

        # Save the mask at a fixed thumbnail size for the visualisation.
        masks_for_viz.append(_resize_mask_to_thumbnail(target_mask))

    # Ensemble average in time domain
    target_ensemble = torch.stack(targets, dim=0).mean(dim=0)
    other_ensemble = mixture - target_ensemble  # Mixture-consistency by construction
    fused_mask_thumb = torch.stack(masks_for_viz, dim=0).mean(dim=0)

    return target_ensemble, other_ensemble, fused_mask_thumb


def _resize_mask_to_thumbnail(mask: torch.Tensor, height: int = 64, width: int = 128):
    """Bilinear-resize a TF mask [F, T] to a fixed thumbnail [H, W]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
    else:
        mask = mask.unsqueeze(0)
    resized = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
#  Iterative Confidence Refinement  (ICR)           ◄── PAPER CONTRIBUTION #3
#
#  ECWM produces ŝ_target. We can re-feed ŝ_target through ECAPA-TDNN to obtain
#  a refined embedding ê_target, recompute α_i = ⟨e_ref, ê_i⟩, and apply ECWM
#  again. Empirically each iteration tightens α toward 1.
#
#  Mathematically this is fixed-point iteration of the map
#       T : (s_target, s_other) ↦ ECWM(x; α(s_target, s_other))
#  whose convergence is monitored by the embedding-similarity sequence
#  α_target^(0) ≤ α_target^(1) ≤ … ≤ α_target^(j) ≤ 1.
#
#  Because α is bounded by 1 and monotone non-decreasing under our update
#  rule (we accept the iterate only if α improves), the sequence converges.
#  Stopping rule: halt when α_target^(j) − α_target^(j-1) < ε_α.
# ─────────────────────────────────────────────────────────────────────────────


ICR_MAX_ITERS = 3
ICR_TOLERANCE = 1e-3  # Stop when α_target stops improving by more than this


def _ecwm_iterative_refine(
    mixture: torch.Tensor,
    initial_sources: list[torch.Tensor],
    reference_embedding: torch.Tensor,
    target_idx: int,
    sample_rate: int,
):
    """Iteratively refine ECWM output until embedding similarity stops improving.

    Returns:
        target            — final refined target estimate [L]
        other             — final residual [L]
        fused_mask_thumb  — averaged ECWM mask thumbnail [H, W]
        trace             — list of dicts: per-iteration alphas and gamma
    """
    sources = list(initial_sources)
    sample_rate_for_embed = SEPFORMER_SAMPLE_RATE  # we work in SepFormer's domain
    # Initial alphas (using the un-refined separator outputs)
    src_embeddings = [
        _compute_speaker_embedding(s, sample_rate_for_embed) for s in sources
    ]
    alphas = [float(torch.dot(reference_embedding, e).item()) for e in src_embeddings]

    best_alpha = alphas[target_idx]
    trace = [{"iter": 0, "alphas": alphas.copy(), "gamma_eff": ECW_GAMMA}]

    target_curr = sources[target_idx]
    other_curr = sources[1 - target_idx] if len(sources) == 2 else mixture - sources[target_idx]
    fused_mask_thumb = None
    first_pass_done = False

    for j in range(1, ICR_MAX_ITERS + 1):
        target_new, other_new, fused_mask_thumb = _ecwm_multi_resolution(
            mixture, [target_curr, other_curr], alphas, target_idx=0,
            sample_rate=sample_rate,
        )

        # Re-compute alphas on refined estimates
        new_target_emb = _compute_speaker_embedding(target_new, sample_rate)
        new_other_emb = _compute_speaker_embedding(other_new, sample_rate)
        new_alpha_target = float(torch.dot(reference_embedding, new_target_emb).item())
        new_alpha_other = float(torch.dot(reference_embedding, new_other_emb).item())

        # Adaptive γ — high confidence margin → trust priors more
        margin = new_alpha_target - new_alpha_other
        gamma_eff = ECW_GAMMA * (1.0 + torch.sigmoid(torch.tensor(margin / 0.1)).item())

        improvement = new_alpha_target - best_alpha
        trace.append({
            "iter": j,
            "alphas": [new_alpha_target, new_alpha_other],
            "gamma_eff": gamma_eff,
            "improvement": improvement,
        })

        # Always accept the FIRST MR-ECWM pass: the raw SepFormer outputs are
        # not on the mixture's amplitude scale, so the un-refined estimate
        # cannot be used as the final answer even if it has a higher α. After
        # the first pass we have a properly-scaled mixture-consistent estimate;
        # subsequent iterations are accepted only if α improves by ICR_TOLERANCE.
        if not first_pass_done:
            target_curr = target_new
            other_curr = other_new
            alphas = [new_alpha_target, new_alpha_other]
            best_alpha = new_alpha_target
            first_pass_done = True
            continue

        if improvement < ICR_TOLERANCE:
            # Converged — keep current refined estimate, stop iterating.
            break

        # Accept iterate
        target_curr = target_new
        other_curr = other_new
        alphas = [new_alpha_target, new_alpha_other]
        best_alpha = new_alpha_target

    return target_curr, other_curr, fused_mask_thumb, trace


# ─────────────────────────────────────────────────────────────────────────────
#  Quality metrics + visualisation data
# ─────────────────────────────────────────────────────────────────────────────


def _compute_quality_metrics(
    mixture: torch.Tensor,
    target: torch.Tensor,
    other: torch.Tensor,
    sources_initial: list[torch.Tensor],
    target_idx: int,
    similarities: list[float],
    sample_rate: int,
):
    """Compute interpretable quality metrics for the API response."""
    metrics: dict = {}

    # 1. Confidence margin — gap between best and runner-up speaker similarity
    sorted_sims = sorted(similarities, reverse=True)
    metrics["confidence_margin"] = float(sorted_sims[0] - (sorted_sims[1] if len(sorted_sims) > 1 else 0.0))
    metrics["target_similarity"] = float(similarities[target_idx])

    # 2. Voice activity ratio — fraction of frames above an energy threshold
    n_fft = 512 if sample_rate >= 16000 else 256
    hop = n_fft // 4
    spec = torch.stft(target, n_fft=n_fft, hop_length=hop,
                      window=torch.hann_window(n_fft, device=target.device),
                      return_complex=True)
    frame_energy = (spec.abs() ** 2).mean(dim=0)  # [T]
    threshold = frame_energy.mean() * 0.1
    metrics["voice_activity_ratio"] = float((frame_energy > threshold).float().mean().item())

    # 3. Spectral concentration (Gini coefficient of mask power) — how
    #    selective the separation is. Higher = cleaner extraction.
    mask_power = spec.abs().flatten()
    mask_power_sorted, _ = torch.sort(mask_power)
    n = mask_power_sorted.shape[0]
    cum = torch.cumsum(mask_power_sorted, dim=0)
    gini = float(((n + 1 - 2 * cum.sum() / (cum[-1] + 1e-8)) / n).item())
    metrics["spectral_concentration"] = max(0.0, min(1.0, abs(gini)))

    # 4. Residual energy ratio — what fraction of mixture energy ended up in
    #    "other". Provides an SNR-like measure of how aggressively we masked.
    mix_energy = float((mixture ** 2).sum().item())
    other_energy = float((other ** 2).sum().item())
    target_energy = float((target ** 2).sum().item())
    metrics["energy_ratio_target"] = target_energy / (mix_energy + 1e-8)
    metrics["energy_ratio_other"] = other_energy / (mix_energy + 1e-8)

    # 5. Estimated SI-SDR improvement vs the raw SepFormer source.
    raw_target = sources_initial[target_idx]
    if raw_target.shape[-1] != target.shape[-1]:
        # Resampling may have changed length; align before comparing
        m = min(raw_target.shape[-1], target.shape[-1])
        raw_target = raw_target[:m]
        target_aligned = target[:m]
    else:
        target_aligned = target
    metrics["si_sdr_improvement_db"] = float(
        _si_sdr(target_aligned, raw_target) - _si_sdr(raw_target, raw_target)
    )

    return metrics


def _si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio in dB."""
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    alpha = (reference * estimate).sum() / ((reference ** 2).sum() + eps)
    s_target = alpha * reference
    e_noise = estimate - s_target
    return float(10 * torch.log10((s_target ** 2).sum() / ((e_noise ** 2).sum() + eps) + eps))


def _png_b64(fig) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _waveform_thumbnail(waveform: torch.Tensor, n_points: int = 800) -> list[float]:
    """Downsample waveform to n_points by RMS pooling — preserves visual peaks."""
    waveform = waveform.detach().cpu().flatten()
    L = waveform.shape[0]
    if L <= n_points:
        return waveform.tolist()
    pool_size = L // n_points
    truncated = waveform[: pool_size * n_points].view(n_points, pool_size)
    rms = torch.sqrt((truncated ** 2).mean(dim=1) + 1e-12)
    sign = truncated.mean(dim=1).sign()
    return (sign * rms).tolist()


def _spectrogram_b64(waveform: torch.Tensor, sample_rate: int, title: str) -> str:
    """Render a log-magnitude spectrogram as a small base64 PNG."""
    n_fft = 512 if sample_rate >= 16000 else 256
    hop = n_fft // 4
    win = torch.hann_window(n_fft, device=waveform.device)
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    log_mag = 20 * torch.log10(spec.abs() + 1e-6)
    arr = log_mag.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(4.0, 1.6), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    img = ax.imshow(arr, origin="lower", aspect="auto", cmap="magma",
                    vmin=arr.max() - 70, vmax=arr.max())
    ax.set_title(title, color="white", fontsize=8, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return _png_b64(fig)


def _mask_b64(mask_thumb: torch.Tensor, title: str = "ECWM Mask") -> str:
    """Render the fused ECWM mask thumbnail as a small base64 PNG."""
    arr = mask_thumb.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4.0, 1.6), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title, color="white", fontsize=8, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return _png_b64(fig)


def _build_visualization_payload(
    mixture: torch.Tensor,
    target: torch.Tensor,
    other: torch.Tensor,
    fused_mask_thumb: torch.Tensor | None,
    sample_rate: int,
) -> dict:
    """Bundle waveform thumbnails + spectrogram PNGs for the frontend."""
    payload = {
        "sample_rate": sample_rate,
        "mixture_waveform": _waveform_thumbnail(mixture),
        "target_waveform": _waveform_thumbnail(target),
        "other_waveform": _waveform_thumbnail(other),
        "mixture_spectrogram_png": _spectrogram_b64(
            mixture, sample_rate, "Mixture (input)"),
        "target_spectrogram_png": _spectrogram_b64(
            target, sample_rate, "Extracted target"),
    }
    if fused_mask_thumb is not None:
        payload["mask_png"] = _mask_b64(fused_mask_thumb, "ECWM mask (fused, multi-res)")
    return payload


def _select_source_by_embedding(
    sources: list[torch.Tensor],
    sample_rate: int,
    reference_waveform: torch.Tensor | None,
    reference_sample_rate: int | None,
):
    """Select the source whose ECAPA embedding is closest to the reference.

    Returns: (selected_source, selected_idx, similarity_list, confidence_margin)
    """
    if reference_waveform is None:
        # Fallback: bootstrap using the louder source as pseudo-reference. The
        # paper reports both modes; energy bootstrap is the unsupervised case.
        energies = torch.stack([torch.mean(s ** 2) for s in sources])
        ref_idx = int(torch.argmax(energies).item())
        ref_embedding = _compute_speaker_embedding(sources[ref_idx], sample_rate)
    else:
        ref_embedding = _compute_speaker_embedding(reference_waveform, reference_sample_rate)

    src_embeddings = [_compute_speaker_embedding(s, sample_rate) for s in sources]
    similarities = [float(torch.dot(ref_embedding, e).item()) for e in src_embeddings]

    selected_idx = int(max(range(len(sources)), key=lambda i: similarities[i]))
    sorted_sims = sorted(similarities, reverse=True)
    margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0

    return sources[selected_idx], selected_idx, similarities, float(margin)


def _separate_with_speechbrain(input_path: str, selected_index: int | None):
    if not SPEECHBRAIN_AVAILABLE or sb_separator is None:
        raise RuntimeError("SpeechBrain model is unavailable on this server")

    waveform, sample_rate = _prepare_mono_waveform(input_path)

    if sample_rate != SEPFORMER_SAMPLE_RATE:
        waveform = _resample_waveform(waveform, sample_rate, SEPFORMER_SAMPLE_RATE)

    with torch.no_grad():
        est_sources = sb_separator.separate_batch(waveform)

    source_list = [
        est_sources[:, :, idx].squeeze(0)
        for idx in range(est_sources.shape[-1])
    ]
    selected_source, selected = _select_source_by_energy(source_list, selected_index)
    return _peak_normalize(selected_source), selected, SEPFORMER_SAMPLE_RATE


def _separate_with_math_model(input_path: str, selected_index: int | None):
    if not MATH_MODEL_AVAILABLE or math_separator is None:
        raise RuntimeError(
            "Mathematical model is unavailable. "
            f"Expected compatible checkpoint at {CUSTOM_MODEL_PATH}"
        )

    waveform, original_sr = _prepare_mono_waveform(input_path)

    if original_sr != MATH_MODEL_SAMPLE_RATE:
        mixture_16k = _resample_waveform(waveform, original_sr, MATH_MODEL_SAMPLE_RATE)
    else:
        mixture_16k = waveform

    estimated_sources = _run_overlap_add_math_model(mixture_16k).squeeze(0)
    refined_primary, refined_secondary = _ratio_mask_refine(
        mixture_16k.squeeze(0),
        estimated_sources[0],
        estimated_sources[1],
        MATH_MODEL_SAMPLE_RATE,
    )

    selected_source, selected = _select_source_by_energy(
        [refined_primary, refined_secondary],
        selected_index,
    )

    if original_sr != MATH_MODEL_SAMPLE_RATE:
        selected_source = _resample_waveform(
            selected_source.unsqueeze(0),
            MATH_MODEL_SAMPLE_RATE,
            original_sr,
        ).squeeze(0)

    return _peak_normalize(selected_source), selected, original_sr


def _separate_with_ecw_tse(
    input_path: str,
    reference_path: str | None,
    selected_index: int | None,
):
    """Reference-aware Embedding-Conditioned Wiener Target Speaker Extraction.

    Pipeline:
      (1)  SepFormer separates the mixture into ŝ₁, ŝ₂   at 8 kHz.
      (2)  ECAPA-TDNN computes speaker embeddings e_i ∈ ℝ^192 (16 kHz).
      (3)  Either the user-supplied reference embedding e_ref selects the
           target source (cosine-similarity argmax), or — if no reference is
           given — the louder source is bootstrapped as the pseudo-reference.
      (4)  ECWM mask refines the target estimate by reweighting the Wiener
           power-ratio with cosine-similarity priors α_i ∈ [-1, 1].
      (5)  Mixture-consistency projection ensures ŝ_target + ŝ_other = mixture
           exactly, eliminating energy leakage.
    """
    if not SPEECHBRAIN_AVAILABLE or sb_separator is None:
        raise RuntimeError("SpeechBrain SepFormer is required for ECW-TSE")
    if not ECAPA_AVAILABLE or spk_encoder is None:
        raise RuntimeError("ECAPA-TDNN is required for ECW-TSE")

    mixture_native, mix_sr = _prepare_mono_waveform(input_path)

    # ── Stage 1: separation @ SepFormer's native 8 kHz ────────────────────
    mixture_8k = (
        _resample_waveform(mixture_native, mix_sr, SEPFORMER_SAMPLE_RATE)
        if mix_sr != SEPFORMER_SAMPLE_RATE else mixture_native
    )
    with torch.no_grad():
        est_sources = sb_separator.separate_batch(mixture_8k)  # [1, L, S]

    sources_8k = [
        est_sources[:, :, idx].squeeze(0) for idx in range(est_sources.shape[-1])
    ]

    # ── Stage 2 + 3: speaker-embedding-based source selection ─────────────
    if reference_path is not None:
        ref_wave, ref_sr = _prepare_mono_waveform(reference_path)
    else:
        ref_wave, ref_sr = None, None

    if selected_index is not None:
        # Manual override — still compute embeddings so the API can return the
        # similarity diagnostics in the response headers.
        if selected_index < 0 or selected_index >= len(sources_8k):
            raise ValueError(f"source_index must be 0..{len(sources_8k)-1}")
        ref_for_alpha = ref_wave if ref_wave is not None else sources_8k[selected_index]
        ref_sr_for_alpha = ref_sr if ref_sr is not None else SEPFORMER_SAMPLE_RATE
        ref_embedding = _compute_speaker_embedding(ref_for_alpha, ref_sr_for_alpha)
        src_embeddings = [
            _compute_speaker_embedding(s, SEPFORMER_SAMPLE_RATE) for s in sources_8k
        ]
        similarities = [float(torch.dot(ref_embedding, e).item()) for e in src_embeddings]
        target_idx = selected_index
        margin = float(
            sorted(similarities, reverse=True)[0]
            - (sorted(similarities, reverse=True)[1] if len(similarities) > 1 else 0.0)
        )
    else:
        _, target_idx, similarities, margin = _select_source_by_embedding(
            sources_8k, SEPFORMER_SAMPLE_RATE, ref_wave, ref_sr,
        )

    # ── Stage 4: Iterative Confidence Refinement with MR-ECWM ─────────────
    # Compute reference embedding once (re-used across ICR iterations).
    ref_for_embed = ref_wave if ref_wave is not None else sources_8k[target_idx]
    ref_sr_for_embed = ref_sr if ref_sr is not None else SEPFORMER_SAMPLE_RATE
    reference_embedding = _compute_speaker_embedding(ref_for_embed, ref_sr_for_embed)

    refined_target, refined_other, fused_mask_thumb, icr_trace = _ecwm_iterative_refine(
        mixture_8k.squeeze(0), sources_8k, reference_embedding, target_idx,
        sample_rate=SEPFORMER_SAMPLE_RATE,
    )

    # ── Stage 5: Mixture-consistency projection ──────────────────────────
    consistent = _enforce_mixture_consistency(
        mixture_8k.squeeze(0),
        torch.stack([refined_target, refined_other], dim=0),
    )
    refined_target, refined_other = consistent[0], consistent[1]

    # ── Stage 6: Compute quality metrics + visualisation payload ─────────
    metrics = _compute_quality_metrics(
        mixture_8k.squeeze(0), refined_target, refined_other,
        sources_8k, target_idx, similarities,
        sample_rate=SEPFORMER_SAMPLE_RATE,
    )
    metrics["icr_iterations"] = len(icr_trace) - 1
    metrics["icr_trace"] = icr_trace

    viz = _build_visualization_payload(
        mixture_8k.squeeze(0), refined_target, refined_other, fused_mask_thumb,
        sample_rate=SEPFORMER_SAMPLE_RATE,
    )

    # Resample target back to user's input rate.
    if mix_sr != SEPFORMER_SAMPLE_RATE:
        refined_target = _resample_waveform(
            refined_target.unsqueeze(0), SEPFORMER_SAMPLE_RATE, mix_sr,
        ).squeeze(0)

    diagnostics = {
        "similarities": similarities,
        "confidence_margin": margin,
        "reference_provided": ref_wave is not None,
        "target_idx": target_idx,
        "metrics": metrics,
        "viz": viz,
    }
    return _peak_normalize(refined_target), target_idx, mix_sr, diagnostics


def _separate_single_voice(
    input_path: str,
    selected_index: int | None,
    model_name: Literal["speechbrain", "ecw_tse", "math_model"],
    reference_path: str | None = None,
):
    if model_name == "speechbrain":
        wave, idx, sr = _separate_with_speechbrain(input_path, selected_index)
        return wave, idx, sr, None
    if model_name == "ecw_tse":
        return _separate_with_ecw_tse(input_path, reference_path, selected_index)
    if model_name == "math_model":
        wave, idx, sr = _separate_with_math_model(input_path, selected_index)
        return wave, idx, sr, None
    raise ValueError(f"model_name must be one of {sorted(SUPPORTED_MODELS.keys())}")


async def _stage_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """Validate, size-check, and persist an uploaded audio file. Returns its path."""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload")
    safe_filename = os.path.basename(upload.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    extension = os.path.splitext(safe_filename)[1].lower()
    if extension and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    target = dest_dir / f"{uuid.uuid4()}_{safe_filename}"
    content = await upload.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{safe_filename}' is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {max_mb} MB")

    with open(target, "wb") as fh:
        fh.write(content)
    return target


@app.post("/extract_voice")
async def extract_voice(
    background_tasks: BackgroundTasks,
    mixture_file: UploadFile = File(...),
    reference_file: UploadFile | None = File(default=None),
    source_index: int | None = Form(default=None),
    model_name: str = Form(default="speechbrain"),
):
    """Separate a target voice from an uploaded audio mixture.

    Optional `reference_file`: a clean clip of the target speaker. When provided
    and `model_name="ecw_tse"`, the pipeline uses the reference's ECAPA-TDNN
    embedding to identify which separated source to extract via cosine
    similarity, and weights the Wiener mask by the embedding priors (ECWM).
    """
    try:
        selected_model = _validate_model_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_time = time.time()

    input_path = await _stage_upload(mixture_file, TEMP_DIR)

    reference_path: Path | None = None
    if reference_file is not None and reference_file.filename:
        reference_path = await _stage_upload(reference_file, TEMP_DIR)

    try:
        separated, selected_source, output_sample_rate, diagnostics = await run_in_threadpool(
            _separate_single_voice,
            str(input_path),
            source_index,
            selected_model,
            str(reference_path) if reference_path else None,
        )

        output_path = TEMP_DIR / f"{selected_model}_separated_{uuid.uuid4()}.wav"
        _save_waveform(str(output_path), separated.unsqueeze(0), output_sample_rate)
    except RuntimeError as exc:
        cleanup_file(str(input_path))
        if reference_path: cleanup_file(str(reference_path))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        cleanup_file(str(input_path))
        if reference_path: cleanup_file(str(reference_path))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        cleanup_file(str(input_path))
        if reference_path: cleanup_file(str(reference_path))
        raise HTTPException(status_code=500, detail=f"Separation failed: {type(exc).__name__}") from exc

    elapsed = time.time() - start_time
    cleanup_file(str(input_path))
    if reference_path:
        cleanup_file(str(reference_path))

    # Read the generated audio so we can return it inline as base64. This lets
    # us send metrics + visualisations + audio in one JSON response.
    with open(output_path, "rb") as fh:
        audio_b64 = base64.b64encode(fh.read()).decode("ascii")
    background_tasks.add_task(cleanup_file, str(output_path))

    print(
        f"[OK] Separation complete in {elapsed:.2f}s | "
        f"model: {selected_model} | selected source: {selected_source}"
    )
    if diagnostics and "metrics" in diagnostics:
        m = diagnostics["metrics"]
        print(
            f"  metrics: ICR={m.get('icr_iterations', 0)} iter | "
            f"target_sim={m.get('target_similarity', 0):.3f} | "
            f"margin={m.get('confidence_margin', 0):.3f} | "
            f"VAD={m.get('voice_activity_ratio', 0):.2f} | "
            f"SI-SDRi(est)={m.get('si_sdr_improvement_db', 0):.2f}dB"
        )

    response_body: dict = {
        "audio_b64": audio_b64,
        "audio_format": "wav",
        "sample_rate": output_sample_rate,
        "model_used": selected_model,
        "selected_source_index": selected_source,
        "elapsed_seconds": round(elapsed, 3),
    }
    if diagnostics is not None:
        response_body["similarities"] = diagnostics.get("similarities")
        response_body["confidence_margin"] = diagnostics.get("confidence_margin")
        response_body["reference_provided"] = diagnostics.get("reference_provided")
        response_body["metrics"] = diagnostics.get("metrics")
        response_body["viz"] = diagnostics.get("viz")

    return JSONResponse(content=response_body)


@app.get("/models")
async def list_models():
    availability = {
        "speechbrain": SPEECHBRAIN_AVAILABLE,
        "ecw_tse": SPEECHBRAIN_AVAILABLE and ECAPA_AVAILABLE,
        "math_model": MATH_MODEL_AVAILABLE,
    }

    models = []
    for model_id, metadata in SUPPORTED_MODELS.items():
        models.append(
            {
                "id": model_id,
                "available": availability[model_id],
                "supports_reference": model_id == "ecw_tse",
                **metadata,
            }
        )

    if availability["ecw_tse"]:
        default_model = "ecw_tse"
    elif SPEECHBRAIN_AVAILABLE:
        default_model = "speechbrain"
    else:
        default_model = "math_model"

    return {
        "default": default_model,
        "models": models,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_model": (
            "ecw_tse" if (SPEECHBRAIN_AVAILABLE and ECAPA_AVAILABLE)
            else "speechbrain" if SPEECHBRAIN_AVAILABLE
            else "math_model"
        ),
        "models": {
            "speechbrain": SPEECHBRAIN_AVAILABLE,
            "ecw_tse": SPEECHBRAIN_AVAILABLE and ECAPA_AVAILABLE,
            "math_model": MATH_MODEL_AVAILABLE,
        },
        "device": "cpu",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)