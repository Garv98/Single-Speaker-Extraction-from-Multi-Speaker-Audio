import os
import time
import uuid
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from speechbrain.inference.separation import SepformerSeparation
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
CUSTOM_MODEL_PATH = BASE_DIR / "best_tfgridnet.pth"

TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}

SEPFORMER_SAMPLE_RATE = 8000
MATH_MODEL_SAMPLE_RATE = 16000

SUPPORTED_MODELS = {
    "speechbrain": {
        "label": "SpeechBrain SepFormer (SOTA)",
        "architecture": "SepFormer",
        "dataset": "WSJ0-2Mix",
        "benchmark": "22.3 dB SI-SDRi",
    },
    "math_model": {
        "label": "Robust TF-GridNet (Mathematical)",
        "architecture": "TF-GridNet + ratio-mask refinement",
        "dataset": "Libri2Mix (project) + synthetic overlaps",
        "benchmark": "Project-trained",
    },
}

TORCHAUDIO_IO_AVAILABLE = True
TORCHAUDIO_SAVE_AVAILABLE = True
SPEECHBRAIN_AVAILABLE = False
MATH_MODEL_AVAILABLE = False

sb_separator = None
math_separator = None


def _load_models():
    global sb_separator, math_separator, SPEECHBRAIN_AVAILABLE, MATH_MODEL_AVAILABLE

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
        print("  Loading project TF-GridNet checkpoint...")
        if not CUSTOM_MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found at {CUSTOM_MODEL_PATH}")

        checkpoint = torch.load(CUSTOM_MODEL_PATH, map_location="cpu")

        model_config = {
            "n_fft": 256,
            "in_channels": 16,
            "n_layers": 2,
            "num_sources": 2,
        }

        if isinstance(checkpoint, dict):
            config = checkpoint.get("config")
            if isinstance(config, dict):
                for key in model_config:
                    if key in config:
                        model_config[key] = int(config[key])

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

        math_separator.load_state_dict(checkpoint, strict=False)
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


def _validate_model_name(model_name: str) -> Literal["speechbrain", "math_model"]:
    normalized = model_name.strip().lower()
    if normalized == "speechbrain":
        return "speechbrain"
    if normalized == "math_model":
        return "math_model"
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
    return refined_primary, refined_residual


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
        raise RuntimeError("Mathematical model is unavailable. Train and save best_tfgridnet.pth first")

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


def _separate_single_voice(
    input_path: str,
    selected_index: int | None,
    model_name: Literal["speechbrain", "math_model"],
):
    if model_name == "speechbrain":
        return _separate_with_speechbrain(input_path, selected_index)
    if model_name == "math_model":
        return _separate_with_math_model(input_path, selected_index)
    raise ValueError(f"model_name must be one of {sorted(SUPPORTED_MODELS.keys())}")


@app.post("/extract_voice")
async def extract_voice(
    mixture_file: UploadFile = File(...),
    source_index: int | None = Form(default=None),
    model_name: str = Form(default="speechbrain"),
    background_tasks: BackgroundTasks,
):
    """Separate a dominant voice from an uploaded audio mixture."""
    try:
        selected_model = _validate_model_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_time = time.time()

    if not mixture_file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload")

    safe_filename = os.path.basename(mixture_file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    extension = os.path.splitext(safe_filename)[1].lower()
    if extension and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    input_path = TEMP_DIR / f"{uuid.uuid4()}_{safe_filename}"
    content = await mixture_file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {max_mb} MB")

    with open(input_path, "wb") as file_handle:
        file_handle.write(content)

    try:
        separated, selected_source, output_sample_rate = await run_in_threadpool(
            _separate_single_voice,
            str(input_path),
            source_index,
            selected_model,
        )

        output_path = TEMP_DIR / f"{selected_model}_separated_{uuid.uuid4()}.wav"
        _save_waveform(str(output_path), separated.unsqueeze(0), output_sample_rate)
    except RuntimeError as exc:
        cleanup_file(str(input_path))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        cleanup_file(str(input_path))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        cleanup_file(str(input_path))
        raise HTTPException(status_code=500, detail=f"Separation failed: {type(exc).__name__}") from exc

    elapsed = time.time() - start_time
    cleanup_file(str(input_path))
    background_tasks.add_task(cleanup_file, str(output_path))

    print(
        f"[OK] Separation complete in {elapsed:.2f}s | "
        f"model: {selected_model} | selected source: {selected_source}"
    )

    return FileResponse(
        str(output_path),
        media_type="audio/wav",
        headers={
            "X-Selected-Source-Index": str(selected_source),
            "X-Model-Name": selected_model,
            "X-Output-Sample-Rate": str(output_sample_rate),
        },
    )


@app.get("/models")
async def list_models():
    availability = {
        "speechbrain": SPEECHBRAIN_AVAILABLE,
        "math_model": MATH_MODEL_AVAILABLE,
    }

    models = []
    for model_id, metadata in SUPPORTED_MODELS.items():
        models.append(
            {
                "id": model_id,
                "available": availability[model_id],
                **metadata,
            }
        )

    default_model = "speechbrain" if SPEECHBRAIN_AVAILABLE else "math_model"
    return {
        "default": default_model,
        "models": models,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_model": "speechbrain" if SPEECHBRAIN_AVAILABLE else "math_model",
        "models": {
            "speechbrain": SPEECHBRAIN_AVAILABLE,
            "math_model": MATH_MODEL_AVAILABLE,
        },
        "device": "cpu",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)