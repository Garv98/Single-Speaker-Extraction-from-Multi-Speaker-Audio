"""Real, ground-truth-anchored evaluation harness for the ECW-TSE paper.

Runs every pipeline variant on the entire Libri2Mix mini-set, computes
permutation-invariant SI-SDR and SDR against the clean reference sources, and
dumps both per-mixture and aggregate statistics to JSON. The paper's
Methodology and Results sections cite the numbers in `paper_results.json`
directly — no hand-tuned values anywhere.

Run:
    cd backend
    python evaluate_paper.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import pandas as pd
import soundfile as sf
import torch
import torchaudio

import main as backend  # Imports trigger model loading once


# ────────────────────────────────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────────────────────────────────


def _si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    alpha = (reference * estimate).sum() / ((reference ** 2).sum() + eps)
    target = alpha * reference
    noise = estimate - target
    return float(10 * torch.log10((target ** 2).sum() / ((noise ** 2).sum() + eps) + eps))


def _sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    noise = reference - estimate
    return float(10 * torch.log10((reference ** 2).sum() / ((noise ** 2).sum() + eps) + eps))


def _pit_score(
    estimate: torch.Tensor,
    refs: list[torch.Tensor],
    metric: Callable[[torch.Tensor, torch.Tensor], float],
) -> tuple[float, int]:
    """Permutation-invariant: pick the reference that yields the best metric."""
    scores = [metric(estimate, r) for r in refs]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return scores[best_idx], best_idx


# ────────────────────────────────────────────────────────────────────────────
#  Audio I/O helpers (work in SepFormer's 8 kHz domain consistently)
# ────────────────────────────────────────────────────────────────────────────


def _load_mono(path: str | Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(audio.T).to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def _to_8k(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == backend.SEPFORMER_SAMPLE_RATE:
        return waveform.squeeze(0)
    return torchaudio.functional.resample(
        waveform, sr, backend.SEPFORMER_SAMPLE_RATE
    ).squeeze(0)


def _align_lengths(*tensors: torch.Tensor) -> list[torch.Tensor]:
    n = min(t.shape[-1] for t in tensors)
    return [t[..., :n] for t in tensors]


# ────────────────────────────────────────────────────────────────────────────
#  Variants under test
# ────────────────────────────────────────────────────────────────────────────


def _variant_sepformer_energy(mix_path: str, ref_path: str | None) -> torch.Tensor:
    """Baseline 1 — SepFormer + energy-based source selection (no embedding, no ECWM)."""
    wave, idx, sr = backend._separate_with_speechbrain(mix_path, selected_index=None)
    return wave  # already 8 kHz, peak-normalised


def _variant_sepformer_embed(mix_path: str, ref_path: str | None) -> torch.Tensor:
    """Baseline 2 — SepFormer + ECAPA cosine-similarity selection (no mask reweighting).

    Isolates the contribution of *embedding-based selection alone*, so that the
    paper can attribute downstream gains to the ECWM mask itself.
    """
    if ref_path is None:
        return _variant_sepformer_energy(mix_path, ref_path)

    mixture_native, mix_sr = backend._prepare_mono_waveform(mix_path)
    mixture_8k = (
        backend._resample_waveform(mixture_native, mix_sr, backend.SEPFORMER_SAMPLE_RATE)
        if mix_sr != backend.SEPFORMER_SAMPLE_RATE else mixture_native
    )
    with torch.no_grad():
        est = backend.sb_separator.separate_batch(mixture_8k)
    sources_8k = [est[:, :, i].squeeze(0) for i in range(est.shape[-1])]

    ref_wave, ref_sr = backend._prepare_mono_waveform(ref_path)
    ref_emb = backend._compute_speaker_embedding(ref_wave, ref_sr)
    src_embs = [
        backend._compute_speaker_embedding(s, backend.SEPFORMER_SAMPLE_RATE)
        for s in sources_8k
    ]
    sims = [float(torch.dot(ref_emb, e).item()) for e in src_embs]
    target_idx = max(range(len(sims)), key=lambda i: sims[i])
    return backend._peak_normalize(sources_8k[target_idx])


def _variant_ecw_tse(mix_path: str, ref_path: str | None) -> tuple[torch.Tensor, dict]:
    """ECW-TSE full pipeline (MR-ECWM + ICR + mixture-consistency)."""
    wave, idx, sr, diag = backend._separate_with_ecw_tse(
        mix_path, ref_path, selected_index=None
    )
    if sr != backend.SEPFORMER_SAMPLE_RATE:
        wave = backend._resample_waveform(
            wave.unsqueeze(0), sr, backend.SEPFORMER_SAMPLE_RATE
        ).squeeze(0)
    return wave, diag


def _variant_math_model(mix_path: str, ref_path: str | None) -> torch.Tensor:
    """Project's custom TF-GridNet."""
    wave, idx, sr = backend._separate_with_math_model(mix_path, selected_index=None)
    if sr != backend.SEPFORMER_SAMPLE_RATE:
        wave = backend._resample_waveform(
            wave.unsqueeze(0), sr, backend.SEPFORMER_SAMPLE_RATE
        ).squeeze(0)
    return wave


# ────────────────────────────────────────────────────────────────────────────
#  Driver
# ────────────────────────────────────────────────────────────────────────────


def evaluate(csv_path: str | Path, out_path: str | Path) -> dict:
    df = pd.read_csv(csv_path)
    n_mixtures = len(df)

    pipelines: list[str] = []
    if backend.SPEECHBRAIN_AVAILABLE:
        pipelines += ["sepformer_energy", "sepformer_embed_select", "ecw_tse_blind"]
        if backend.ECAPA_AVAILABLE:
            pipelines += ["ecw_tse_with_ref"]
    if backend.MATH_MODEL_AVAILABLE:
        pipelines += ["tf_gridnet"]

    print(f"\n[evaluate_paper] {n_mixtures} mixtures × {len(pipelines)} pipelines")
    print(f"[evaluate_paper] Pipelines: {pipelines}")

    per_mix: list[dict] = []

    for row_idx, row in df.iterrows():
        mix_path = row["mixture_path"]
        s1_path = row["source_1_path"]
        s2_path = row["source_2_path"]

        # Ground-truth references in SepFormer's 8 kHz domain
        s1_w, sr1 = _load_mono(s1_path)
        s2_w, sr2 = _load_mono(s2_path)
        s1_8k = _to_8k(s1_w, sr1)
        s2_8k = _to_8k(s2_w, sr2)
        mix_w, mixsr = _load_mono(mix_path)
        mix_8k = _to_8k(mix_w, mixsr)

        # Mixture baseline (no separation at all). Use a single common-length
        # alignment so SI-SDRi is computed against the same mixture each pipeline saw.
        s1_align, s2_align, mix_align = _align_lengths(s1_8k, s2_8k, mix_8k)
        mixture_si_sdr_vs_s1 = _si_sdr(mix_align, s1_align)
        mixture_si_sdr_vs_s2 = _si_sdr(mix_align, s2_align)

        mix_record: dict = {
            "index": int(row_idx),
            "mix_path": Path(mix_path).name,
            "mixture_si_sdr_vs_s1_db": mixture_si_sdr_vs_s1,
            "mixture_si_sdr_vs_s2_db": mixture_si_sdr_vs_s2,
            "pipelines": {},
        }

        for name in pipelines:
            t0 = time.time()
            try:
                if name == "sepformer_energy":
                    out = _variant_sepformer_energy(mix_path, None)
                    diag = {}
                elif name == "sepformer_embed_select":
                    # Use s1 as the speaker reference (paper's reference-aware setting)
                    out = _variant_sepformer_embed(mix_path, s1_path)
                    diag = {}
                elif name == "ecw_tse_blind":
                    out, diag = _variant_ecw_tse(mix_path, None)
                elif name == "ecw_tse_with_ref":
                    out, diag = _variant_ecw_tse(mix_path, s1_path)
                elif name == "tf_gridnet":
                    out = _variant_math_model(mix_path, None)
                    diag = {}
                else:
                    continue
            except Exception as exc:
                print(f"  [{row_idx}] {name}: FAILED ({type(exc).__name__}: {exc})")
                mix_record["pipelines"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue

            elapsed = time.time() - t0

            # Align everything (output, both refs, mixture) to a common length.
            out_8k, s1_a, s2_a, mix_a = _align_lengths(out, s1_8k, s2_8k, mix_8k)

            # Permutation-invariant scoring against [s1, s2]
            si_sdr_pit, perm_idx = _pit_score(out_8k, [s1_a, s2_a], _si_sdr)
            sdr_pit, _ = _pit_score(out_8k, [s1_a, s2_a], _sdr)
            chosen_ref = [s1_a, s2_a][perm_idx]
            # SI-SDRi: improvement over the un-separated mixture, evaluated against
            # the SAME ground-truth source the estimate was matched to.
            si_sdr_improvement = si_sdr_pit - _si_sdr(mix_a, chosen_ref)

            entry = {
                "si_sdr_db": si_sdr_pit,
                "sdr_db": sdr_pit,
                "si_sdri_db": si_sdr_improvement,  # vs. mixture
                "matched_speaker": "s1" if perm_idx == 0 else "s2",
                "elapsed_s": round(elapsed, 3),
            }
            # Add ECW-TSE-specific diagnostics
            if "metrics" in diag:
                m = diag["metrics"]
                entry.update({
                    "icr_iterations": int(m.get("icr_iterations", 0)),
                    "target_similarity": float(m.get("target_similarity", 0.0)),
                    "confidence_margin": float(m.get("confidence_margin", 0.0)),
                    "voice_activity_ratio": float(m.get("voice_activity_ratio", 0.0)),
                    "spectral_concentration": float(m.get("spectral_concentration", 0.0)),
                    "energy_ratio_target": float(m.get("energy_ratio_target", 0.0)),
                    "energy_ratio_other": float(m.get("energy_ratio_other", 0.0)),
                })
                trace = m.get("icr_trace", [])
                if trace:
                    entry["icr_alpha_trace"] = [
                        float(t["alphas"][0]) if isinstance(t.get("alphas"), list) and t["alphas"]
                        else None
                        for t in trace
                    ]

            mix_record["pipelines"][name] = entry
            print(
                f"  [{row_idx}] {name:24s} SI-SDR={si_sdr_pit:6.2f} dB | "
                f"SI-SDRi={si_sdr_improvement:5.2f} dB | t={elapsed:5.2f}s"
            )

        per_mix.append(mix_record)

    # Aggregate
    aggregates: dict = {}
    for name in pipelines:
        si_sdrs, sdrs, si_sdris, times = [], [], [], []
        icr_iters, target_sims, margins = [], [], []
        for rec in per_mix:
            entry = rec["pipelines"].get(name)
            if not entry or "error" in entry:
                continue
            si_sdrs.append(entry["si_sdr_db"])
            sdrs.append(entry["sdr_db"])
            si_sdris.append(entry["si_sdri_db"])
            times.append(entry["elapsed_s"])
            if "icr_iterations" in entry:
                icr_iters.append(entry["icr_iterations"])
                target_sims.append(entry["target_similarity"])
                margins.append(entry["confidence_margin"])
        if not si_sdrs:
            aggregates[name] = {"n": 0}
            continue
        agg = {
            "n": len(si_sdrs),
            "si_sdr_db_mean": mean(si_sdrs),
            "si_sdr_db_std": pstdev(si_sdrs),
            "sdr_db_mean": mean(sdrs),
            "sdr_db_std": pstdev(sdrs),
            "si_sdri_db_mean": mean(si_sdris),
            "si_sdri_db_std": pstdev(si_sdris),
            "elapsed_s_mean": mean(times),
        }
        if icr_iters:
            agg.update({
                "icr_iterations_mean": mean(icr_iters),
                "target_similarity_mean": mean(target_sims),
                "confidence_margin_mean": mean(margins),
            })
        aggregates[name] = agg

    results = {
        "n_mixtures": n_mixtures,
        "sample_rate_hz": backend.SEPFORMER_SAMPLE_RATE,
        "pipelines": pipelines,
        "aggregates": aggregates,
        "per_mixture": per_mix,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\n[evaluate_paper] Aggregates:")
    for name, agg in aggregates.items():
        if agg.get("n", 0) == 0:
            print(f"  {name:24s}  (no successful runs)")
            continue
        line = (
            f"  {name:24s}  SI-SDR = {agg['si_sdr_db_mean']:6.2f} ± "
            f"{agg['si_sdr_db_std']:4.2f} dB | "
            f"SI-SDRi = {agg['si_sdri_db_mean']:5.2f} dB | "
            f"t = {agg['elapsed_s_mean']:5.2f}s"
        )
        if "icr_iterations_mean" in agg:
            line += (
                f" | ICR={agg['icr_iterations_mean']:.2f} | "
                f"alpha={agg['target_similarity_mean']:.3f} | "
                f"margin={agg['confidence_margin_mean']:.3f}"
            )
        print(line)

    print(f"\n[evaluate_paper] Wrote {out_path}")
    return results


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent
    # Default to the real-speech test set; fall back to legacy synthetic only if explicitly requested.
    if len(sys.argv) > 1 and sys.argv[1] == "synthetic":
        csv_path = base / "data" / "Libri2Mix" / "train" / "train.csv"
        out_path = base / "paper_results_synthetic.json"
    else:
        csv_path = base / "data" / "RealLibri2Mix" / "test" / "test.csv"
        out_path = base / "paper_results.json"
    evaluate(csv_path, out_path)
