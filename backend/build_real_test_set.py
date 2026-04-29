"""Build a real-speech 2-speaker test set for paper evaluation.

The bundled `data/Libri2Mix/` mini-set is synthetic sine waves and is useless
for benchmarking SepFormer/ECAPA, which were trained on speech. This script
pulls real LibriSpeech `test-clean` utterances via the Hugging Face
datasets-server REST API, picks one utterance per *distinct* speaker, slices
each into 4-second segments, and pairs cross-speaker segments at 0 dB SNR to
form 2-speaker mixtures. The resulting CSV is drop-in compatible with
`evaluate_paper.py`.

Reproducible (numpy seed=42). Saves to `data/RealLibri2Mix/test/`.
"""
from __future__ import annotations

import csv
import io
import itertools
import os
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio


HF_API = "https://datasets-server.huggingface.co/rows"
DATASET = "openslr/librispeech_asr"
CONFIG = "clean"
SPLIT = "test"
NUM_SPEAKERS = 6                 # 6 speakers -> C(6,2)=15 cross-speaker pairs
TARGET_SR = 16_000               # All audio at 16 kHz before mixing
SEGMENT_DURATION = 4.0           # Seconds per source clip in the mixture
MIN_AUDIO_SECONDS = 4.5          # Reject clips that are too short for SEGMENT_DURATION
NUM_MIXTURES = 10                # Cap output mixtures
RNG_SEED = 42


def _api_rows(offset: int, length: int = 100) -> list[dict]:
    params = {
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "offset": offset, "length": length,
    }
    r = requests.get(HF_API, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("rows", [])


def _download_audio(url: str) -> tuple[np.ndarray, int]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    audio, sr = sf.read(io.BytesIO(r.content), always_2d=True)
    waveform = torch.from_numpy(audio.T).to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0).numpy(), TARGET_SR


def _find_distinct_speakers() -> dict[str, np.ndarray]:
    """Walk the dataset until we've collected N distinct speaker recordings."""
    speakers: dict[str, np.ndarray] = {}
    offset = 0
    seen_speakers: set[int] = set()

    print(f"  Searching for {NUM_SPEAKERS} distinct speakers in {DATASET}/{CONFIG}/{SPLIT}...")
    while len(speakers) < NUM_SPEAKERS and offset < 3000:
        try:
            rows = _api_rows(offset=offset, length=100)
        except Exception as exc:
            print(f"    [warn] offset {offset}: {exc}")
            offset += 100
            continue
        if not rows:
            break

        for row in rows:
            rd = row["row"]
            spk = int(rd["speaker_id"])
            if spk in seen_speakers:
                continue

            audio_field = rd.get("audio")
            url = None
            if isinstance(audio_field, list) and audio_field:
                url = audio_field[0].get("src")
            elif isinstance(audio_field, dict):
                url = audio_field.get("src")
            if not url:
                continue

            try:
                audio, sr = _download_audio(url)
            except Exception as exc:
                print(f"    [warn] spk {spk} download failed: {type(exc).__name__}: {exc}")
                continue

            duration = audio.shape[-1] / sr
            if duration < MIN_AUDIO_SECONDS:
                continue

            spk_label = f"spk{spk}"
            speakers[spk_label] = audio
            seen_speakers.add(spk)
            print(f"    [{len(speakers)}/{NUM_SPEAKERS}] {spk_label}: {duration:.2f}s @ {sr} Hz")
            if len(speakers) >= NUM_SPEAKERS:
                break

        offset += 100

    if len(speakers) < 2:
        raise RuntimeError(
            f"Only got {len(speakers)} distinct speakers. Cannot build 2-speaker mixtures."
        )
    return speakers


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _scale_to_rms(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    return x * (target_rms / (_rms(x) + 1e-12))


def main() -> None:
    base = Path(__file__).resolve().parent
    out_root = base / "data" / "RealLibri2Mix" / "test"
    (out_root / "mix_clean").mkdir(parents=True, exist_ok=True)
    (out_root / "s1").mkdir(parents=True, exist_ok=True)
    (out_root / "s2").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=RNG_SEED)

    # ── Step 1: collect distinct-speaker utterances ────────────────────────
    speakers = _find_distinct_speakers()
    spk_ids = sorted(speakers.keys())

    # Crop each utterance to SEGMENT_DURATION (mid-clip if longer)
    seg_len = int(SEGMENT_DURATION * TARGET_SR)
    segments: dict[str, np.ndarray] = {}
    for spk, audio in speakers.items():
        if audio.shape[-1] < seg_len:
            audio = np.pad(audio, (0, seg_len - audio.shape[-1]))
        else:
            mid = (audio.shape[-1] - seg_len) // 2
            audio = audio[mid:mid + seg_len]
        segments[spk] = audio.astype(np.float32)

    # ── Step 2: cross-speaker pairs ────────────────────────────────────────
    pairs = list(itertools.combinations(spk_ids, 2))
    if len(pairs) > NUM_MIXTURES:
        idxs = rng.choice(len(pairs), size=NUM_MIXTURES, replace=False)
        pairs = [pairs[i] for i in sorted(idxs)]

    print(f"\n  Building {len(pairs)} mixtures from {len(speakers)} speakers...")

    rows: list[list[str]] = [["mixture_path", "source_1_path", "source_2_path", "spk1", "spk2"]]
    for k, (spkA, spkB) in enumerate(pairs):
        s1 = _scale_to_rms(segments[spkA], target_rms=0.1)
        s2 = _scale_to_rms(segments[spkB], target_rms=0.1)
        n = min(s1.shape[-1], s2.shape[-1])
        s1, s2 = s1[:n], s2[:n]
        mix = s1 + s2

        mix_path = out_root / "mix_clean" / f"mix_{k}.wav"
        s1_path = out_root / "s1" / f"s1_{k}.wav"
        s2_path = out_root / "s2" / f"s2_{k}.wav"
        sf.write(str(mix_path), mix, TARGET_SR)
        sf.write(str(s1_path), s1, TARGET_SR)
        sf.write(str(s2_path), s2, TARGET_SR)

        rows.append([
            os.path.abspath(str(mix_path)),
            os.path.abspath(str(s1_path)),
            os.path.abspath(str(s2_path)),
            spkA, spkB,
        ])
        print(f"    mix_{k}: {spkA} + {spkB}")

    csv_path = out_root / "test.csv"
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\n  Wrote {csv_path}")
    print(f"  Speakers used: {spk_ids}")


if __name__ == "__main__":
    main()
