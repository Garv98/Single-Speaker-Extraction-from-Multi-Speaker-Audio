# Neural Voice Separation

A full-stack application for isolating individual voices from audio mixtures using speaker embeddings and novel masking techniques.

## Overview

**Frontend:** Next.js web interface for uploading audio and comparing separation results  
**Backend:** FastAPI service with three speaker separation models

## Models

- **SpeechBrain SepFormer** — WSJ0-2Mix baseline (energy-based selection)
- **ECW-TSE** — Novel embedding-conditioned pipeline with reference speaker support
- **Custom TF-GridNet** — Project-trained architecture on Libri2Mix with ratio-mask refinement

## Key Features

- **Reference-aware extraction** — optional speaker reference clip for targeted separation
- **Embedding-Conditioned Wiener Mask (ECWM)** — combines speaker similarity with spectral masking
- **Multi-Resolution ECWM** — reduces artifacts across STFT resolutions
- **Iterative Confidence Refinement (ICR)** — refines speaker identification through multiple passes
- **Visualizations** — waveforms, spectrograms, masks, and quality metrics

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Runs on `http://localhost:8000`. Models load on startup (SpeechBrain, ECAPA-TDNN, TF-GridNet).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:3000`.

## API

**POST `/extract_voice`** — Separate voice from mixture
- `mixture_file` — Audio file (WAV, FLAC, OGG; ≤25MB)
- `reference_file` — *(optional)* Clean speaker clip for ECW-TSE
- `model_name` — `speechbrain`, `ecw_tse`, or `math_model`
- `source_index` — *(optional)* Force selection of specific source (0 or 1)

**GET `/models`** — List available models and metadata  
**GET `/health`** — Health check

## Outputs

- Separated audio (base64-encoded WAV)
- Speaker similarity scores and confidence margins
- Quality metrics (SI-SDRi, voice activity ratio, spectral concentration)
- Visualizations: waveforms, spectrograms, ECWM mask

## Architecture

The **ECW-TSE** pipeline:
1. SepFormer separates mixture into two sources @ 8 kHz
2. ECAPA-TDNN computes 192-dim speaker embeddings @ 16 kHz
3. Reference embedding selects target source via cosine similarity
4. ECWM refines with embedding-weighted Wiener masking
5. Mixture-consistency projection enforces `s_target + s_other = mixture`
6. ICR optionally refines over multiple passes

See `backend/main.py` for mathematical derivations and hyperparameters.

## Requirements

- Python 3.9+
- PyTorch, torchaudio
- FastAPI, SpeechBrain
- Next.js 16+, React 19+
