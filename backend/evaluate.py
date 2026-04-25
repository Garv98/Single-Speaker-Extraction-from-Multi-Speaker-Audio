import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import os

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy
from model import TFGridNet

def compute_si_sdr(estimated, reference, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    The gold standard metric for speech separation evaluation.
    Higher is better. Units: dB.
    """
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)

    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    dot_product = torch.sum(reference * estimated, dim=-1, keepdim=True)

    alpha = dot_product / (ref_energy + eps)
    target_scaled = alpha * reference
    noise = estimated - target_scaled

    target_energy = torch.sum(target_scaled ** 2, dim=-1)
    noise_energy = torch.sum(noise ** 2, dim=-1)

    si_sdr = 10 * torch.log10(target_energy / (noise_energy + eps))
    return si_sdr.mean().item()


def compute_sdr(estimated, reference, eps=1e-8):
    """Standard Signal-to-Distortion Ratio (SDR). Units: dB."""
    noise = reference - estimated
    sdr = 10 * torch.log10(
        torch.sum(reference ** 2) / (torch.sum(noise ** 2) + eps)
    )
    return sdr.item()


def select_best_source(est_sources, reference):
    """Pick the estimated source with the highest SI-SDR against reference."""
    best_score = None
    best_idx = 0
    best_source = None

    for idx in range(est_sources.shape[-1]):
        candidate = est_sources[:, :, idx]
        score = compute_si_sdr(candidate, reference)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
            best_source = candidate

    return best_source, best_idx, best_score


def generate_publication_graphs():
    """
    Generates high-DPI publication-ready figures comparing:
    1. SOTA SepFormer separation
    2. Project Robust TF-GridNet model
    against a synthetic mixture.
    """
    print("=" * 60)
    print("  Evaluation Suite — Generating Publication Figures")
    print("=" * 60)

    sr = 8000  # SepFormer native rate
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create a realistic synthetic mixture
    # Speaker 1: multi-harmonic voice-like signal
    s1 = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)
    # Speaker 2: different frequency profile
    s2 = 0.5 * np.sin(2 * np.pi * 600 * t) + 0.2 * np.sin(2 * np.pi * 900 * t)
    # Mixture
    mixture = s1 + s2

    mix_tensor = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)
    s1_tensor = torch.tensor(s1, dtype=torch.float32).unsqueeze(0)

    # ─── SOTA SepFormer Inference ───
    print("\n[1/3] Running SOTA SepFormer inference...")
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="pretrained_models/sepformer-wsj02mix",
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cpu"}
    )

    with torch.no_grad():
        est_sources = separator.separate_batch(mix_tensor)
    sepformer_source, sepformer_idx, sf_si_sdr = select_best_source(est_sources, s1_tensor)
    sepformer_out = sepformer_source.squeeze(0)

    # ─── Project TF-GridNet Inference ───
    print("[2/3] Running project TF-GridNet inference...")
    # Resample to 16k for TF-GridNet
    mix_16k = torchaudio.transforms.Resample(8000, 16000)(mix_tensor)

    model_config = {
        "n_fft": 256,
        "in_channels": 16,
        "n_layers": 2,
        "num_sources": 2,
    }
    checkpoint_path = "best_tfgridnet.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            for key in model_config:
                if key in checkpoint["config"]:
                    model_config[key] = int(checkpoint["config"][key])

    gridnet = TFGridNet(**model_config)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            if any(k.startswith("module.") for k in checkpoint.keys()):
                checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
            gridnet.load_state_dict(checkpoint, strict=False)
            print(f"    Loaded trained checkpoint: {checkpoint_path}")
    else:
        print("    [WARN] No best_tfgridnet.pth found. Using randomly initialized model.")

    gridnet.eval()
    with torch.no_grad():
        gridnet_sources = gridnet(mix_16k, return_all_sources=True)

    resample_16k_to_8k = torchaudio.transforms.Resample(16000, 8000)
    gridnet_candidates_8k = []
    for idx in range(gridnet_sources.shape[1]):
        candidate = resample_16k_to_8k(gridnet_sources[:, idx, :]).squeeze(0)
        gridnet_candidates_8k.append(candidate)

    gridnet_scores = [
        compute_si_sdr(candidate.unsqueeze(0), s1_tensor)
        for candidate in gridnet_candidates_8k
    ]
    best_gridnet_idx = int(np.argmax(gridnet_scores))
    gridnet_out_8k = gridnet_candidates_8k[best_gridnet_idx].unsqueeze(0)

    # ─── Compute Metrics ───
    print("[3/3] Computing metrics...\n")

    # SepFormer metrics
    sf_si_sdr = compute_si_sdr(sepformer_out.unsqueeze(0), s1_tensor)
    sf_sdr = compute_sdr(sepformer_out.unsqueeze(0), s1_tensor)

    # TF-GridNet metrics
    gn_si_sdr = compute_si_sdr(gridnet_out_8k, s1_tensor)
    gn_sdr = compute_sdr(gridnet_out_8k, s1_tensor)

    print(f"  Selected SepFormer source index (best SI-SDR match): {sepformer_idx}")

    print(f"  Selected TF-GridNet source index (best SI-SDR match): {best_gridnet_idx}")

    print(f"  {'Metric':<15} {'SepFormer (SOTA)':<20} {'TF-GridNet (Project)'}")
    print(f"  {'─' * 55}")
    print(f"  {'SI-SDR (dB)':<15} {sf_si_sdr:<20.2f} {gn_si_sdr:.2f}")
    print(f"  {'SDR (dB)':<15} {sf_sdr:<20.2f} {gn_sdr:.2f}")

    os.makedirs("figures", exist_ok=True)

    # ─── FIGURE 1: Waveform Comparison ───
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), facecolor='#0a0e1a')
    signals = [
        (mixture, 'Input Mixture', '#888888'),
        (s1, 'Ground Truth (Speaker 1)', '#22c55e'),
        (sepformer_out.numpy(), f'SepFormer Output (SI-SDR: {sf_si_sdr:.1f} dB)', '#a855f7'),
        (gridnet_out_8k.squeeze(0).numpy(), f'TF-GridNet Project (SI-SDR: {gn_si_sdr:.1f} dB)', '#ef4444'),
    ]

    for i, (sig, title, color) in enumerate(signals):
        ax = axes[i]
        ax.set_facecolor('#0a0e1a')
        time_axis = np.linspace(0, len(sig) / sr, len(sig))
        ax.plot(time_axis, sig, color=color, linewidth=0.5, alpha=0.9)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.set_xlim(0, duration)
        ax.tick_params(colors='#555')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/figure_1_waveform_comparison.png", dpi=300, facecolor='#0a0e1a', bbox_inches='tight')
    plt.close()
    print("\n=> Saved: figures/figure_1_waveform_comparison.png")

    # ─── FIGURE 2: Spectrogram Comparison ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor='#0a0e1a')

    spec_data = [
        (mixture, 'Mixture', axes[0, 0]),
        (s1, 'Ground Truth', axes[0, 1]),
        (sepformer_out.numpy(), 'SepFormer Output', axes[1, 0]),
        (gridnet_out_8k.squeeze(0).numpy(), 'TF-GridNet (Project)', axes[1, 1]),
    ]

    for sig, title, ax in spec_data:
        ax.set_facecolor('#0a0e1a')
        _, _, _, img = ax.specgram(sig, NFFT=512, Fs=sr, noverlap=384, cmap='magma')
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(colors='#555')
        color_bar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
        color_bar.ax.tick_params(colors='#555')

    plt.tight_layout(pad=1.5)
    plt.savefig("figures/figure_2_spectrogram_comparison.png", dpi=300, facecolor='#0a0e1a', bbox_inches='tight')
    plt.close()
    print("=> Saved: figures/figure_2_spectrogram_comparison.png")

    # ─── FIGURE 3: Metrics Bar Chart ───
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    models = ['SepFormer\n(SOTA)', 'TF-GridNet\n(Project)']
    si_sdr_vals = [sf_si_sdr, gn_si_sdr]
    sdr_vals = [sf_sdr, gn_sdr]

    x = np.arange(len(models))
    width = 0.3

    bars1 = ax.bar(x - width/2, si_sdr_vals, width, label='SI-SDR (dB)', color='#a855f7', edgecolor='#a855f7', alpha=0.85)
    bars2 = ax.bar(x + width/2, sdr_vals, width, label='SDR (dB)', color='#22c55e', edgecolor='#22c55e', alpha=0.85)

    ax.set_ylabel('Score (dB)', color='white', fontsize=12)
    ax.set_title('Model Performance Comparison', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color='white', fontsize=11)
    ax.tick_params(colors='#555')
    ax.legend(facecolor='#1a1e2e', edgecolor='#333', labelcolor='white')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='#444', linewidth=0.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/figure_3_metrics_comparison.png", dpi=300, facecolor='#0a0e1a', bbox_inches='tight')
    plt.close()
    print("=> Saved: figures/figure_3_metrics_comparison.png")

    print("\n" + "=" * 60)
    print("  All publication figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    generate_publication_graphs()
