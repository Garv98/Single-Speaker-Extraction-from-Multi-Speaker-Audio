import random
import csv
import time
from pathlib import Path

import matplotlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler

from dataset import Libri2MixDataset
from model import TFGridNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACT_DIR   = Path("figures") / "training"
CSV_LOG_PATH   = ARTIFACT_DIR / "training_history.csv"
CURVES_PATH    = ARTIFACT_DIR / "training_curves.png"
BEST_CKPT      = "best_tfgridnet.pth"
LAST_CKPT      = "last_tfgridnet.pth"   # Saved every epoch — resume here after Colab disconnect

# ── Hyper-parameters ───────────────────────────────────────────────────────────
MODEL_CONFIG = dict(
    n_fft=512,
    hop_length=128,
    d_model=64,
    n_heads=4,
    lstm_hidden=256,   # Reduce to 128 if Colab runs out of VRAM
    n_layers=6,        # Reduce to 4 for lighter GPU budget
    num_sources=2,
    dropout=0.1,
)

TRAIN_CONFIG = dict(
    learning_rate=1e-3,
    epochs=80,
    batch_size=4,       # Reduce to 2 if VRAM is tight
    stft_weight=0.3,
    warmup_epochs=5,
    early_stop_patience=12,
    min_delta=1e-3,
    chunk_duration=3.0,
    num_workers=4,      # Set to 0 on Windows; 4 works well in Colab
    grad_clip=5.0,
)

CSV_FIELDS = [
    "epoch", "train_loss", "train_pit", "train_stft",
    "val_loss", "val_pit", "val_stft",
    "lr", "best_val_loss", "is_best", "no_improve_epochs", "epoch_seconds",
]


# ── Logging helpers ────────────────────────────────────────────────────────────

def initialize_csv_logger(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_log(csv_path: Path, row: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def save_training_curves(history: dict, curves_path: Path):
    epochs = history["epoch"]
    if not epochs:
        return
    curves_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    ax_total, ax_pit, ax_stft, ax_lr = axes.flat

    ax_total.plot(epochs, history["train_loss"], label="Train",      color="#1f77b4", linewidth=2)
    ax_total.plot(epochs, history["val_loss"],   label="Validation", color="#ff7f0e", linewidth=2)
    ax_total.set_title("Total Loss"); ax_total.set_xlabel("Epoch"); ax_total.grid(alpha=0.25); ax_total.legend()

    ax_pit.plot(epochs, history["train_pit"], label="Train PIT",      color="#2ca02c", linewidth=2)
    ax_pit.plot(epochs, history["val_pit"],   label="Validation PIT", color="#d62728", linewidth=2)
    ax_pit.set_title("PIT SI-SDR Loss"); ax_pit.set_xlabel("Epoch"); ax_pit.grid(alpha=0.25); ax_pit.legend()

    ax_stft.plot(epochs, history["train_stft"], label="Train STFT",      color="#9467bd", linewidth=2)
    ax_stft.plot(epochs, history["val_stft"],   label="Validation STFT", color="#8c564b", linewidth=2)
    ax_stft.set_title("MR-STFT Loss"); ax_stft.set_xlabel("Epoch"); ax_stft.grid(alpha=0.25); ax_stft.legend()

    ax_lr.plot(epochs, history["lr"], label="Learning Rate", color="#17becf", linewidth=2)
    ax_lr.set_title("Learning Rate Schedule"); ax_lr.set_xlabel("Epoch"); ax_lr.grid(alpha=0.25); ax_lr.legend()

    if history["val_loss"]:
        best_idx = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
        for ax in axes.flat:
            ax.axvline(epochs[best_idx], color="#111", linestyle="--", linewidth=1, alpha=0.35)

    fig.suptitle("TF-GridNet Training Curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(curves_path, dpi=220)
    plt.close(fig)


# ── Loss functions ─────────────────────────────────────────────────────────────

def si_sdr_per_sample(estimated, reference, eps=1e-8):
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    dot        = torch.sum(reference * estimated, dim=-1, keepdim=True)
    alpha      = dot / (ref_energy + eps)
    target_scaled = alpha * reference
    noise         = estimated - target_scaled
    return 10 * torch.log10(
        (torch.sum(target_scaled ** 2, dim=-1) + eps) /
        (torch.sum(noise ** 2, dim=-1) + eps)
    )


def pit_si_sdr_loss(estimated_sources, reference_sources):
    """Permutation-invariant SI-SDR loss for 2-speaker separation."""
    s00 = si_sdr_per_sample(estimated_sources[:, 0], reference_sources[:, 0])
    s11 = si_sdr_per_sample(estimated_sources[:, 1], reference_sources[:, 1])
    s01 = si_sdr_per_sample(estimated_sources[:, 0], reference_sources[:, 1])
    s10 = si_sdr_per_sample(estimated_sources[:, 1], reference_sources[:, 0])
    best = torch.maximum(s00 + s11, s01 + s10)
    return -torch.mean(best)


def multi_resolution_stft_loss(estimated_sources, reference_sources,
                                fft_sizes=(256, 512, 1024), eps=1e-8):
    """Multi-resolution STFT loss for perceptual spectral quality."""
    est = estimated_sources.reshape(-1, estimated_sources.shape[-1])
    ref = reference_sources.reshape(-1, reference_sources.shape[-1])
    total = 0.0
    for n_fft in fft_sizes:
        hop    = n_fft // 4
        window = torch.hann_window(n_fft, device=est.device)
        kw     = dict(n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        est_spec = torch.stft(est, **kw).abs()
        ref_spec = torch.stft(ref, **kw).abs()
        sc  = torch.linalg.norm(ref_spec - est_spec, dim=(-2, -1)) / (
              torch.linalg.norm(ref_spec, dim=(-2, -1)) + eps)
        lm  = torch.mean(torch.abs(torch.log(ref_spec + eps) - torch.log(est_spec + eps)),
                         dim=(-2, -1))
        total += torch.mean(sc + lm)
    return total / len(fft_sizes)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_epoch(model, dataloader, device, stft_weight):
    model.eval()
    total_loss = total_pit = total_stft = 0.0
    with torch.no_grad():
        for batch in dataloader:
            mixture = batch["mixture"].to(device)
            targets = batch["targets"].to(device)
            # AMP inference for consistency
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                estimated = model(mixture, return_all_sources=True)
                pit_loss  = pit_si_sdr_loss(estimated, targets)
                stft_loss = multi_resolution_stft_loss(estimated, targets)
                loss      = pit_loss + stft_weight * stft_loss
            total_loss += loss.item()
            total_pit  += pit_loss.item()
            total_stft += stft_loss.item()
    n = max(1, len(dataloader))
    return total_loss / n, total_pit / n, total_stft / n


# ── Main training loop ─────────────────────────────────────────────────────────

def train():
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = TFGridNet(**MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    lr        = TRAIN_CONFIG["learning_rate"]
    epochs    = TRAIN_CONFIG["epochs"]
    bs        = TRAIN_CONFIG["batch_size"]
    stft_w    = TRAIN_CONFIG["stft_weight"]
    warmup_e  = TRAIN_CONFIG["warmup_epochs"]
    patience  = TRAIN_CONFIG["early_stop_patience"]
    min_delta = TRAIN_CONFIG["min_delta"]
    nw        = TRAIN_CONFIG["num_workers"]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Linear warmup → cosine annealing
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_e
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_e), eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_e]
    )

    # AMP scaler — no-op on CPU
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Dataset & loaders ────────────────────────────────────────────────────
    train_ds_full = Libri2MixDataset(
        csv_file="data/Libri2Mix/train/train.csv",
        chunk_duration=TRAIN_CONFIG["chunk_duration"],
        chunk_mode="random",
    )
    val_ds_full = Libri2MixDataset(
        csv_file="data/Libri2Mix/train/train.csv",
        chunk_duration=TRAIN_CONFIG["chunk_duration"],
        chunk_mode="center",
    )

    dataset_size = len(train_ds_full)
    if dataset_size < 4:
        raise RuntimeError("Dataset too small — run generate_mini_dataset.py first.")
    if dataset_size < 50:
        print("[WARN] Very small dataset — metrics will not reflect real performance.")

    val_size   = max(2, int(dataset_size * 0.2))
    train_size = dataset_size - val_size

    indices       = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_dataset = Subset(train_ds_full, indices[:train_size])
    val_dataset   = Subset(val_ds_full,   indices[train_size:])

    loader_kw = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kw)

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Batch size: {bs} | Warmup: {warmup_e} epochs | Epochs: {epochs}")

    # ── Resume from last checkpoint if present ───────────────────────────────
    start_epoch       = 0
    best_val_loss     = float("inf")
    no_improve_epochs = 0

    if Path(LAST_CKPT).exists():
        print(f"[RESUME] Loading checkpoint from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch       = ckpt["epoch"]
        best_val_loss     = ckpt["best_val_loss"]
        no_improve_epochs = ckpt["no_improve_epochs"]
        print(f"[RESUME] Resuming from epoch {start_epoch + 1} | Best val loss: {best_val_loss:.4f}")

    initialize_csv_logger(CSV_LOG_PATH)
    history = {k: [] for k in ["epoch", "train_loss", "train_pit", "train_stft",
                                "val_loss", "val_pit", "val_stft", "lr", "is_best"]}

    # ── Training loop ────────────────────────────────────────────────────────
    print("\nStarting training — PIT SI-SDR + MR-STFT objective")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        running_loss = running_pit = running_stft = 0.0

        for batch_idx, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                estimated = model(mixture, return_all_sources=True)
                pit_loss  = pit_si_sdr_loss(estimated, targets)
                stft_loss = multi_resolution_stft_loss(estimated, targets)
                loss      = pit_loss + stft_w * stft_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            running_loss  += loss.item()
            running_pit   += pit_loss.item()
            running_stft  += stft_loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  Ep [{epoch+1:02d}/{epochs}] "
                    f"Batch [{batch_idx+1:03d}/{len(train_loader):03d}] "
                    f"Loss: {loss.item():.4f} | PIT: {pit_loss.item():.4f} | "
                    f"STFT: {stft_loss.item():.4f}"
                )

        n_train = max(1, len(train_loader))
        avg_train_loss  = running_loss  / n_train
        avg_train_pit   = running_pit   / n_train
        avg_train_stft  = running_stft  / n_train

        val_loss, val_pit, val_stft = evaluate_epoch(model, val_loader, device, stft_w)
        scheduler.step()
        current_lr   = optimizer.param_groups[0]["lr"]
        epoch_secs   = time.time() - epoch_start

        is_best = val_loss < (best_val_loss - min_delta)
        if is_best:
            best_val_loss     = val_loss
            no_improve_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss":    best_val_loss,
                "config":           MODEL_CONFIG,
            }, BEST_CKPT)
            print(f"  [BEST] Saved best checkpoint → {BEST_CKPT}")
        else:
            no_improve_epochs += 1

        # Always save last checkpoint for Colab session resumption
        torch.save({
            "epoch":               epoch + 1,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_val_loss":        best_val_loss,
            "no_improve_epochs":    no_improve_epochs,
            "config":               MODEL_CONFIG,
        }, LAST_CKPT)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train {avg_train_loss:.4f} (PIT {avg_train_pit:.4f} STFT {avg_train_stft:.4f}) | "
            f"Val {val_loss:.4f} (PIT {val_pit:.4f} STFT {val_stft:.4f}) | "
            f"LR {current_lr:.2e} | {epoch_secs:.0f}s"
            + (" ★" if is_best else f" [no improve {no_improve_epochs}/{patience}]")
        )
        print("-" * 70)

        # CSV logging
        append_csv_log(CSV_LOG_PATH, {
            "epoch":            epoch + 1,
            "train_loss":       f"{avg_train_loss:.6f}",
            "train_pit":        f"{avg_train_pit:.6f}",
            "train_stft":       f"{avg_train_stft:.6f}",
            "val_loss":         f"{val_loss:.6f}",
            "val_pit":          f"{val_pit:.6f}",
            "val_stft":         f"{val_stft:.6f}",
            "lr":               f"{current_lr:.10f}",
            "best_val_loss":    f"{best_val_loss:.6f}",
            "is_best":          int(is_best),
            "no_improve_epochs": no_improve_epochs,
            "epoch_seconds":    f"{epoch_secs:.3f}",
        })

        for key, val in [
            ("epoch", epoch + 1), ("train_loss", avg_train_loss), ("train_pit", avg_train_pit),
            ("train_stft", avg_train_stft), ("val_loss", val_loss), ("val_pit", val_pit),
            ("val_stft", val_stft), ("lr", current_lr), ("is_best", is_best),
        ]:
            history[key].append(val)

        if no_improve_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    save_training_curves(history, CURVES_PATH)
    print(f"\n[DONE] Best val loss: {best_val_loss:.4f}")
    print(f"[ARTIFACT] CSV    → {CSV_LOG_PATH}")
    print(f"[ARTIFACT] Curves → {CURVES_PATH}")


if __name__ == "__main__":
    train()
