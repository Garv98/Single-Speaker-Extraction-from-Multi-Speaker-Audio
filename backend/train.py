import random
import csv
import time
from pathlib import Path

import matplotlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import Libri2MixDataset
from model import TFGridNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARTIFACT_DIR = Path("figures") / "training"
CSV_LOG_PATH = ARTIFACT_DIR / "training_history.csv"
CURVES_PATH = ARTIFACT_DIR / "training_curves.png"

CSV_FIELDS = [
    "epoch",
    "train_loss",
    "train_pit",
    "train_stft",
    "val_loss",
    "val_pit",
    "val_stft",
    "lr",
    "best_val_loss",
    "is_best",
    "no_improve_epochs",
    "epoch_seconds",
]


def initialize_csv_logger(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_log(csv_path: Path, row: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def save_training_curves(history: dict, curves_path: Path):
    epochs = history["epoch"]
    if not epochs:
        return

    curves_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    ax_total = axes[0, 0]
    ax_pit = axes[0, 1]
    ax_stft = axes[1, 0]
    ax_lr = axes[1, 1]

    ax_total.plot(epochs, history["train_loss"], label="Train", color="#1f77b4", linewidth=2)
    ax_total.plot(epochs, history["val_loss"], label="Validation", color="#ff7f0e", linewidth=2)
    ax_total.set_title("Total Loss")
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.grid(alpha=0.25)
    ax_total.legend()

    ax_pit.plot(epochs, history["train_pit"], label="Train PIT", color="#2ca02c", linewidth=2)
    ax_pit.plot(epochs, history["val_pit"], label="Validation PIT", color="#d62728", linewidth=2)
    ax_pit.set_title("PIT Loss")
    ax_pit.set_xlabel("Epoch")
    ax_pit.set_ylabel("Loss")
    ax_pit.grid(alpha=0.25)
    ax_pit.legend()

    ax_stft.plot(epochs, history["train_stft"], label="Train STFT", color="#9467bd", linewidth=2)
    ax_stft.plot(epochs, history["val_stft"], label="Validation STFT", color="#8c564b", linewidth=2)
    ax_stft.set_title("MR-STFT Loss")
    ax_stft.set_xlabel("Epoch")
    ax_stft.set_ylabel("Loss")
    ax_stft.grid(alpha=0.25)
    ax_stft.legend()

    ax_lr.plot(epochs, history["lr"], label="Learning Rate", color="#17becf", linewidth=2)
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.grid(alpha=0.25)
    ax_lr.legend()

    if history["val_loss"]:
        best_epoch_index = min(
            range(len(history["val_loss"])),
            key=lambda idx: history["val_loss"][idx],
        )
        best_epoch = epochs[best_epoch_index]
        for axis in (ax_total, ax_pit, ax_stft, ax_lr):
            axis.axvline(best_epoch, color="#111111", linestyle="--", linewidth=1, alpha=0.35)

    fig.suptitle("TF-GridNet Training Curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(curves_path, dpi=220)
    plt.close(fig)


def si_sdr_per_sample(estimated, reference, eps=1e-8):
    """Compute SI-SDR for each sample in a batch."""
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)

    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    dot_product = torch.sum(reference * estimated, dim=-1, keepdim=True)

    alpha = dot_product / (ref_energy + eps)
    target_scaled = alpha * reference
    noise = estimated - target_scaled

    target_energy = torch.sum(target_scaled ** 2, dim=-1)
    noise_energy = torch.sum(noise ** 2, dim=-1)
    return 10 * torch.log10((target_energy + eps) / (noise_energy + eps))


def pit_si_sdr_loss(estimated_sources, reference_sources):
    """
    Permutation-invariant SI-SDR loss for 2-speaker separation.
    Shapes:
      estimated_sources: [B, 2, T]
      reference_sources: [B, 2, T]
    """
    s00 = si_sdr_per_sample(estimated_sources[:, 0, :], reference_sources[:, 0, :])
    s11 = si_sdr_per_sample(estimated_sources[:, 1, :], reference_sources[:, 1, :])

    s01 = si_sdr_per_sample(estimated_sources[:, 0, :], reference_sources[:, 1, :])
    s10 = si_sdr_per_sample(estimated_sources[:, 1, :], reference_sources[:, 0, :])

    perm_a = s00 + s11
    perm_b = s01 + s10
    best_si_sdr = torch.maximum(perm_a, perm_b)
    return -torch.mean(best_si_sdr)


def multi_resolution_stft_loss(
    estimated_sources,
    reference_sources,
    fft_sizes=(256, 512, 1024),
    eps=1e-8,
):
    """Multi-resolution STFT loss to improve perceptual spectral quality."""
    est = estimated_sources.reshape(-1, estimated_sources.shape[-1])
    ref = reference_sources.reshape(-1, reference_sources.shape[-1])

    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=est.device)

        est_spec = torch.stft(
            est,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        ref_spec = torch.stft(
            ref,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()

        spectral_convergence = torch.linalg.norm(ref_spec - est_spec, dim=(-2, -1)) / (
            torch.linalg.norm(ref_spec, dim=(-2, -1)) + eps
        )
        log_magnitude = torch.mean(
            torch.abs(torch.log(ref_spec + eps) - torch.log(est_spec + eps)),
            dim=(-2, -1),
        )
        total += torch.mean(spectral_convergence + log_magnitude)

    return total / float(len(fft_sizes))


def evaluate_epoch(model, dataloader, device, stft_weight):
    model.eval()
    total_loss = 0.0
    total_pit = 0.0
    total_stft = 0.0

    with torch.no_grad():
        for batch in dataloader:
            mixture = batch["mixture"].to(device)
            targets = batch["targets"].to(device)

            estimated = model(mixture, return_all_sources=True)
            pit_loss = pit_si_sdr_loss(estimated, targets)
            stft_loss = multi_resolution_stft_loss(estimated, targets)
            loss = pit_loss + stft_weight * stft_loss

            total_loss += loss.item()
            total_pit += pit_loss.item()
            total_stft += stft_loss.item()

    num_batches = max(1, len(dataloader))
    return (
        total_loss / num_batches,
        total_pit / num_batches,
        total_stft / num_batches,
    )


def train():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = TFGridNet(n_fft=256, in_channels=16, n_layers=3, num_sources=2).to(device)

    learning_rate = 2e-4
    epochs = 80
    batch_size = 6
    stft_weight = 0.3

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    train_dataset_full = Libri2MixDataset(
        csv_file="data/Libri2Mix/train/train.csv",
        chunk_duration=3.0,
        chunk_mode="random",
    )
    val_dataset_full = Libri2MixDataset(
        csv_file="data/Libri2Mix/train/train.csv",
        chunk_duration=3.0,
        chunk_mode="center",
    )

    dataset_size = len(train_dataset_full)
    if dataset_size < 4:
        raise RuntimeError("Dataset too small. Generate more training examples before running robust training")

    if dataset_size < 50:
        print(
            "[WARN] Very small dataset detected."
            " Training will overfit quickly; increase dataset size for publication-quality metrics."
        )

    val_size = max(2, int(dataset_size * 0.2)) if dataset_size >= 10 else 1
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Starting robust training with PIT SI-SDR + MR-STFT objective...")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    initialize_csv_logger(CSV_LOG_PATH)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_pit": [],
        "train_stft": [],
        "val_loss": [],
        "val_pit": [],
        "val_stft": [],
        "lr": [],
        "is_best": [],
    }

    best_val_loss = float("inf")
    no_improve_epochs = 0
    early_stop_patience = 10
    min_delta = 1e-3

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_pit = 0.0
        running_stft = 0.0

        for batch_idx, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad(set_to_none=True)

            estimated = model(mixture, return_all_sources=True)
            pit_loss = pit_si_sdr_loss(estimated, targets)
            stft_loss = multi_resolution_stft_loss(estimated, targets)
            loss = pit_loss + stft_weight * stft_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            running_pit += pit_loss.item()
            running_stft += stft_loss.item()

            print(
                f"Epoch [{epoch + 1:02d}/{epochs}] Batch [{batch_idx + 1:03d}/{len(train_loader):03d}] "
                f"Loss: {loss.item():.4f} | PIT: {pit_loss.item():.4f} | STFT: {stft_loss.item():.4f}"
            )

        num_train_batches = max(1, len(train_loader))
        avg_train_loss = running_loss / num_train_batches
        avg_train_pit = running_pit / num_train_batches
        avg_train_stft = running_stft / num_train_batches

        val_loss, val_pit, val_stft = evaluate_epoch(model, val_loader, device, stft_weight)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start_time

        print("-" * 78)
        print(
            f"Epoch {epoch + 1:02d} Summary | "
            f"Train Loss: {avg_train_loss:.4f} (PIT {avg_train_pit:.4f}, STFT {avg_train_stft:.4f}) | "
            f"Val Loss: {val_loss:.4f} (PIT {val_pit:.4f}, STFT {val_stft:.4f})"
        )

        is_best = val_loss < (best_val_loss - min_delta)
        if is_best:
            best_val_loss = val_loss
            no_improve_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {
                    "n_fft": 256,
                    "in_channels": 16,
                    "n_layers": 3,
                    "num_sources": 2,
                },
            }
            torch.save(checkpoint, "best_tfgridnet.pth")
            print("[CHECKPOINT] Saved improved model to best_tfgridnet.pth")
        else:
            no_improve_epochs += 1

        append_csv_log(
            CSV_LOG_PATH,
            {
                "epoch": epoch + 1,
                "train_loss": f"{avg_train_loss:.6f}",
                "train_pit": f"{avg_train_pit:.6f}",
                "train_stft": f"{avg_train_stft:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_pit": f"{val_pit:.6f}",
                "val_stft": f"{val_stft:.6f}",
                "lr": f"{current_lr:.10f}",
                "best_val_loss": f"{best_val_loss:.6f}",
                "is_best": int(is_best),
                "no_improve_epochs": no_improve_epochs,
                "epoch_seconds": f"{epoch_seconds:.3f}",
            },
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_pit"].append(avg_train_pit)
        history["train_stft"].append(avg_train_stft)
        history["val_loss"].append(val_loss)
        history["val_pit"].append(val_pit)
        history["val_stft"].append(val_stft)
        history["lr"].append(current_lr)
        history["is_best"].append(is_best)

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} non-improving epochs")
            break

    save_training_curves(history, CURVES_PATH)
    print(f"[ARTIFACT] CSV history saved to: {CSV_LOG_PATH}")
    print(f"[ARTIFACT] Training curves saved to: {CURVES_PATH}")


if __name__ == "__main__":
    train()