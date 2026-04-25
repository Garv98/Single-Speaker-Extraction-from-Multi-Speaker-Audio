import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import soundfile as sf
import random
import numpy as np

class Libri2MixDataset(Dataset):
    """
    Robust PyTorch Dataset mapped to a CSV file structuring Libri2Mix audio files.
    Dynamically loads and chunks audio files to prevent RAM overflow during training.
    """
    def __init__(self, csv_file, sample_rate=16000, chunk_duration=3.0, chunk_mode="random"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.sample_rate = sample_rate
        # For training, we crop exactly N continuous seconds of audio
        self.chunk_size = int(sample_rate * chunk_duration)
        if chunk_mode not in {"random", "center", "start"}:
            raise ValueError("chunk_mode must be one of {'random', 'center', 'start'}")
        self.chunk_mode = chunk_mode
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Read files (Soundfile natively grabs length without full memory load)
        info = sf.info(row["mixture_path"])
        total_frames = info.frames
        
        # 2. Dynamic Chunking Logic
        # Ensure we always grab a consistent chunk size
        if total_frames <= self.chunk_size:
            start_frame = 0
            frames_to_read = total_frames
        else:
            max_start = total_frames - self.chunk_size
            if self.chunk_mode == "random":
                start_frame = random.randint(0, max_start)
            elif self.chunk_mode == "center":
                start_frame = max_start // 2
            else:
                start_frame = 0
            frames_to_read = self.chunk_size
            
        # 3. Read specific chunks
        mix, _ = sf.read(row["mixture_path"], start=start_frame, frames=frames_to_read)
        s1, _ = sf.read(row["source_1_path"], start=start_frame, frames=frames_to_read)
        s2, _ = sf.read(row["source_2_path"], start=start_frame, frames=frames_to_read)
        
        # Zero-pad if the audio was shorter than the chunk size
        if len(mix) < self.chunk_size:
            pad_length = self.chunk_size - len(mix)
            mix = np.pad(mix, (0, pad_length))
            s1 = np.pad(s1, (0, pad_length))
            s2 = np.pad(s2, (0, pad_length))
            
        # 4. Convert to PyTorch tensors (Float32 required for Model graph)
        mix_tensor = torch.tensor(mix, dtype=torch.float32)
        s1_tensor = torch.tensor(s1, dtype=torch.float32)
        s2_tensor = torch.tensor(s2, dtype=torch.float32)
        
        # Return both targets for PIT training while preserving legacy keys.
        return {
            "mixture": mix_tensor,
            "target": s1_tensor,
            "target_1": s1_tensor,
            "target_2": s2_tensor,
            "targets": torch.stack([s1_tensor, s2_tensor], dim=0),
        }

if __name__ == "__main__":
    # Test pipeline Validation
    try:
        dataset = Libri2MixDataset(csv_file="data/Libri2Mix/train/train.csv")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch in dataloader:
            print(f"[Success] Loaded Batch:")
            print(f"  Mixture Shape: {batch['mixture'].shape}")
            print(f"  Target Shape:  {batch['target'].shape}")
            break
        print("Backend Libri2Mix pipeline strictly validated.")
    except FileNotFoundError:
        print("CSV not found. Did you run `generate_mini_dataset.py` first?")
