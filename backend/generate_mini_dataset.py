import os
import numpy as np
import soundfile as sf
import csv

def generate_mini_dataset(base_dir="data/Libri2Mix/train"):
    """
    Generates a tiny footprint dataset (10 samples) mapping to avoid 
    downloading 50GB of standard Libri2Mix data, keeping project size < 20MB.
    Generates 'mix', 's1', and 's2' directories and a 'train.csv'.
    """
    print("Generating ultra-lightweight Mini-Dataset...")
    os.makedirs(base_dir, exist_ok=True)
    
    dirs = ['mix_clean', 's1', 's2']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    csv_data = [["mixture_path", "source_1_path", "source_2_path"]]
    
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    for i in range(10):
        # Generate random overlapping frequencies to simulate speech
        freq1 = np.random.uniform(200, 600)
        freq2 = np.random.uniform(600, 1000)
        
        s1 = 0.5 * np.sin(2 * np.pi * freq1 * t)
        s2 = 0.5 * np.sin(2 * np.pi * freq2 * t)
        mix = s1 + s2
        
        mix_path = os.path.join(base_dir, "mix_clean", f"mix_{i}.wav")
        s1_path = os.path.join(base_dir, "s1", f"s1_{i}.wav")
        s2_path = os.path.join(base_dir, "s2", f"s2_{i}.wav")
        
        sf.write(mix_path, mix, sr)
        sf.write(s1_path, s1, sr)
        sf.write(s2_path, s2, sr)
        
        # We save absolute paths to mimic real environment pipelines robustly
        csv_data.append([os.path.abspath(mix_path), os.path.abspath(s1_path), os.path.abspath(s2_path)])
        
    csv_path = os.path.join(base_dir, "train.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    print(f"Successfully generated dataset at: {base_dir}")
    print(f"CSV Mapping created at: {csv_path}")

if __name__ == "__main__":
    generate_mini_dataset()
