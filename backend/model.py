import torch
import torch.nn as nn

class TFGridNetBlock(nn.Module):
    """
    Core mathematical block of TF-GridNet incorporating:
    1. Intra-frame spectral modeling
    2. Sub-band modeling
    3. Inter-frame temporal modeling
    """
    def __init__(self, in_channels, hidden_channels=64):
        super(TFGridNetBlock, self).__init__()
        
        # 1. Intra-frame (Frequency) Modeling: Processes information within a single time frame
        self.intra_frame = nn.Sequential(
            nn.GroupNorm(1, in_channels), # GroupNorm(1, C) acts as LayerNorm for [B, C, L] tensors
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        )
        
        # 2. Sub-band Modeling: Attention/Convolution across local frequency bands
        self.sub_band = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        )
        
        # 3. Inter-frame (Time) Modeling: Long-term temporal dependencies (Skeleton uses LSTM)
        self.inter_frame_lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True, bidirectional=True)
        self.inter_frame_proj = nn.Linear(hidden_channels * 2, in_channels)
        self.layer_norm_inter = nn.LayerNorm(in_channels) # Expects [..., C]

    def forward(self, x):
        # x is expected to be [Batch, Channels, Freqs, Time]
        B, C, F, T = x.shape
        
        # Intra-frame modeling (fold Time into Batch) -> [B*T, C, F]
        x_intra = x.permute(0, 3, 1, 2).reshape(B * T, C, F)
        x_intra = self.intra_frame(x_intra) + x_intra
        x = x_intra.reshape(B, T, C, F).permute(0, 2, 3, 1)
        
        # Sub-band modeling (2D modeling over Freq/Time) -> [B, C, F, T]
        x_sub = self.sub_band(x) + x
        
        # Inter-frame modeling (fold Freq into Batch) -> [B*F, T, C]
        x_inter = x_sub.permute(0, 2, 3, 1).reshape(B * F, T, C)
        x_inter = self.layer_norm_inter(x_inter)
        lstm_out, _ = self.inter_frame_lstm(x_inter)
        x_inter = self.inter_frame_proj(lstm_out) + x_inter
        x = x_inter.reshape(B, F, T, C).permute(0, 3, 1, 2)
        
        return x

class TFGridNet(nn.Module):
    """
    Mathematical TF-GridNet separator with source-mask decoding.
    The network can return either the primary source only (legacy mode)
    or all estimated sources for PIT-based training and inference.
    """
    def __init__(self, n_fft=256, in_channels=16, n_layers=2, num_sources=2):
        super(TFGridNet, self).__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 2
        self.num_sources = num_sources

        self.freq_bins = n_fft // 2 + 1
        
        # Input Projection
        self.input_proj = nn.Conv2d(1, in_channels, kernel_size=1)
        
        # Stack of TF-GridNet Blocks
        self.blocks = nn.ModuleList([
            TFGridNetBlock(in_channels) for _ in range(n_layers)
        ])
        
        # Mask estimator outputs one mask per separated source.
        self.mask_estimator = nn.Conv2d(in_channels, num_sources, kernel_size=1)

    def forward(self, mixture, target_embedding=None, return_all_sources=False):
        # mixture shape: [Batch, Time]
        window = torch.hann_window(self.n_fft, device=mixture.device)
        spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        mag = torch.abs(spec)        # [B, F, T]
        phase = torch.angle(spec)    # [B, F, T]
        
        # Unsqueeze to add channel dimension: [B, 1, F, T]
        x = mag.unsqueeze(1)
        
        # 1. Input Projection
        x = self.input_proj(x)
        
        # 2. Mathematical Processing blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Mask Estimation
        masks = torch.sigmoid(self.mask_estimator(x))         # [B, S, F, T]
        masks = masks / (torch.sum(masks, dim=1, keepdim=True) + 1e-8)

        # 4. Mask application on complex spectrogram for each source.
        complex_sources = masks * spec.unsqueeze(1)           # [B, S, F, T]

        # 5. Inverse STFT to get separated waveforms.
        separated_sources = []
        for source_idx in range(self.num_sources):
            separated_wav = torch.istft(
                complex_sources[:, source_idx, :, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                length=mixture.shape[-1]
            )
            separated_sources.append(separated_wav)

        separated_sources = torch.stack(separated_sources, dim=1)  # [B, S, T]
        if return_all_sources:
            return separated_sources

        # Backward-compatible default: return source 0 only.
        return separated_sources[:, 0, :]

if __name__ == "__main__":
    # Test strict pipeline
    model = TFGridNet()
    dummy_audio = torch.randn(2, 16000) # 2 samples of 1-second audio
    output_single = model(dummy_audio)
    output_multi = model(dummy_audio, return_all_sources=True)
    print(f"Mathematical execution trace successful. Single-source shape: {output_single.shape}")
    print(f"Mathematical execution trace successful. Multi-source shape:  {output_multi.shape}")
