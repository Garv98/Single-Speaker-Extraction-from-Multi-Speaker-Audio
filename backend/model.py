import torch
import torch.nn as nn


class IntraFrameMHSA(nn.Module):
    """
    Multi-head self-attention over the frequency axis within each time frame.

    At each time step t, all F frequency bins attend to each other, allowing
    the model to capture spectral correlations across the full bandwidth.
    Uses Pre-LN (layer norm before attention) for stable gradient flow.
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D, F, T]
        B, D, F, T = x.shape
        # Fold T into batch so each time frame is processed independently
        x_2d = x.permute(0, 3, 2, 1).reshape(B * T, F, D)   # [B*T, F, D]
        attn_out, _ = self.attn(self.norm(x_2d), self.norm(x_2d), self.norm(x_2d))
        x_out = x_2d + self.dropout(attn_out)
        return x_out.reshape(B, T, F, D).permute(0, 3, 2, 1) # [B, D, F, T]


class SubBandConv(nn.Module):
    """
    Local sub-band spectral modeling via depthwise-separable convolution.

    Models short-range frequency correlations between adjacent bands.
    This is a novel architectural contribution over vanilla TF-GridNet:
    the extra sub-band module captures harmonic structure that global
    attention may miss when F is large.
    """
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.norm = nn.GroupNorm(1, d_model)
        # Depthwise conv along frequency axis only (groups=d_model)
        self.dw_conv = nn.Conv2d(
            d_model, d_model, kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0), groups=d_model
        )
        self.pw_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.act = nn.PReLU()

    def forward(self, x):
        # Pre-norm residual: x + f(norm(x))
        return x + self.pw_conv(self.act(self.dw_conv(self.norm(x))))


class InterFrameBiLSTM(nn.Module):
    """
    Bidirectional LSTM over the time axis at each frequency bin.

    At each frequency bin f, all T time frames are processed in sequence,
    capturing long-range temporal dependencies. Pre-LN residual style.
    """
    def __init__(self, d_model, hidden_size, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_size * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D, F, T]
        B, D, F, T = x.shape
        # Fold F into batch so each frequency bin is processed independently
        x_2d = x.permute(0, 2, 3, 1).reshape(B * F, T, D)   # [B*F, T, D]
        lstm_out, _ = self.lstm(self.norm(x_2d))
        x_out = x_2d + self.dropout(self.proj(lstm_out))
        return x_out.reshape(B, F, T, D).permute(0, 3, 1, 2) # [B, D, F, T]


class TFGridNetBlock(nn.Module):
    """
    One TF-GridNet processing block.

    Order: intra-frame MHSA → sub-band conv → inter-frame BiLSTM.
    Each sub-module uses its own Pre-LN residual connection.
    """
    def __init__(self, d_model, n_heads, lstm_hidden, dropout=0.0):
        super().__init__()
        self.intra = IntraFrameMHSA(d_model, n_heads, dropout)
        self.sub_band = SubBandConv(d_model)
        self.inter = InterFrameBiLSTM(d_model, lstm_hidden, dropout)

    def forward(self, x):
        x = self.intra(x)
        x = self.sub_band(x)
        x = self.inter(x)
        return x


class TFGridNet(nn.Module):
    """
    TF-GridNet for monaural 2-speaker separation.

    Architecture follows Luo & Mesgarani (ICASSP 2023) with one enhancement:
    an additional sub-band depthwise conv module between the intra-frame
    attention and inter-frame LSTM to capture local harmonic structure.

    Key design choices vs naive magnitude-mask baseline:
      - Real+imaginary STFT stacked as 2-channel input: richer representation
      - MHSA for intra-frame: long-range spectral modelling
      - Complex Ratio Mask (CRM) output: preserves phase information
      - n_fft=512: 32 ms frames at 16 kHz (standard for Libri2Mix)

    Args:
        n_fft:       STFT window length. Default 512 (32 ms at 16 kHz).
        hop_length:  STFT hop size. Default 128 (8 ms stride).
        d_model:     Internal embedding dimension (paper uses 64–128).
        n_heads:     Attention heads for intra-frame MHSA. Must divide d_model.
        lstm_hidden: Hidden size per direction in inter-frame BiLSTM.
        n_layers:    Number of TF-GridNet blocks (paper uses 6).
        num_sources: Number of speakers to separate.
        dropout:     Dropout probability applied in attention and LSTM.
    """
    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        d_model=64,
        n_heads=4,
        lstm_hidden=256,
        n_layers=6,
        num_sources=2,
        dropout=0.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_sources = num_sources

        # Project stacked real+imag (2 channels) → d_model
        self.input_proj = nn.Conv2d(2, d_model, kernel_size=1)

        self.blocks = nn.ModuleList([
            TFGridNetBlock(d_model, n_heads, lstm_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Predict real and imaginary mask components for each source
        # Output channels: [r_s0, i_s0, r_s1, i_s1, ...]
        self.output_proj = nn.Conv2d(d_model, num_sources * 2, kernel_size=1)

    def forward(self, mixture, return_all_sources=False):
        """
        Args:
            mixture:           [B, L] raw waveform
            return_all_sources: if True return [B, S, L], else return [B, L] (source 0)
        """
        window = torch.hann_window(self.n_fft, device=mixture.device)
        spec = torch.stft(
            mixture, n_fft=self.n_fft, hop_length=self.hop_length,
            window=window, return_complex=True
        )  # [B, F, T]

        # Stack real + imag → [B, 2, F, T]
        x = torch.stack([spec.real, spec.imag], dim=1)

        # Input projection → [B, D, F, T]
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # Output projection → [B, S*2, F, T]
        out = self.output_proj(x)

        # Split interleaved channels into per-source complex masks
        # Channels: 0→r_s0, 1→i_s0, 2→r_s1, 3→i_s1, ...
        mask_r = torch.tanh(out[:, 0::2])  # [B, S, F, T]
        mask_i = torch.tanh(out[:, 1::2])  # [B, S, F, T]

        spec_r = spec.real.unsqueeze(1)    # [B, 1, F, T]
        spec_i = spec.imag.unsqueeze(1)    # [B, 1, F, T]

        # Complex multiplication: (spec_r + j*spec_i)(mask_r + j*mask_i)
        out_r = spec_r * mask_r - spec_i * mask_i  # [B, S, F, T]
        out_i = spec_r * mask_i + spec_i * mask_r  # [B, S, F, T]

        separated_sources = []
        for s in range(self.num_sources):
            complex_spec = torch.complex(out_r[:, s], out_i[:, s])
            wav = torch.istft(
                complex_spec, n_fft=self.n_fft, hop_length=self.hop_length,
                window=window, length=mixture.shape[-1]
            )
            separated_sources.append(wav)

        separated_sources = torch.stack(separated_sources, dim=1)  # [B, S, L]

        if return_all_sources:
            return separated_sources
        return separated_sources[:, 0, :]


if __name__ == "__main__":
    model = TFGridNet()
    dummy = torch.randn(2, 16000)
    out_single = model(dummy)
    out_multi = model(dummy, return_all_sources=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Single output: {out_single.shape}")
    print(f"Multi  output: {out_multi.shape}")
    print(f"Parameters:    {n_params:,}")
