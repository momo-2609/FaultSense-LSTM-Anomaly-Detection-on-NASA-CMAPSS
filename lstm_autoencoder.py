"""
models/lstm_autoencoder.py
--------------------------
Bidirectional LSTM Autoencoder for anomaly detection on CMAPSS.

Architecture
  Encoder : BiLSTM(128) → Dropout(0.2) → LSTM(64) → Dense(32, tanh)
  Decoder : RepeatVector(30) → LSTM(64) → LSTM(128) → TimeDistributed Dense(14)

Anomaly score = per-window MSE reconstruction error.
Threshold = μ + 3σ over healthy training windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder → fixed-size latent vector."""

    def __init__(self, n_sensors: int, hidden: int = 128, latent: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Second LSTM reduces bidirectional output → single direction
        self.lstm2 = nn.LSTM(
            input_size=hidden * 2,   # fwd + bwd concatenated
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fc_bottleneck = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, n_sensors)
        returns: (B, latent)
        """
        out, _ = self.bilstm(x)          # (B, T, 2*hidden)
        out    = self.dropout(out)
        out, (h, _) = self.lstm2(out)    # h : (1, B, hidden)
        h = h.squeeze(0)                 # (B, hidden)
        z = torch.tanh(self.fc_bottleneck(h))   # (B, latent)
        return z


class LSTMDecoder(nn.Module):
    """LSTM decoder: latent vector → reconstructed sequence."""

    def __init__(self, n_sensors: int, hidden: int = 128, latent: int = 32,
                 seq_len: int = 30):
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(
            input_size=latent,
            hidden_size=hidden // 2,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden // 2,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden, n_sensors)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, latent)
        returns: (B, T, n_sensors)
        """
        # Repeat latent vector across timesteps
        z_rep = z.unsqueeze(1).expand(-1, self.seq_len, -1)   # (B, T, latent)
        out, _ = self.lstm1(z_rep)        # (B, T, hidden//2)
        out, _ = self.lstm2(out)          # (B, T, hidden)
        recon  = self.output_proj(out)    # (B, T, n_sensors)
        return recon


class LSTMAutoencoder(nn.Module):
    """
    Full autoencoder: encode → bottleneck → decode.
    Trained to reconstruct healthy sensor windows.
    Anomaly score = MSE(input, reconstruction).
    """

    def __init__(self, n_sensors: int = 14, hidden: int = 128,
                 latent: int = 32, seq_len: int = 30, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(n_sensors, hidden, latent, dropout)
        self.decoder = LSTMDecoder(n_sensors, hidden, latent, seq_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, n_sensors)
        returns: (reconstruction, latent_z)
        """
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample MSE reconstruction error.
        x : (B, T, n_sensors)
        returns: (B,)
        """
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))

    def per_sensor_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sensor reconstruction error for explainability.
        x : (B, T, n_sensors)
        returns: (B, n_sensors)
        """
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)   # avg over T


class RULHead(nn.Module):
    """
    Optional regression head attached after the encoder.
    Shares encoder weights; adds a small MLP for RUL prediction.
    """

    def __init__(self, latent: int = 32, rul_cap: float = 125.0):
        super().__init__()
        self.rul_cap = rul_cap
        self.mlp = nn.Sequential(
            nn.Linear(latent, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, latent) → rul_pred : (B,)"""
        rul = self.mlp(z).squeeze(-1)
        return torch.clamp(rul, 0, self.rul_cap)


class FaultSenseModel(nn.Module):
    """
    Combined model: autoencoder anomaly detection + RUL regression.
    Single forward pass returns both outputs.
    """

    def __init__(self, n_sensors: int = 14, hidden: int = 128,
                 latent: int = 32, seq_len: int = 30,
                 dropout: float = 0.2, rul_cap: float = 125.0):
        super().__init__()
        self.autoencoder = LSTMAutoencoder(n_sensors, hidden, latent,
                                           seq_len, dropout)
        self.rul_head    = RULHead(latent, rul_cap)
        self.threshold   = None   # set after calibration

    def forward(self, x: torch.Tensor):
        recon, z  = self.autoencoder(x)
        rul_pred  = self.rul_head(z)
        return recon, rul_pred, z

    @torch.no_grad()
    def calibrate_threshold(self, X_healthy: np.ndarray,
                             k_sigma: float = 3.0,
                             device: str = "cpu") -> float:
        """
        Set anomaly threshold = μ + k*σ of healthy reconstruction errors.
        Call this after training, before deployment.
        """
        self.eval()
        errors = []
        loader = torch.utils.data.DataLoader(
            torch.from_numpy(X_healthy).float(),
            batch_size=256, shuffle=False
        )
        for batch in loader:
            batch = batch.to(device)
            err   = self.autoencoder.reconstruction_error(batch)
            errors.append(err.cpu().numpy())

        errors = np.concatenate(errors)
        mu, sig = errors.mean(), errors.std()
        self.threshold = float(mu + k_sigma * sig)
        print(f"Threshold calibrated: μ={mu:.4f}  σ={sig:.4f}  "
              f"threshold={self.threshold:.4f}")
        return self.threshold

    @torch.no_grad()
    def predict(self, x: np.ndarray, device: str = "cpu"):
        """
        Full inference on a numpy window.
        x : (T, n_sensors) or (B, T, n_sensors)
        Returns dict with anomaly score, rul, per-sensor errors, flag.
        """
        self.eval()
        single = x.ndim == 2
        if single:
            x = x[np.newaxis]   # add batch dim

        t = torch.from_numpy(x.astype(np.float32)).to(device)
        recon, rul_pred, _ = self.forward(t)
        raw_err  = F.mse_loss(recon, t, reduction="none")
        score    = raw_err.mean(dim=(1, 2)).cpu().numpy()
        per_sens = raw_err.mean(dim=1).cpu().numpy()   # (B, n_sensors)
        rul      = rul_pred.cpu().numpy()

        result = {
            "anomaly_score":  score[0] if single else score,
            "rul":            rul[0]   if single else rul,
            "per_sensor_mse": per_sens[0] if single else per_sens,
            "is_anomaly": (score[0] > self.threshold) if (
                single and self.threshold is not None) else None,
        }
        return result


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FaultSenseModel()
    print(f"Total trainable parameters: {count_params(model):,}")

    # Smoke test
    dummy = torch.randn(4, 30, 14)
    recon, rul, z = model(dummy)
    print(f"Input:         {dummy.shape}")
    print(f"Reconstruction:{recon.shape}")
    print(f"RUL prediction:{rul.shape}")
    print(f"Latent z:      {z.shape}")
