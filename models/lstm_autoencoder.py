"""
models/lstm_autoencoder.py
--------------------------
Simple LSTM for RUL estimation — paper Algorithm 5.

Architecture (exact match to paper):
  LSTM(n_sensors → hidden=32, batch_first=True)
  → take last hidden state hT
  → Dropout(0.5)
  → Linear(32 → 8) → ReLU
  → Linear(8  → 8) → ReLU
  → Linear(8  → 1)
  → clamp(0, 1)   [targets are normalised to [0,1] during training]

At inference: output × rul_cap gives predicted RUL in cycles.

Note: the old BiLSTM autoencoder (LSTMEncoder, LSTMDecoder, LSTMAutoencoder,
RULHead) is replaced by this single class. FaultSenseModel is kept as the
external interface so train.py and app.py require no changes.
The calibrate_threshold / anomaly-detection path is retained as a no-op
stub so existing checkpoint-loading code does not break.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Core model ────────────────────────────────────────────────────────────────

class FaultSenseModel(nn.Module):
    """
    Simple LSTM RUL estimator matching paper Algorithm 5.

    forward(x) returns (None, rul_pred, hT) to preserve the three-element
    tuple that train.py and app.py expect from the old FaultSenseModel.
    The first element (reconstruction) is None — train.py must NOT compute
    ae_loss when using this model (see train.py Phase 2 loss comment).
    """

    def __init__(self, n_sensors: int = 14, hidden: int = 32,
                 dropout: float = 0.5, rul_cap: float = 130.0,
                 # Legacy kwargs accepted but ignored — keeps train.py compat
                 latent: int = 32, seq_len: int = 30, **kwargs):
        super().__init__()
        self.rul_cap   = rul_cap
        self.threshold = None   # anomaly threshold stub

        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x   : (B, T, n_sensors)
        returns (None, rul_pred, hT)
          None     — no reconstruction (no AE)
          rul_pred — (B,) predicted RUL normalised to [0, 1]
          hT       — (B, hidden) last hidden state (latent representation)
        """
        out, _  = self.lstm(x)            # (B, T, hidden)
        hT      = out[:, -1, :]           # last timestep: (B, hidden)
        hT_drop = self.dropout(hT)
        rul     = self.head(hT_drop).squeeze(-1)   # (B,)
        rul     = torch.clamp(rul, 0.0, 1.0)
        return None, rul, hT

    def predict(self, x: np.ndarray, device: str = "cpu") -> dict:
        """
        Inference on a single window or batch.
        x : (T, n_sensors) or (B, T, n_sensors)
        Returns dict with 'rul' in cycles and stub anomaly fields.
        """
        self.eval()
        single = x.ndim == 2
        if single:
            x = x[np.newaxis]

        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(device)
            _, rul_pred, _ = self.forward(t)
            rul_cycles = rul_pred.cpu().numpy() * self.rul_cap

        return {
            "rul":            rul_cycles[0] if single else rul_cycles,
            "anomaly_score":  None,
            "per_sensor_mse": None,
            "is_anomaly":     None,
        }

    @torch.no_grad()
    def calibrate_threshold(self, X_healthy: np.ndarray,
                             k_sigma: float = 3.0,
                             device: str = "cpu") -> float:
        """
        Stub — no autoencoder so no reconstruction threshold.
        Sets self.threshold = 0.0 to satisfy train.py's calibrate call.
        """
        self.threshold = 0.0
        print("Threshold calibration: N/A (simple LSTM has no AE). "
              "Anomaly detection disabled.")
        return self.threshold


# ── Parameter count helper ────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FaultSenseModel(n_sensors=14, hidden=32, dropout=0.5, rul_cap=130.0)
    print(f"Total trainable parameters: {count_params(model):,}")

    dummy = torch.randn(4, 30, 14)
    _, rul, hT = model(dummy)
    print(f"Input shape   : {dummy.shape}")
    print(f"RUL pred shape: {rul.shape}   (normalised, expect values in [0,1])")
    print(f"Latent hT     : {hT.shape}")
    print(f"Sample preds  : {rul.detach().numpy().round(3)}")
