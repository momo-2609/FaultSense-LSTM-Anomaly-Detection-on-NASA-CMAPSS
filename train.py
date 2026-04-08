"""
train.py
--------
Training script for FaultSense LSTM Autoencoder + RUL regression head.

Stages
  1. Phase 1 — train autoencoder on HEALTHY-ONLY windows (reconstruction loss)
  2. Phase 2 — fine-tune full model with combined AE + RUL loss on all windows
  3. Calibrate anomaly threshold on healthy training set
  4. Save model checkpoint + training metrics

Run:
  python train.py
  python train.py --data data/processed/cmapss_fd001.pkl --epochs 60
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.lstm_autoencoder import FaultSenseModel, count_params


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_sensors":    14,
    "hidden":       128,
    "latent":       32,
    "seq_len":      30,
    "dropout":      0.2,
    "rul_cap":      125.0,

    # Phase 1: autoencoder pre-training (healthy only)
    "ae_epochs":    30,
    "ae_lr":        1e-3,
    "ae_batch":     64,

    # Phase 2: joint fine-tuning
    "rul_epochs":   30,
    "rul_lr":       5e-4,
    "rul_batch":    64,
    "rul_weight":   0.3,    # loss = ae_loss + rul_weight * rul_loss

    "patience":     10,     # early stopping patience
    "grad_clip":    1.0,
    "k_sigma":      3.0,    # threshold = μ + k*σ
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d


def make_loader(X: np.ndarray, y: np.ndarray = None,
                batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    if y is not None:
        y_t = torch.from_numpy(y).float()
        ds  = TensorDataset(X_t, y_t)
    else:
        ds  = TensorDataset(X_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ── Loss functions ────────────────────────────────────────────────────────────

def ae_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def rul_loss(pred: torch.Tensor, true: torch.Tensor,
             cap: float = 125.0) -> torch.Tensor:
    """Asymmetric loss: penalise overestimating RUL (optimistic) more."""
    diff = pred - true
    return torch.where(diff > 0,
                       torch.exp(diff / 13) - 1,
                       torch.exp(-diff / 10) - 1).mean()


# ── Training loops ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False   # continue
        self.counter += 1
        return self.counter >= self.patience   # stop


def train_phase1(model: FaultSenseModel, ae_loader: DataLoader,
                 val_loader: DataLoader, cfg: dict, device: str) -> dict:
    """Phase 1: autoencoder pre-training on healthy windows only."""
    print("\n── Phase 1: Autoencoder pre-training ──")
    opt      = Adam(model.autoencoder.parameters(), lr=cfg["ae_lr"])
    sched    = ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=True)
    stopper  = EarlyStopping(patience=cfg["patience"])
    history  = {"train": [], "val": []}

    for epoch in range(1, cfg["ae_epochs"] + 1):
        model.train()
        train_losses = []
        for (X_batch,) in ae_loader:
            X_batch = X_batch.to(device)
            recon, _ = model.autoencoder(X_batch)
            loss = ae_loss(recon, X_batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(device)
                recon, _ = model.autoencoder(X_batch)
                val_losses.append(ae_loss(recon, X_batch).item())

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        history["train"].append(t_loss)
        history["val"].append(v_loss)
        sched.step(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['ae_epochs']}  "
                  f"train={t_loss:.5f}  val={v_loss:.5f}")

        if stopper(v_loss):
            print(f"  Early stop at epoch {epoch}")
            break

    return history


def train_phase2(model: FaultSenseModel, train_loader: DataLoader,
                 val_loader: DataLoader, cfg: dict, device: str) -> dict:
    """Phase 2: joint AE + RUL fine-tuning on all windows."""
    print("\n── Phase 2: Joint AE + RUL fine-tuning ──")
    opt     = Adam(model.parameters(), lr=cfg["rul_lr"])
    sched   = ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=True)
    stopper = EarlyStopping(patience=cfg["patience"])
    history = {"train": [], "val": [], "rul": []}

    for epoch in range(1, cfg["rul_epochs"] + 1):
        model.train()
        t_losses, rul_losses = [], []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            recon, rul_pred, _ = model(X_batch)
            l_ae  = ae_loss(recon, X_batch)
            l_rul = rul_loss(rul_pred, y_batch, cfg["rul_cap"])
            loss  = l_ae + cfg["rul_weight"] * l_rul
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            t_losses.append(loss.item())
            rul_losses.append(l_rul.item())

        # Validation
        model.eval()
        v_losses, v_rul = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                recon, rul_pred, _ = model(X_batch)
                v_losses.append(ae_loss(recon, X_batch).item())
                v_rul.append(F.mse_loss(rul_pred, y_batch).item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        history["train"].append(t_loss)
        history["val"].append(v_loss)
        history["rul"].append(np.mean(v_rul))
        sched.step(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            rul_rmse = np.sqrt(np.mean(v_rul)) * cfg["rul_cap"]
            print(f"  Epoch {epoch:3d}/{cfg['rul_epochs']}  "
                  f"train={t_loss:.5f}  val={v_loss:.5f}  "
                  f"RUL-RMSE={rul_rmse:.1f} cycles")

        if stopper(v_loss):
            print(f"  Early stop at epoch {epoch}")
            break

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_rul(model: FaultSenseModel, X_test: np.ndarray,
                 y_test: np.ndarray, device: str, rul_cap: float = 125.0):
    """Compute RMSE and NASA scoring function on test set."""
    model.eval()
    X_t = torch.from_numpy(X_test).float().to(device)
    _, rul_pred, _ = model(X_t)
    rul_pred = rul_pred.cpu().numpy() * rul_cap   # denormalise

    rmse  = float(np.sqrt(np.mean((rul_pred - y_test) ** 2)))
    d     = rul_pred - y_test
    score = float(np.where(d < 0,
                           np.exp(-d / 13) - 1,
                           np.exp(d / 10)  - 1).sum())

    print(f"\nTest-set evaluation:")
    print(f"  RMSE  = {rmse:.2f} cycles")
    print(f"  Score = {score:.1f}  (lower is better)")
    return {"rmse": rmse, "score": score,
            "pred": rul_pred, "true": y_test}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = DEFAULT_CONFIG.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data} …")
    data = load_data(args.data)

    # Loaders
    ae_loader    = make_loader(data["X_ae"],    batch_size=cfg["ae_batch"])
    train_loader = make_loader(data["X_train"], data["y_train"] / cfg["rul_cap"],
                               batch_size=cfg["rul_batch"])
    val_loader   = make_loader(data["X_val"],   data["y_val"]   / cfg["rul_cap"],
                               shuffle=False)
    val_ae_loader = make_loader(data["X_val"],  batch_size=256, shuffle=False)

    # Model
    model = FaultSenseModel(
        n_sensors=len(data["sensor_cols"]),
        hidden=cfg["hidden"], latent=cfg["latent"],
        seq_len=cfg["seq_len"], dropout=cfg["dropout"],
        rul_cap=cfg["rul_cap"],
    ).to(device)
    print(f"Model parameters: {count_params(model):,}")

    # Training
    t0 = time.time()
    h1 = train_phase1(model, ae_loader, val_ae_loader, cfg, device)
    h2 = train_phase2(model, train_loader, val_ae_loader, cfg, device)

    print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")

    # Calibrate threshold
    model.calibrate_threshold(data["X_ae"], k_sigma=cfg["k_sigma"], device=device)

    # Evaluate on test set
    rul_results = evaluate_rul(model, data["X_test"], data["y_test"],
                                device, cfg["rul_cap"])

    # Save checkpoint
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "faultsense.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config":      cfg,
        "threshold":   model.threshold,
        "sensor_cols": data["sensor_cols"],
        "history_ae":  h1,
        "history_rul": h2,
        "test_results": rul_results,
    }, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")


def load_checkpoint(path: str, device: str = "cpu") -> FaultSenseModel:
    """Load a saved model for inference."""
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt["config"]
    model = FaultSenseModel(
        n_sensors=len(ckpt["sensor_cols"]),
        hidden=cfg["hidden"], latent=cfg["latent"],
        seq_len=cfg["seq_len"], dropout=cfg["dropout"],
        rul_cap=cfg["rul_cap"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.threshold   = ckpt["threshold"]
    model.sensor_cols = ckpt["sensor_cols"]
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/cmapss_fd001.pkl")
    parser.add_argument("--out",  default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
