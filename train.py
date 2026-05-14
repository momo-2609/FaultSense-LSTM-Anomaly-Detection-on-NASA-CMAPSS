"""
train.py
--------
Training script for FaultSense — supports all 4 CMAPSS subsets.

Usage:
  python train.py                         # train on FD001
  python train.py --subset FD002          # train on FD002
  python train.py --subset FD001 FD002    # train multiple subsets
  python train.py --all                   # train all 4 subsets
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.lstm_autoencoder import FaultSenseModel, count_params

import mlflow


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "hidden":      128,
    "latent":       32,
    "seq_len":      30,
    "dropout":      0.2,
    "rul_cap":    125.0,
    "ae_epochs":    50,
    "ae_lr":       1e-3,
    "ae_batch":     64,
    "rul_epochs":  100,
    "rul_lr":      1e-4,
    "rul_batch":    64,
    "rul_weight":   0.5,
    "patience":     30,
    "grad_clip":    1.0,
    "k_sigma":      2.5,
    "oversample":   3.0,
}

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# ── Per-subset overrides ──────────────────────────────────────────────────────
# FD001/FD003: 1 operating condition → AE loss converges very low (~0.30)
# quickly, starving the RUL head. rul_weight must be higher + faster lr.
# FD002/FD004: 6 operating conditions → AE loss stays higher (~0.50).
SUBSET_CONFIG = {
    "FD001": {"rul_weight": 2.0,   # was 0.3 — AE loss is now ~0.004 so needs rebalancing
          "oversample": 8.0, "rul_lr": 1e-3,
          "ae_epochs": 30, "patience": 25, "rul_epochs": 150},
    "FD002": {"rul_weight": 2.0,   # was 0.3 — AE loss is now ~0.004 so needs rebalancing
          "oversample": 8.0, "rul_lr": 1e-3,
          "ae_epochs": 30, "patience": 25, "rul_epochs": 150},
    "FD003": {"rul_weight": 2.0,   # was 0.3 — AE loss is now ~0.004 so needs rebalancing
          "oversample": 8.0, "rul_lr": 1e-3,
          "ae_epochs": 30, "patience": 25, "rul_epochs": 150},
    "FD004": {"rul_weight": 2.0,   # was 0.3 — AE loss is now ~0.004 so needs rebalancing
          "oversample": 8.0, "rul_lr": 1e-3,
          "ae_epochs": 30, "patience": 25, "rul_epochs": 150},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_loader(X: np.ndarray, y: np.ndarray = None,
                batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    ds  = TensorDataset(X_t, torch.from_numpy(y).float()) if y is not None \
          else TensorDataset(X_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ── Losses ────────────────────────────────────────────────────────────────────

def ae_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)

""" def nasa_loss(pred, true, rul_cap=125.0):
    err = (pred - true) * rul_cap / rul_cap   # already normalized, just use directly
    # Actually simplifies to:
    err = pred - true    # both in [0,1], errors are small → no explosion
    loss = torch.where(
        err >= 0,
        torch.exp(err / (13.0 / rul_cap)) - 1,
        torch.exp(-err / (10.0 / rul_cap)) - 1,
    )
    return loss.mean() """


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Training phases ───────────────────────────────────────────────────────────

def train_phase1(model, ae_loader, val_loader, cfg, device):
    """Phase 1: AE pre-training on healthy-only windows."""
    print("\n── Phase 1: Autoencoder pre-training (healthy only) ──")
    opt     = Adam(model.autoencoder.parameters(), lr=cfg["ae_lr"])
    sched   = ReduceLROnPlateau(opt, patience=5, factor=0.5)
    stopper = EarlyStopping(patience=cfg["patience"])
    history = {"train": [], "val": []}

    for epoch in range(1, cfg["ae_epochs"] + 1):
        model.train()
        t_losses = []
        for (X_batch,) in ae_loader:
            X_batch = X_batch.to(device)
            recon, _ = model.autoencoder(X_batch)
            loss = ae_loss(recon, X_batch)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(device)
                recon, _ = model.autoencoder(X_batch)
                v_losses.append(ae_loss(recon, X_batch).item())

        tl, vl = np.mean(t_losses), np.mean(v_losses)
        history["train"].append(tl)
        history["val"].append(vl)
        sched.step(vl)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['ae_epochs']}  "
                  f"train={tl:.5f}  val={vl:.5f}")
        if stopper(vl):
            print(f"  Early stop at epoch {epoch}")
            break

    return history


def train_phase2(model, train_loader, val_loader, cfg, device, eval_cap=125.0):
    """
    Phase 2: joint AE + RUL fine-tuning.
    Uses F.mse_loss for training (stable gradients) — NOT asymmetric NASA loss.
    NASA loss is used only for final evaluation.
    """
    print("\n── Phase 2: Joint AE + RUL fine-tuning ──")
    opt     = Adam(model.parameters(), lr=cfg["rul_lr"])
    sched   = ReduceLROnPlateau(opt, patience=5, factor=0.5)
    stopper = EarlyStopping(patience=cfg["patience"])
    history = {"train": [], "val": []}

    for epoch in range(1, cfg["rul_epochs"] + 1):
        model.train()
        t_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            recon, rul_pred, _ = model(X_batch)
            # AFTER
            # TEMPORARY — remove nasa_loss entirely to get stable training
            loss = (ae_loss(recon, X_batch)
            + cfg["rul_weight"] * F.mse_loss(rul_pred, y_batch))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses, v_rul = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                recon, rul_pred, _ = model(X_batch)
                v_losses.append(ae_loss(recon, X_batch).item())
                v_rul.append(F.mse_loss(rul_pred, y_batch).item())

        tl, vl   = np.mean(t_losses), np.mean(v_losses)
        rul_rmse = np.sqrt(np.mean(v_rul)) * eval_cap   # denormalize for display
        history["train"].append(tl)
        history["val"].append(vl)
        sched.step(rul_rmse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['rul_epochs']}  "
                  f"train={tl:.5f}  val={vl:.5f}  "
                  f"RUL-RMSE={rul_rmse:.1f} cyc")
        if stopper(rul_rmse):
            print(f"  Early stop at epoch {epoch}")
            break

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, X_test, y_test, device, rul_cap=125.0):
    """RMSE and NASA score on the test set."""
    model.eval()
    X_t = torch.from_numpy(X_test).float().to(device)
    _, rul_pred, _ = model(X_t)
    pred = rul_pred.cpu().numpy() * rul_cap   # back to cycles
    true = y_test

    rmse  = float(np.sqrt(np.mean((pred - true) ** 2)))
    d     = pred - true
    score = float(np.where(d < 0,
                            np.exp(-d / 13) - 1,
                            np.exp(d / 10)  - 1).sum())

    print(f"\n  Test RMSE  = {rmse:.2f} cycles")
    print(f"  NASA score = {score:.1f}  (lower = better)")
    print(f"  Pred mean={pred.mean():.1f}  std={pred.std():.1f}")
    print(f"  True mean={true.mean():.1f}  std={true.std():.1f}")
    errors = np.abs(pred - true)
    worst  = np.argsort(errors)[-5:]
    print("  Worst 5: " + ", ".join(
        f"#{i} pred={pred[i]:.0f} true={true[i]:.0f}" for i in worst))
    return {"rmse": rmse, "score": score, "pred": pred, "true": true}


# ── Per-subset training ───────────────────────────────────────────────────────

def train_subset(subset: str, data_dir: str, ckpt_dir: str,
                  cfg: dict, device: str) -> None:
    print(f"\n{'='*55}")
    print(f"Training on {subset}")
    print(f"{'='*55}")

    # Apply subset-specific overrides
    cfg = {**cfg, **SUBSET_CONFIG.get(subset, {})}
    print(f"  rul_weight={cfg['rul_weight']}  "
          f"rul_lr={cfg['rul_lr']}  "
          f"oversample={cfg.get('oversample', 3.0)}  "
          f"ae_epochs={cfg['ae_epochs']}")

    pkl = Path(data_dir) / f"cmapss_{subset.lower()}.pkl"
    if not pkl.exists():
        print(f"\nERROR: {pkl} not found. Run: python preprocess.py --subset {subset}")
        return

    data      = load_data(str(pkl))
    n_sensors = len(data["sensor_cols"])
    train_cap = data.get("rul_train_cap", cfg["rul_cap"])
    eval_cap  = data.get("rul_cap", cfg["rul_cap"])

    print(f"Sensors: {n_sensors}  |  "
          f"Train: {data['X_train'].shape[0]:,} windows  |  "
          f"Val: {data['X_val'].shape[0]:,} windows  |  "
          f"Test: {data['X_test'].shape[0]} engines")

    ae_loader     = make_loader(data["X_ae"], batch_size=cfg["ae_batch"])
    val_ae_loader = make_loader(data["X_val"], batch_size=256, shuffle=False)

    # Keep this as before — normalized targets
    y_raw   = data["y_train"] / train_cap      # back to [0, 1]
    weights = np.where(y_raw >= 1.0, cfg.get("oversample", 3.0), 1.0)
    weights = weights / weights.sum()
    n   = len(data["X_train"])
    idx = np.random.default_rng(42).choice(n, size=n, replace=True, p=weights)
    X_bal, y_bal = data["X_train"][idx], y_raw[idx]
    train_loader = make_loader(X_bal, y_bal, batch_size=cfg["rul_batch"])

    val_rul_loader = make_loader(
    data["X_val"],
    data["y_val"] / train_cap,   # normalized
    batch_size=256, shuffle=False
    )

    model = FaultSenseModel(
        n_sensors=n_sensors,
        hidden=cfg["hidden"],
        latent=cfg["latent"],
        seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
        rul_cap=cfg["rul_cap"],
    ).to(device)
    print(f"Parameters: {count_params(model):,}")

    # ── MLflow run ────────────────────────────────────────────────────────
    mlflow.set_experiment("FaultSense")
    with mlflow.start_run(run_name=subset):

        mlflow.log_params({
            "subset":      subset,
            "hidden":      cfg["hidden"],
            "latent":      cfg["latent"],
            "seq_len":     cfg["seq_len"],
            "dropout":     cfg["dropout"],
            "rul_weight":  cfg["rul_weight"],
            "rul_lr":      cfg["rul_lr"],
            "ae_epochs":   cfg["ae_epochs"],
            "rul_epochs":  cfg["rul_epochs"],
            "oversample":  cfg.get("oversample", 3.0),
            "train_cap":   train_cap,
            "k_sigma":     cfg["k_sigma"],
            "n_sensors":   n_sensors,
        })

        t0 = time.time()
        h1 = train_phase1(model, ae_loader, val_ae_loader, cfg, device)
        h2 = train_phase2(model, train_loader, val_rul_loader, cfg, device,
                          eval_cap=eval_cap)
        train_time = time.time() - t0
        print(f"\nTraining time: {train_time/60:.1f} min")

        # Log loss curves
        for epoch, (tl, vl) in enumerate(zip(h1["train"], h1["val"]), 1):
            mlflow.log_metrics({"ae_train_loss": tl, "ae_val_loss": vl}, step=epoch)
        for epoch, (tl, vl) in enumerate(zip(h2["train"], h2["val"]), 1):
            mlflow.log_metrics({"rul_train_loss": tl, "rul_val_loss": vl}, step=epoch)

        model.calibrate_threshold(data["X_ae"], k_sigma=cfg["k_sigma"], device=device)

        test_results = evaluate_test(model, data["X_test"], data["y_test"],
                                      device, eval_cap)

        mlflow.log_metrics({
            "test_rmse":       round(test_results["rmse"], 2),
            "test_nasa_score": round(test_results["score"], 1),
            "threshold":       round(float(model.threshold), 5),
            "train_time_min":  round(train_time / 60, 2),
        })

        ckpt_path = Path(ckpt_dir) / f"faultsense_{subset.lower()}.pt"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save({
            "subset":       subset,
            "model_state":  model.state_dict(),
            "config":       cfg,
            "threshold":    model.threshold,
            "sensor_cols":  data["sensor_cols"],
            "n_conditions": data["n_conditions"],
            "history_ae":   h1,
            "history_rul":  h2,
            "test_results": test_results,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }, str(ckpt_path))
        print(f"\nCheckpoint saved → {ckpt_path}")
        print(f"MLflow run ID:    {mlflow.active_run().info.run_id}")


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = FaultSenseModel(
        n_sensors=len(ckpt["sensor_cols"]),
        hidden=cfg["hidden"],
        latent=cfg["latent"],
        seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
        rul_cap=cfg["rul_cap"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.threshold   = ckpt["threshold"]
    model.sensor_cols = ckpt["sensor_cols"]
    model.eval()
    return model, ckpt


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        print(f"MLflow tracking: {tracking_uri}")
    except Exception:
        mlflow.set_tracking_uri("mlruns")
        print("MLflow tracking: ./mlruns (local fallback)")

    script_dir = Path(__file__).resolve().parent
    data_dir   = args.data or str(script_dir / "data" / "processed")
    ckpt_dir   = args.out  or str(script_dir / "checkpoints")

    subsets = ALL_SUBSETS if args.all else (args.subset or ["FD001"])

    for subset in subsets:
        train_subset(subset, data_dir, ckpt_dir, DEFAULT_CONFIG.copy(), device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", nargs="+", choices=ALL_SUBSETS)
    parser.add_argument("--all",   action="store_true")
    parser.add_argument("--data",  default=None)
    parser.add_argument("--out",   default=None)
    main(parser.parse_args())
