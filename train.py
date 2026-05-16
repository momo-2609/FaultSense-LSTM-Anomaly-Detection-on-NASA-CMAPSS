"""
train.py
--------
Training script for FaultSense simple LSTM (paper Algorithm 5).
No autoencoder pre-training phase — single RUL regression stage only.

Training spec (matching paper):
  Optimiser  : RMSprop, lr=0.001, weight_decay=1e-5
  Loss       : MSE(rul_pred, rul_true)  — targets normalised to [0, 1]
  Batch size : 64
  Scheduler  : ReduceLROnPlateau, factor=0.5, patience=5
  Early stop : patience=20 epochs on val MSE loss
  Max epochs : 200
  Seed       : 42

Usage:
  python train.py                         # train on FD001
  python train.py --subset FD002
  python train.py --subset FD001 FD002
  python train.py --all
"""

import argparse
import os
import pickle
import random
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
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.lstm_autoencoder import FaultSenseModel, count_params

import mlflow

# ── Global seed ───────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model — matches Algorithm 5
    "hidden":        32,      # LSTM hidden size
    "dropout":       0.5,     # dropout on last hidden state
    "rul_cap":      130.0,    # matches preprocess RUL_CAP

    # Training — matches paper spec
    "epochs":       200,      # max epochs
    "lr":           1e-3,     # RMSprop learning rate
    "weight_decay": 1e-5,     # RMSprop weight decay
    "batch_size":    64,      # batch size
    "patience":      20,      # early stop patience
    "sched_factor":  0.5,     # LR scheduler reduction factor
    "sched_patience": 5,      # LR scheduler patience
    "grad_clip":     1.0,     # gradient norm clipping

    # Sampling
    "oversample":    2.0,     # weight multiplier for high-RUL windows

    # Anomaly detection (stub — no AE)
    "k_sigma":       2.5,

    # Legacy keys kept so load_checkpoint stays compatible
    "latent":        32,
    "seq_len":       30,
    "ae_epochs":      0,
    "ae_lr":         1e-3,
    "ae_wd":         1e-5,
    "ae_batch":       64,
    "rul_weight":     1.0,
    "rul_lr":         1e-3,
    "rul_wd":         1e-5,
    "rul_batch":       64,
    "rul_epochs":    200,
    "sched_patience_rul": 5,
}

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Per-subset overrides — empty by default, add here to tune per subset
SUBSET_CONFIG = {
    "FD001": {},
    "FD002": {},
    "FD003": {},
    "FD004": {},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop when val loss does not improve for `patience` consecutive epochs."""
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
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


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, cfg, device, eval_cap=130.0):
    """
    Single-stage RUL training — Algorithm 5.

    Loss       : MSE(rul_pred, rul_true)  both in [0, 1]
    Optimiser  : RMSprop(lr, weight_decay)
    Scheduler  : ReduceLROnPlateau(factor=0.5, patience=5) on val MSE
    Early stop : patience=20 on val MSE
    Best weights saved and restored before returning.
    """
    print("\n── Training LSTM RUL model ──")
    opt = RMSprop(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    sched   = ReduceLROnPlateau(opt, factor=cfg["sched_factor"],
                                 patience=cfg["sched_patience"], verbose=False)
    stopper = EarlyStopping(patience=cfg["patience"])
    history = {"train": [], "val": []}

    best_val   = np.inf
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        t_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            _, rul_pred, _   = model(X_batch)
            loss = F.mse_loss(rul_pred, y_batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            t_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, rul_pred, _   = model(X_batch)
                v_losses.append(F.mse_loss(rul_pred, y_batch).item())

        tl       = np.mean(t_losses)
        vl       = np.mean(v_losses)
        rul_rmse = np.sqrt(vl) * eval_cap   # display in cycles

        history["train"].append(tl)
        history["val"].append(vl)
        sched.step(vl)

        # Save best
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg['epochs']}  "
                  f"train={tl:.5f}  val={vl:.5f}  "
                  f"RUL-RMSE={rul_rmse:.1f} cyc")

        if stopper(vl):
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights (val={best_val:.5f}  "
              f"RMSE={np.sqrt(best_val)*eval_cap:.1f} cyc)")

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, X_test, y_test, device, rul_cap=130.0):
    """
    RMSE and official NASA scoring function on the test set.

    NASA score = Σ_i f(d_i)
      d_i = pred_i - true_i
      f(d) = exp(-d/13) - 1   if d < 0  (late — steep penalty)
      f(d) = exp( d/10) - 1   if d >= 0 (early — mild penalty)
    Lower is better.
    """
    model.eval()
    X_t = torch.from_numpy(X_test).float().to(device)
    _, rul_pred, _ = model(X_t)
    pred = rul_pred.cpu().numpy() * rul_cap   # [0,1] → cycles
    true = y_test                              # already in cycles

    rmse  = float(np.sqrt(np.mean((pred - true) ** 2)))
    d     = pred - true
    score = float(np.where(
        d < 0,
        np.exp(-d / 13) - 1,
        np.exp( d / 10) - 1,
    ).sum())

    errors  = pred - true
    abs_err = np.abs(errors)

    print(f"\n  Test RMSE      = {rmse:.2f} cycles")
    print(f"  NASA score     = {score:.1f}  (lower is better)")
    print(f"  Pred mean={pred.mean():.1f}  std={pred.std():.1f}")
    print(f"  True mean={true.mean():.1f}  std={true.std():.1f}")
    print(f"  Mean error     = {errors.mean():.1f} cyc  "
          f"(positive=early, negative=late)")
    print(f"  Mean |error|   = {abs_err.mean():.1f} cyc")

    worst = np.argsort(abs_err)[-5:][::-1]
    print("  Worst 5 engines:")
    for i in worst:
        tag = "LATE" if errors[i] < 0 else "early"
        print(f"    #{i:3d}  pred={pred[i]:6.1f}  true={true[i]:6.1f}  "
              f"err={errors[i]:+.1f}  ({tag})")

    return {"rmse": rmse, "score": score, "pred": pred, "true": true}


# ── Per-subset training ───────────────────────────────────────────────────────

def train_subset(subset: str, data_dir: str, ckpt_dir: str,
                  cfg: dict, device: str) -> None:
    print(f"\n{'='*55}")
    print(f"Training on {subset}")
    print(f"{'='*55}")

    cfg = {**cfg, **SUBSET_CONFIG.get(subset, {})}
    print(f"  lr={cfg['lr']}  wd={cfg['weight_decay']}  "
          f"batch={cfg['batch_size']}  patience={cfg['patience']}  "
          f"max_epochs={cfg['epochs']}  hidden={cfg['hidden']}  "
          f"dropout={cfg['dropout']}")

    pkl = Path(data_dir) / f"cmapss_{subset.lower()}.pkl"
    if not pkl.exists():
        print(f"\nERROR: {pkl} not found. "
              f"Run: python preprocess.py --subset {subset}")
        return

    data      = load_data(str(pkl))
    n_sensors = len(data["sensor_cols"])
    eval_cap  = float(data.get("rul_cap", cfg["rul_cap"]))
    train_cap = eval_cap

    print(f"Sensors: {n_sensors}  |  "
          f"Train: {data['X_train'].shape[0]:,} windows  |  "
          f"Val: {data['X_val'].shape[0]:,} windows  |  "
          f"Test: {data['X_test'].shape[0]} engines")

    # ── Data loaders ──────────────────────────────────────────────────────
    # Targets normalised to [0, 1] — model output is also clamped to [0, 1]
    y_raw   = data["y_train"] / train_cap
    weights = np.where(y_raw >= 0.99, cfg.get("oversample", 2.0), 1.0)
    weights = weights / weights.sum()
    rng     = np.random.default_rng(SEED)
    idx     = rng.choice(len(data["X_train"]), size=len(data["X_train"]),
                          replace=True, p=weights)
    X_bal, y_bal = data["X_train"][idx], y_raw[idx]
    train_loader = make_loader(X_bal, y_bal, batch_size=cfg["batch_size"])

    val_loader = make_loader(
        data["X_val"],
        data["y_val"] / train_cap,
        batch_size=256, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    set_seed(SEED)
    model = FaultSenseModel(
        n_sensors=n_sensors,
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
        rul_cap=cfg["rul_cap"],
    ).to(device)
    print(f"Parameters: {count_params(model):,}")

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_experiment("FaultSense")
    with mlflow.start_run(run_name=subset):

        mlflow.log_params({
            "subset":         subset,
            "hidden":         cfg["hidden"],
            "dropout":        cfg["dropout"],
            "lr":             cfg["lr"],
            "weight_decay":   cfg["weight_decay"],
            "batch_size":     cfg["batch_size"],
            "epochs":         cfg["epochs"],
            "patience":       cfg["patience"],
            "sched_factor":   cfg["sched_factor"],
            "sched_patience": cfg["sched_patience"],
            "oversample":     cfg.get("oversample", 2.0),
            "rul_cap":        train_cap,
            "n_sensors":      n_sensors,
            "seed":           SEED,
        })

        t0   = time.time()
        hist = train_model(model, train_loader, val_loader, cfg, device,
                           eval_cap=eval_cap)
        train_time = time.time() - t0
        print(f"\nTraining time: {train_time/60:.1f} min")

        # Log curves
        for epoch, (tl, vl) in enumerate(zip(hist["train"], hist["val"]), 1):
            mlflow.log_metrics({"rul_train_loss": tl, "rul_val_loss": vl},
                               step=epoch)

        # Threshold stub (no AE)
        model.calibrate_threshold(data["X_ae"], k_sigma=cfg["k_sigma"],
                                   device=device)

        test_results = evaluate_test(model, data["X_test"], data["y_test"],
                                      device, eval_cap)

        mlflow.log_metrics({
            "test_rmse":       round(test_results["rmse"], 2),
            "test_nasa_score": round(test_results["score"], 1),
            "train_time_min":  round(train_time / 60, 2),
        })

        ckpt_path = Path(ckpt_dir) / f"faultsense_{subset.lower()}.pt"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save({
            "subset":        subset,
            "model_state":   model.state_dict(),
            "config":        cfg,
            "threshold":     model.threshold,
            "sensor_cols":   data["sensor_cols"],
            "n_conditions":  data["n_conditions"],
            "norm_stats":    data.get("norm_stats"),
            "history_ae":    {"train": [], "val": []},   # empty — no AE phase
            "history_rul":   hist,
            "test_results":  test_results,
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
        hidden=cfg.get("hidden", 32),
        dropout=cfg.get("dropout", 0.5),
        rul_cap=cfg.get("rul_cap", 130.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.threshold   = ckpt.get("threshold", 0.0)
    model.sensor_cols = ckpt["sensor_cols"]
    model.eval()
    return model, ckpt


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--data",   default=None)
    parser.add_argument("--out",    default=None)
    main(parser.parse_args())
