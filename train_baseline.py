"""
train_baseline.py
-----------------
Ridge Regression baseline for RUL estimation — Algorithm 1.
 
Two modes:
  raw        : flatten the 30-cycle window → feature vector of size 30 × n_sensors
  engineered : extract [mean, std, last, delta, slope] per sensor
               → feature vector of size 5 × n_sensors
 
Usage:
  python train_baseline.py --subset FD001
  python train_baseline.py --subset FD001 --mode engineered
  python train_baseline.py --all
  python train_baseline.py --all --mode engineered
 
Output:
  checkpoints/ridge_{mode}_{subset.lower()}.pkl
  Results printed to stdout and logged to MLflow.
"""
 
import argparse
import pickle
import time
from pathlib import Path
 
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
 
import mlflow
 
SEED        = 42
RUL_CAP     = 130.0
ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
_HERE       = Path(__file__).resolve().parent
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def extract_features(X: np.ndarray, mode: str = "raw") -> np.ndarray:
    """
    X : (N, window, n_sensors)
    mode = 'raw'        → flatten each window → (N, window * n_sensors)
    mode = 'engineered' → [mean, std, last, delta, slope] per sensor
                          → (N, 5 * n_sensors)
 
    Algorithm 1 lines 6-10:
      engineered: F ← [mean, std, last, delta, slope] per sensor
      raw:        F ← flatten(W)
    """
    N, T, S = X.shape
 
    if mode == "raw":
        return X.reshape(N, T * S)
 
    if mode == "engineered":
        mean  = X.mean(axis=1)                     # (N, S)
        std   = X.std(axis=1)                      # (N, S)
        last  = X[:, -1, :]                        # (N, S) — last cycle
        delta = X[:, -1, :] - X[:, 0, :]          # (N, S) — end minus start
        # Slope: linear regression coefficient over timesteps per sensor
        t     = np.arange(T, dtype=np.float32)
        t_c   = t - t.mean()
        denom = (t_c ** 2).sum()
        # slope[i, s] = cov(t, X[i, :, s]) / var(t)
        slope = ((X - X.mean(axis=1, keepdims=True)) *
                 t_c[None, :, None]).sum(axis=1) / denom   # (N, S)
        return np.concatenate([mean, std, last, delta, slope], axis=1)
 
    raise ValueError(f"Unknown mode: {mode!r}. Choose 'raw' or 'engineered'.")
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NASA SCORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Official NASA asymmetric scoring function.
    d = pred − true
    S = Σ exp(-d/13)-1  if d < 0  (late, steep penalty)
        Σ exp( d/10)-1  if d ≥ 0  (early, mild penalty)
    Lower is better.
    """
    d = pred - true
    return float(np.where(d < 0, np.exp(-d / 13) - 1,
                                  np.exp( d / 10) - 1).sum())
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN TRAINING FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def train_ridge(subset: str, mode: str, data_dir: str,
                ckpt_dir: str, alpha: float = 1.0) -> dict:
    """
    Algorithm 1: Ridge Regression for RUL Estimation.
 
    Steps:
      1. Load preprocessed windows (X_train, y_train, X_test, y_test)
         from the same pickle used by the LSTM — identical preprocessing,
         identical train/val/test split, identical normalisation.
      2. Extract features (raw flatten or hand-crafted statistics).
      3. Fit z-score scaler on training features only (Algorithm 1 line 14).
      4. Train Ridge(alpha=1.0) on normalised features (Algorithm 1 line 15).
      5. Predict on test set last-window features (Algorithm 1 line 16).
      6. Evaluate RMSE + NASA score.
    """
    print(f"\n{'='*55}")
    print(f"Ridge Regression · {subset} · mode={mode}")
    print(f"{'='*55}")
 
    pkl = Path(data_dir) / f"cmapss_{subset.lower()}.pkl"
    if not pkl.exists():
        print(f"ERROR: {pkl} not found. Run preprocess.py --subset {subset}")
        return {}
 
    with open(str(pkl), "rb") as f:
        data = pickle.load(f)
 
    rul_cap    = float(data.get("rul_cap", RUL_CAP))
    n_sensors  = len(data["sensor_cols"])
    n_train    = data["X_train"].shape[0]
    n_test     = data["X_test"].shape[0]
    print(f"  Sensors: {n_sensors}  |  "
          f"Train windows: {n_train:,}  |  Test engines: {n_test}")
 
    # ── Algorithm 1 lines 3-13: build training feature matrix ────────────
    t0 = time.time()
    print(f"  Extracting features (mode={mode}) ...")
    F_train = extract_features(data["X_train"], mode)
    F_test  = extract_features(data["X_test"],  mode)
    y_train = data["y_train"]                   # raw cycles [0, rul_cap]
    y_test  = data["y_test"]                    # raw cycles [0, rul_cap]
 
    print(f"  Feature dim: {F_train.shape[1]}")
 
    # ── Algorithm 1 line 14: z-score on train only ────────────────────────
    print("  Fitting z-score scaler on training features ...")
    scaler  = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)
 
    # ── Algorithm 1 line 15: train Ridge(α=1.0) ───────────────────────────
    print(f"  Training Ridge(alpha={alpha}) ...")
    ridge = Ridge(alpha=alpha, random_state=SEED)
    ridge.fit(F_train, y_train)
 
    # ── Algorithm 1 line 16: predict on test last-window ──────────────────
    pred   = ridge.predict(F_test)
    pred   = np.clip(pred, 0.0, rul_cap)        # clip to valid RUL range
    true   = y_test
 
    rmse   = float(np.sqrt(mean_squared_error(true, pred)))
    score  = nasa_score(pred, true)
    mae    = float(np.abs(pred - true).mean())
    t_min  = (time.time() - t0) / 60
 
    d      = pred - true
    errors = pred - true
    abs_e  = np.abs(errors)
 
    print(f"\n  Test RMSE      = {rmse:.2f} cycles")
    print(f"  NASA score     = {score:.1f}  (lower is better)")
    print(f"  MAE            = {mae:.2f} cycles")
    print(f"  Pred mean={pred.mean():.1f}  std={pred.std():.1f}")
    print(f"  True mean={true.mean():.1f}  std={true.std():.1f}")
    print(f"  Mean error     = {errors.mean():.1f} cyc  "
          f"(positive=early, negative=late)")
 
    worst = np.argsort(abs_e)[-5:][::-1]
    print("  Worst 5 engines:")
    for i in worst:
        tag = "LATE" if errors[i] < 0 else "early"
        print(f"    #{i:3d}  pred={pred[i]:6.1f}  true={true[i]:6.1f}  "
              f"err={errors[i]:+.1f}  ({tag})")
 
    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt = {
        "subset":      subset,
        "mode":        mode,
        "alpha":       alpha,
        "ridge":       ridge,
        "scaler":      scaler,
        "sensor_cols": data["sensor_cols"],
        "rul_cap":     rul_cap,
        "pred":        pred,
        "true":        true,
        "rmse":        rmse,
        "nasa_score":  score,
        "mae":         mae,
        "train_time_min": t_min,
    }
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    out = Path(ckpt_dir) / f"ridge_{mode}_{subset.lower()}.pkl"
    with open(str(out), "wb") as f:
        pickle.dump(ckpt, f)
    print(f"\n  Checkpoint saved → {out}")
 
    return ckpt
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLFLOW LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def run_with_mlflow(subset: str, mode: str, data_dir: str,
                    ckpt_dir: str, alpha: float = 1.0):
    mlflow.set_experiment("FaultSense")
    with mlflow.start_run(run_name=f"Ridge_{mode}_{subset}"):
        mlflow.log_params({
            "model":   f"Ridge_{mode}",
            "subset":  subset,
            "mode":    mode,
            "alpha":   alpha,
            "seed":    SEED,
        })
        result = train_ridge(subset, mode, data_dir, ckpt_dir, alpha)
        if result:
            mlflow.log_metrics({
                "test_rmse":       round(result["rmse"], 2),
                "test_nasa_score": round(result["nasa_score"], 1),
                "test_mae":        round(result["mae"], 2),
                "train_time_min":  round(result["train_time_min"], 3),
            })
    return result
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPARISON TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def print_comparison(results: list[dict]):
    """Print a clean comparison table across all runs."""
    if not results:
        return
    print(f"\n{'='*65}")
    print(f"{'Model':<30} {'RMSE':>8} {'NASA':>10} {'MAE':>8}")
    print(f"{'-'*65}")
    for r in sorted(results, key=lambda x: x.get("rmse", 9999)):
        label = f"Ridge ({r['mode']}) · {r['subset']}"
        print(f"  {label:<28} {r['rmse']:>8.2f} {r['nasa_score']:>10.1f} "
              f"{r['mae']:>8.2f}")
    print(f"{'='*65}")
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def main(args):
    import os
    data_dir = args.data or str(_HERE / "data" / "processed")
    ckpt_dir = args.out  or str(_HERE / "checkpoints")
 
    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        print(f"MLflow tracking: {tracking_uri}")
    except Exception:
        mlflow.set_tracking_uri("mlruns")
        print("MLflow tracking: ./mlruns (local fallback)")
 
    subsets = ALL_SUBSETS if args.all else (args.subset or ["FD001"])
    modes   = ["raw", "engineered"] if args.both_modes else [args.mode]
 
    all_results = []
    for subset in subsets:
        for mode in modes:
            r = run_with_mlflow(subset, mode, data_dir, ckpt_dir,
                                alpha=args.alpha)
            if r:
                all_results.append(r)
 
    if len(all_results) > 1:
        print_comparison(all_results)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ridge Regression baseline for RUL estimation (Algorithm 1)"
    )
    parser.add_argument("--subset", nargs="+", choices=ALL_SUBSETS,
                        help="Subset(s) to train on (default: FD001)")
    parser.add_argument("--all",   action="store_true",
                        help="Train on all 4 subsets")
    parser.add_argument("--mode",  default="raw",
                        choices=["raw", "engineered"],
                        help="Feature mode (default: raw)")
    parser.add_argument("--both-modes", action="store_true",
                        help="Run both raw and engineered modes")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="L2 regularisation strength (default: 1.0)")
    parser.add_argument("--data",  default=None,
                        help="Processed data directory")
    parser.add_argument("--out",   default=None,
                        help="Checkpoint output directory")
    main(parser.parse_args())
