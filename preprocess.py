"""
data/preprocess.py
------------------
NASA CMAPSS FD001 preprocessing pipeline.

Download the dataset from:
  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
Files needed: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
Place them in data/raw/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

# ── Column definitions ──────────────────────────────────────────────────────
COLUMNS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001 (empirically verified)
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Sensors used for modelling (14 remaining)
SENSOR_COLS = [c for c in COLUMNS if c.startswith("s") and c not in DROP_SENSORS]

RUL_CAP    = 125   # piecewise linear cap — cycles above this are "equally healthy"
WINDOW     = 30    # timesteps per input window
BATCH_SIZE = 64
VAL_FRAC   = 0.20
RANDOM_SEED = 42


def load_raw(path: str) -> pd.DataFrame:
    """Load a CMAPSS text file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df.drop(columns=["op1", "op2", "op3"] + DROP_SENSORS, inplace=True)
    return df


def add_rul(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """Compute and cap remaining useful life labels."""
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    df.drop(columns="max_cycle", inplace=True)
    return df


def normalize_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each sensor within each engine unit."""
    df = df.copy()
    for uid in df["unit"].unique():
        mask = df["unit"] == uid
        for col in SENSOR_COLS:
            mu  = df.loc[mask, col].mean()
            sig = df.loc[mask, col].std() + 1e-8
            df.loc[mask, col] = (df.loc[mask, col] - mu) / sig
    return df


def make_windows(df: pd.DataFrame, window: int = WINDOW):
    """
    Slide a window of length `window` over each engine trajectory.
    Returns:
        X : np.ndarray  (N, window, n_sensors)
        y : np.ndarray  (N,)  RUL at the last timestep of each window
    """
    X_list, y_list = [], []
    for uid in df["unit"].unique():
        eng  = df[df["unit"] == uid]
        vals = eng[SENSOR_COLS].values   # (T, 14)
        ruls = eng["rul"].values         # (T,)
        for i in range(len(vals) - window):
            X_list.append(vals[i : i + window])
            y_list.append(ruls[i + window - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_val_split(df: pd.DataFrame, val_frac: float = VAL_FRAC, seed: int = RANDOM_SEED):
    """Split by engine unit to avoid trajectory leakage."""
    rng   = np.random.default_rng(seed)
    units = df["unit"].unique()
    rng.shuffle(units)
    n_val      = int(len(units) * val_frac)
    val_units  = units[:n_val]
    train_units = units[n_val:]
    return df[df["unit"].isin(train_units)], df[df["unit"].isin(val_units)]


def load_test_rul(test_path: str, rul_path: str, window: int = WINDOW):
    """
    Load test set with ground-truth RUL labels.
    The test set is truncated (engines haven't failed yet).
    RUL_FD001.txt gives the true RUL at the last cycle for each test engine.
    """
    df_test = load_raw(test_path)
    df_test = normalize_per_engine(df_test)
    true_rul = np.loadtxt(rul_path, dtype=np.float32).clip(max=RUL_CAP)

    X_list, y_list = [], []
    for i, uid in enumerate(df_test["unit"].unique()):
        eng  = df_test[df_test["unit"] == uid]
        vals = eng[SENSOR_COLS].values
        if len(vals) >= window:
            X_list.append(vals[-window:])   # last window only
            y_list.append(true_rul[i])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def run(raw_dir=None, out_dir=None):
    base_dir = Path(__file__).resolve().parent

    raw_dir = Path(raw_dir) if raw_dir else base_dir
    out_dir = Path(out_dir) if out_dir else base_dir / "processed"

    # 🔴 THIS LINE IS REQUIRED
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data …")
    df_train_raw = load_raw(raw_dir / "train_FD001.txt")

    print("Adding RUL labels …")
    df_train_raw = add_rul(df_train_raw)

    print("Normalizing per engine …")
    df_norm = normalize_per_engine(df_train_raw)

    print("Splitting train / val …")
    df_train, df_val = train_val_split(df_norm)

    # Autoencoder training uses ONLY healthy windows (RUL == cap)
    df_healthy = df_train[df_train["rul"] == RUL_CAP]

    print("Building windows …")
    X_train, y_train = make_windows(df_train)
    X_val,   y_val   = make_windows(df_val)
    X_ae,    _       = make_windows(df_healthy)   # healthy only, for AE training

    print("Loading test set …")
    X_test, y_test = load_test_rul(
        raw_dir / "test_FD001.txt",
        raw_dir / "RUL_FD001.txt",
    )

    # Save everything
    payload = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_ae":    X_ae,
        "X_test":  X_test,  "y_test":  y_test,
        "sensor_cols": SENSOR_COLS,
        "window":  WINDOW,
    }
    out_path = out_dir / "cmapss_fd001.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved to {out_path}")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_ae    : {X_ae.shape}  (healthy-only for autoencoder)")
    print(f"  X_test  : {X_test.shape}")


if __name__ == "__main__":
    run(raw_dir=r"C:\Users\41789\OneDrive\Bureau\Projets\NASA")
