"""
data/preprocess.py
------------------
NASA CMAPSS preprocessing pipeline — all 4 subsets (FD001–FD004).

Dataset properties:
  FD001 : 1 operating condition,  1 fault mode  (HPC degradation)
  FD002 : 6 operating conditions, 1 fault mode  (HPC degradation)
  FD003 : 1 operating condition,  2 fault modes (HPC + fan degradation)
  FD004 : 6 operating conditions, 2 fault modes (HPC + fan degradation)

Key difference for FD002/FD004:
  6 operating conditions means raw sensor values jump between regimes.
  Without condition-aware normalisation, a high-load cycle looks anomalous
  next to a low-load cycle — the model sees operating conditions, not faults.
  Fix: cluster cycles into 6 conditions, normalise within condition × engine.

Run:
  python data/preprocess.py               # processes all 4 subsets
  python data/preprocess.py --subset FD001  # single subset

Download from:
  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
  Turbofan Engine Degradation Simulation Data Set
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

# ── Column definitions ───────────────────────────────────────────────────────
COLUMNS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Per-subset: sensors to DROP (near-zero variance, verified empirically)
# FD001/FD003 share the same drop list (1 op condition)
# FD002/FD004 keep op columns longer — they're needed for clustering
DROP_SENSORS = {
    "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD002": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD003": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
    "FD004": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
}

# Number of operating condition clusters per subset
N_CONDITIONS = {
    "FD001": 1,   # single condition — no clustering needed
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

RUL_CAP     = 125
WINDOW      = 30
VAL_FRAC    = 0.20
RANDOM_SEED = 42
ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


# ── Low-level loaders ────────────────────────────────────────────────────────

def load_raw(path: Path, subset: str) -> pd.DataFrame:
    """Load raw CMAPSS text file. Keep op columns for clustering."""
    df = pd.read_csv(str(path), sep=r"\s+", header=None,
                     names=COLUMNS, engine="python")
    # Drop only the near-zero-variance sensors — keep op cols for now
    drop = DROP_SENSORS[subset]
    df.drop(columns=drop, inplace=True)
    # Cast all sensor + op columns to float64 to avoid pandas FutureWarning
    # when writing normalised float values back into int-typed columns
    for col in df.columns:
        if col not in ("unit", "cycle"):
            df[col] = df[col].astype(float)
    return df


def sensor_cols_for(df: pd.DataFrame) -> list[str]:
    """Return feature column names: sensors + cycle_norm position feature."""
    cols = [c for c in df.columns if c.startswith("s")]
    if "cycle_norm" in df.columns:
        cols = cols + ["cycle_norm"]   # append cycle position as final feature
    return cols


def add_rul(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    # Normalised cycle position [0, 1]: 0 = engine start, 1 = end of life
    # This gives the RUL head explicit temporal context — essential for
    # generalising to unseen test engines where only the last window is given
    df["cycle_norm"] = df["cycle"] / df["max_cycle"]
    df.drop(columns="max_cycle", inplace=True)
    return df


# ── Operating condition handling ─────────────────────────────────────────────

def assign_conditions(df: pd.DataFrame, n_cond: int,
                       kmeans: KMeans = None) -> tuple[pd.DataFrame, KMeans]:
    """
    Cluster cycles into operating condition groups using K-means on
    the 3 op-setting columns. Returns df with 'condition' column added.

    If kmeans is provided (fitted on training data), applies it to new data
    (test set) without refitting — critical to avoid leakage.
    """
    if n_cond == 1:
        df = df.copy()
        df["condition"] = 0
        return df, None

    op_cols = ["op1", "op2", "op3"]
    X_op = df[op_cols].values

    if kmeans is None:
        kmeans = KMeans(n_clusters=n_cond, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(X_op)

    df = df.copy()
    df["condition"] = kmeans.predict(X_op)
    return df, kmeans


def normalize_per_engine_condition(df: pd.DataFrame,
                                    sensor_cols: list[str],
                                    n_cond: int) -> pd.DataFrame:
    """
    Z-score normalise each sensor within (engine unit × condition) group.

    For single-condition subsets: equivalent to per-engine normalisation.
    For multi-condition subsets: removes both operating-regime offset AND
    unit-to-unit manufacturing variation, leaving only degradation signal.
    """
    df = df.copy()
    group_cols = ["unit", "condition"] if n_cond > 1 else ["unit"]

    for group_keys, group_df in df.groupby(group_cols):
        idx = group_df.index
        for col in sensor_cols:
            mu  = group_df[col].mean()
            sig = group_df[col].std() + 1e-8
            df.loc[idx, col] = (group_df[col] - mu) / sig

    return df


# ── Window construction ──────────────────────────────────────────────────────

def make_windows(df: pd.DataFrame, sensor_cols: list[str],
                  window: int = WINDOW):
    """Sliding window extraction. Returns (X, y) arrays."""
    X_list, y_list = [], []
    for uid in df["unit"].unique():
        eng  = df[df["unit"] == uid]
        vals = eng[sensor_cols].values
        ruls = eng["rul"].values
        for i in range(len(vals) - window):
            X_list.append(vals[i : i + window])
            y_list.append(ruls[i + window - 1])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


def train_val_split(df: pd.DataFrame,
                    val_frac: float = VAL_FRAC,
                    seed: int = RANDOM_SEED):
    """Split by engine unit to prevent trajectory leakage."""
    rng   = np.random.default_rng(seed)
    units = df["unit"].unique().copy()
    rng.shuffle(units)
    n_val       = int(len(units) * val_frac)
    val_units   = units[:n_val]
    train_units = units[n_val:]
    return (df[df["unit"].isin(train_units)].copy(),
            df[df["unit"].isin(val_units)].copy())


def load_test_set(test_path: Path, rul_path: Path,
                   subset: str, sensor_cols: list[str],
                   kmeans: KMeans, window: int = WINDOW):
    """
    Load test set and apply the SAME normalisation fitted on training data.
    Uses last `window` cycles of each engine as the prediction window.
    cycle_norm for test engines: position within the observed (truncated) life.
    """
    n_cond = N_CONDITIONS[subset]
    df = load_raw(test_path, subset)
    # Add cycle_norm before condition assignment (uses only observed cycles)
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["cycle_norm"] = df["cycle"] / df["max_cycle"]
    df.drop(columns="max_cycle", inplace=True)
    df, _ = assign_conditions(df, n_cond, kmeans=kmeans)  # apply fitted kmeans
    # sensor_cols includes cycle_norm — normalise sensors only (not cycle_norm)
    sensor_only = [c for c in sensor_cols if c != "cycle_norm"]
    df = normalize_per_engine_condition(df, sensor_only, n_cond)

    true_rul = np.loadtxt(str(rul_path), dtype=np.float32).clip(max=RUL_CAP)

    X_list, y_list = [], []
    for i, uid in enumerate(df["unit"].unique()):
        eng  = df[df["unit"] == uid]
        vals = eng[sensor_cols].values
        if len(vals) >= window:
            X_list.append(vals[-window:])
            y_list.append(true_rul[i])

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_subset(subset: str, raw_dir: Path, out_dir: Path) -> None:
    """Full preprocessing pipeline for one CMAPSS subset."""
    print(f"\n{'='*50}")
    print(f"Processing {subset}")
    print(f"{'='*50}")

    # Check files
    needed  = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    missing = [f for f in needed if not (raw_dir / f).exists()]
    if missing:
        print(f"\nERROR: Missing files in {raw_dir}:")
        for f in missing:
            print(f"  - {f}")
        print("\nDownload from:")
        print("  https://ti.arc.nasa.gov/tech/dash/groups/"
              "pcoe/prognostic-data-repository/")
        raise SystemExit(1)

    n_cond = N_CONDITIONS[subset]

    # ── Load training data ────────────────────────────────────────────────
    print(f"Loading train_{subset}.txt ...")
    df_raw = load_raw(raw_dir / f"train_{subset}.txt", subset)
    sensor_cols = sensor_cols_for(df_raw)
    print(f"  {len(df_raw):,} rows, {df_raw['unit'].nunique()} engines, "
          f"{len(sensor_cols)} sensors")

    # ── Operating condition clustering (FD002/FD004 only) ────────────────
    if "cycle_norm" not in sensor_cols:
        sensor_cols = sensor_cols + ["cycle_norm"]
    print(f"Assigning operating conditions (k={n_cond}) ...")
    df_raw, kmeans = assign_conditions(df_raw, n_cond)
    if n_cond > 1:
        counts = df_raw["condition"].value_counts().sort_index()
        for k, v in counts.items():
            print(f"  Condition {k}: {v:,} cycles ({v/len(df_raw)*100:.1f}%)")

    # ── Add RUL labels ────────────────────────────────────────────────────
    print("Adding RUL labels ...")
    df_raw = add_rul(df_raw)

    # ── Normalise (within engine × condition) ─────────────────────────────
    # Normalise sensor readings only — cycle_norm is already in [0,1]
    print("Normalising ...")
    sensor_only = [c for c in sensor_cols if c != "cycle_norm"]
    df_norm = normalize_per_engine_condition(df_raw, sensor_only, n_cond)

    # ── Drop op columns — not needed after normalisation ──────────────────
    op_cols = [c for c in df_norm.columns if c.startswith("op")]
    if op_cols:
        df_norm.drop(columns=op_cols, inplace=True)

    # ── Train / val split (by engine unit) ────────────────────────────────
    print("Splitting train / val ...")
    df_train, df_val = train_val_split(df_norm)
    print(f"  Train: {df_train['unit'].nunique()} engines  |  "
          f"Val: {df_val['unit'].nunique()} engines")

    # Healthy-only subset for AE pre-training
    df_healthy = df_train[df_train["rul"] == RUL_CAP].copy()
    print(f"  Healthy windows source: {len(df_healthy):,} rows "
          f"(RUL = {RUL_CAP})")

    # ── Build windows ─────────────────────────────────────────────────────
    print("Building sliding windows ...")
    X_train, y_train = make_windows(df_train,   sensor_cols)
    X_val,   y_val   = make_windows(df_val,     sensor_cols)
    X_ae,    _       = make_windows(df_healthy, sensor_cols)

    # ── Test set ─────────────────────────────────────────────────────────
    print("Loading test set ...")
    # Must pass the SAME kmeans fitted on training data
    X_test, y_test = load_test_set(
        raw_dir / f"test_{subset}.txt",
        raw_dir / f"RUL_{subset}.txt",
        subset, sensor_cols, kmeans,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    payload = {
        "subset":      subset,
        "n_conditions": n_cond,
        "sensor_cols": sensor_cols,
        "window":      WINDOW,
        "rul_cap":     RUL_CAP,
        "kmeans":      kmeans,          # needed to preprocess new data at inference
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_ae":    X_ae,
        "X_test":  X_test,  "y_test":  y_test,
    }
    out_path = out_dir / f"cmapss_{subset.lower()}.pkl"
    with open(str(out_path), "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved → {out_path}")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_ae    : {X_ae.shape}  (healthy-only)")
    print(f"  X_test  : {X_test.shape}")


def run(subsets: list[str] = None, raw_dir: str = None, out_dir: str = None):
    script_dir = Path(__file__).resolve().parent # project root

    if raw_dir is None:
        raw_dir = script_dir / "data" / "raw"
    if out_dir is None:
        out_dir = script_dir / "data" / "processed"

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if subsets is None:
        subsets = ALL_SUBSETS

    for subset in subsets:
        process_subset(subset, raw_dir, out_dir)

    print(f"\n{'='*50}")
    print(f"All done. Files in: {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NASA CMAPSS dataset")
    parser.add_argument(
        "--subset", nargs="+", default=None,
        choices=ALL_SUBSETS,
        help="Which subsets to process (default: all 4)"
    )
    parser.add_argument("--raw",  default=None, help="Raw data directory")
    parser.add_argument("--out",  default=None, help="Output directory")
    args = parser.parse_args()
    run(subsets=args.subset, raw_dir=args.raw, out_dir=args.out)
