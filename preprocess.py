"""
data/preprocess.py
------------------
NASA CMAPSS preprocessing pipeline — all 4 subsets (FD001–FD004).

Preprocessing spec (matching paper):
  RUL cap      : 130 cycles (piecewise-linear — early cycles fixed at 130)
  Sensor drops : s1, s5, s10, s16, s18, s19 dropped for all subsets
                 s6 additionally dropped for FD001 only (near-zero variance)
                 → 14 sensors for FD001, 15 for FD002/FD003/FD004
  Normalisation: z-score, fit on training data only (no leakage)
  Train/val    : 80/20 split by engine ID
  Window       : 30 cycles, stride 1; label = RUL at last timestep
  Test windows : last 30 cycles per engine; zero-pad if engine < 30 cycles

Dataset properties:
  FD001 : 1 operating condition,  1 fault mode  (HPC degradation)
  FD002 : 6 operating conditions, 1 fault mode  (HPC degradation)
  FD003 : 1 operating condition,  2 fault modes (HPC + fan degradation)
  FD004 : 6 operating conditions, 2 fault modes (HPC + fan degradation)

Run:
  python preprocess.py                    # all 4 subsets
  python preprocess.py --subset FD001     # single subset
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

# Sensors to DROP per subset.
# Base drops (near-zero variance across all subsets): s1 s5 s10 s16 s18 s19
# FD001 additionally drops s6 (constant in single-condition HPC fault mode).
# FD003 RETAINS s6 — correlation -0.215 with RUL, nonlinear spread near
# failure under dual fault mode (HPC + fan degradation).
DROP_SENSORS = {
    "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],  # 14 sensors
    "FD002": ["s1", "s5",       "s10", "s16", "s18", "s19"],  # 15 sensors
    "FD003": ["s1", "s5",       "s10", "s16", "s18", "s19"],  # 15 sensors
    "FD004": ["s1", "s5",       "s10", "s16", "s18", "s19"],  # 15 sensors
}

N_CONDITIONS = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

RUL_CAP     = 130       # piecewise-linear cap — paper uses 130
WINDOW      = 30
VAL_FRAC    = 0.20
RANDOM_SEED = 42
ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


# ── Low-level loaders ────────────────────────────────────────────────────────

def load_raw(path: Path, subset: str) -> pd.DataFrame:
    """Load raw CMAPSS text file and drop near-zero-variance sensors."""
    df = pd.read_csv(str(path), sep=r"\s+", header=None,
                     names=COLUMNS, engine="python")
    df.drop(columns=DROP_SENSORS[subset], inplace=True)
    for col in df.columns:
        if col not in ("unit", "cycle"):
            df[col] = df[col].astype(float)
    return df


def sensor_cols_for(df: pd.DataFrame) -> list[str]:
    """Return sensor feature column names only — no cycle_norm."""
    return [c for c in df.columns if c.startswith("s")]


def add_rul(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Piecewise-linear RUL labelling.
    Cycles in the early healthy phase (remaining > rul_cap) are assigned
    RUL = rul_cap. After degradation onset the label decreases linearly
    to 0 at end of life.
    """
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    df.drop(columns="max_cycle", inplace=True)
    return df


# ── Operating condition handling (FD002 / FD004) ─────────────────────────────

def assign_conditions(df: pd.DataFrame, n_cond: int,
                       kmeans: KMeans = None) -> tuple[pd.DataFrame, KMeans]:
    """
    Cluster each cycle into one of n_cond operating conditions using K-means
    on the 3 op-setting columns. For single-condition subsets assigns 0.
    Pass a fitted kmeans to apply without refitting (avoids test leakage).
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


# ── Normalisation: z-score, fit on train only ────────────────────────────────

def fit_zscore(df_train: pd.DataFrame,
               sensor_cols: list[str],
               n_cond: int) -> dict:
    """
    Fit z-score statistics on training fleet only.

    Single-condition subsets: one global mean/std per sensor across all
    training engines and cycles.
    Multi-condition subsets (FD002/FD004): one mean/std per sensor per
    condition cluster — removes operating-regime offsets without conflating
    them with degradation signal.

    Returns a stats dict saved into the pickle for inference-time use.
    """
    stats = {}

    if n_cond == 1:
        for col in sensor_cols:
            mu  = float(df_train[col].mean())
            sig = float(df_train[col].std())
            sig = sig if sig > 1e-8 else 1.0
            stats[col] = {"mean": mu, "std": sig}
    else:
        for cond in range(n_cond):
            sub = df_train[df_train["condition"] == cond]
            stats[cond] = {}
            for col in sensor_cols:
                mu  = float(sub[col].mean())
                sig = float(sub[col].std())
                sig = sig if sig > 1e-8 else 1.0
                stats[cond][col] = {"mean": mu, "std": sig}

    return stats


def apply_zscore(df: pd.DataFrame,
                  sensor_cols: list[str],
                  stats: dict,
                  n_cond: int) -> pd.DataFrame:
    """
    Apply pre-fitted z-score stats to any split (train, val, test).
    Output is unbounded — no clipping applied.
    """
    df = df.copy()

    if n_cond == 1:
        for col in sensor_cols:
            mu  = stats[col]["mean"]
            sig = stats[col]["std"]
            df[col] = (df[col] - mu) / sig
    else:
        for cond in range(n_cond):
            mask = df["condition"] == cond
            for col in sensor_cols:
                mu  = stats[cond][col]["mean"]
                sig = stats[cond][col]["std"]
                df.loc[mask, col] = (df.loc[mask, col] - mu) / sig

    return df


# ── Window construction ──────────────────────────────────────────────────────

def make_windows(df: pd.DataFrame, sensor_cols: list[str],
                  window: int = WINDOW):
    """Sliding window extraction. Returns (X, y, engine_ids) arrays."""
    X_list, y_list, id_list = [], [], []
    for uid in df["unit"].unique():
        eng  = df[df["unit"] == uid]
        vals = eng[sensor_cols].values
        ruls = eng["rul"].values
        for i in range(len(vals) - window + 1):
            X_list.append(vals[i : i + window])
            y_list.append(ruls[i + window - 1])
            id_list.append(int(uid))              # ← store engine unit ID
    return (np.array(X_list,  dtype=np.float32),
            np.array(y_list,  dtype=np.float32),
            np.array(id_list, dtype=np.int32))


def train_val_split(df: pd.DataFrame,
                    val_frac: float = VAL_FRAC,
                    seed: int = RANDOM_SEED):
    """
    80/20 split by engine unit — model evaluated on entirely unseen
    degradation trajectories with no cycle-level overlap.
    """
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
                   kmeans: KMeans, norm_stats: dict,
                   n_cond: int, window: int = WINDOW):
    """
    Load test set and apply training normalisation without refitting.

    For each engine: take the LAST `window` cycles as the prediction window.
    Zero-pad at the FRONT if the engine has fewer than `window` cycles —
    handles short test trajectories without discarding any engines.

    True RUL values clipped to RUL_CAP to match training labels.
    """
    df = load_raw(test_path, subset)
    df, _ = assign_conditions(df, n_cond, kmeans=kmeans)
    df    = apply_zscore(df, sensor_cols, norm_stats, n_cond)

    true_rul  = np.loadtxt(str(rul_path), dtype=np.float32).clip(max=RUL_CAP)
    n_sensors = len(sensor_cols)

    X_list, y_list = [], []
    for i, uid in enumerate(df["unit"].unique()):
        eng  = df[df["unit"] == uid]
        vals = eng[sensor_cols].values.astype(np.float32)
        T    = len(vals)

        if T >= window:
            X_list.append(vals[-window:])
        else:
            # Zero-pad at the front for short engines
            pad = np.zeros((window - T, n_sensors), dtype=np.float32)
            X_list.append(np.vstack([pad, vals]))

        y_list.append(true_rul[i])

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_subset(subset: str, raw_dir: Path, out_dir: Path) -> None:
    print(f"\n{'='*50}")
    print(f"Processing {subset}")
    print(f"{'='*50}")

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

    # ── Load raw data ─────────────────────────────────────────────────────
    print(f"Loading train_{subset}.txt ...")
    df_raw      = load_raw(raw_dir / f"train_{subset}.txt", subset)
    sensor_cols = sensor_cols_for(df_raw)
    print(f"  {len(df_raw):,} rows, {df_raw['unit'].nunique()} engines, "
          f"{len(sensor_cols)} sensors: {sensor_cols}")

    # ── Operating condition clustering ────────────────────────────────────
    print(f"Assigning operating conditions (k={n_cond}) ...")
    df_raw, kmeans = assign_conditions(df_raw, n_cond)
    if n_cond > 1:
        counts = df_raw["condition"].value_counts().sort_index()
        for k, v in counts.items():
            print(f"  Condition {k}: {v:,} cycles ({v/len(df_raw)*100:.1f}%)")

    # ── Piecewise-linear RUL labels ───────────────────────────────────────
    print(f"Adding piecewise-linear RUL labels (cap={RUL_CAP}) ...")
    df_raw = add_rul(df_raw, rul_cap=RUL_CAP)

    # ── Drop op columns ───────────────────────────────────────────────────
    op_cols = [c for c in df_raw.columns if c.startswith("op")]
    if op_cols:
        df_raw.drop(columns=op_cols, inplace=True)

    # ── Train/val split BEFORE normalisation ──────────────────────────────
    print("Splitting train / val (80/20 by engine ID) ...")
    df_train, df_val = train_val_split(df_raw)
    print(f"  Train: {df_train['unit'].nunique()} engines  |  "
          f"Val: {df_val['unit'].nunique()} engines")

    # ── Fit z-score on training data only ────────────────────────────────
    print("Fitting z-score normalisation on training data ...")
    norm_stats = fit_zscore(df_train, sensor_cols, n_cond)

    # ── Apply normalisation ───────────────────────────────────────────────
    print("Normalising ...")
    df_train = apply_zscore(df_train, sensor_cols, norm_stats, n_cond)
    df_val   = apply_zscore(df_val,   sensor_cols, norm_stats, n_cond)

    # ── Healthy-only subset for AE pre-training ───────────────────────────
    df_healthy = df_train[df_train["rul"] == RUL_CAP].copy()
    print(f"  Healthy rows (RUL={RUL_CAP}): {len(df_healthy):,}")

    # ── Sliding windows ───────────────────────────────────────────────────
    print("Building sliding windows ...")
    X_train, y_train, ids_train = make_windows(df_train,   sensor_cols)
    X_val,   y_val,   ids_val   = make_windows(df_val,     sensor_cols)
    X_ae,    _,       _         = make_windows(df_healthy, sensor_cols)

    # ── Test set ──────────────────────────────────────────────────────────
    print("Loading test set ...")
    X_test, y_test = load_test_set(
        raw_dir / f"test_{subset}.txt",
        raw_dir / f"RUL_{subset}.txt",
        subset, sensor_cols, kmeans, norm_stats, n_cond,
    )

    # ── Sanity checks ─────────────────────────────────────────────────────
    print(f"\n  X_train : {X_train.shape}  "
          f"y ∈ [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  X_val   : {X_val.shape}  "
          f"y ∈ [{y_val.min():.0f}, {y_val.max():.0f}]")
    print(f"  X_ae    : {X_ae.shape}  (healthy-only)")
    print(f"  X_test  : {X_test.shape}  "
          f"y ∈ [{y_test.min():.0f}, {y_test.max():.0f}]")
    print(f"  X_train sensor mean={X_train.mean():.4f}  "
          f"std={X_train.std():.4f}  (expect ~0.0, ~1.0)")

    # ── Save ──────────────────────────────────────────────────────────────
    payload = {
        "subset":       subset,
        "n_conditions": n_cond,
        "sensor_cols":  sensor_cols,
        "window":       WINDOW,
        "rul_cap":      RUL_CAP,
        "kmeans":       kmeans,
        "norm_stats":   norm_stats,
        "X_train": X_train, "y_train": y_train, "ids_train": ids_train,
        "X_val":   X_val,   "y_val":   y_val,   "ids_val":   ids_val,
        "X_ae":    X_ae,
        "X_test":  X_test,  "y_test":  y_test,
    }
    out_path = out_dir / f"cmapss_{subset.lower()}.pkl"
    with open(str(out_path), "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved → {out_path}")


def run(subsets: list[str] = None, raw_dir: str = None, out_dir: str = None):
    script_dir = Path(__file__).resolve().parent

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
