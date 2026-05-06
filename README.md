# FaultSense

**End-to-end predictive maintenance system** — LSTM Autoencoder anomaly detection and Remaining Useful Life (RUL) prediction on NASA CMAPSS turbofan engine data, with an Extended Kalman Filter baseline, REST API, MLflow experiment tracking, and a Streamlit dashboard.

---

## Results

| Subset | Conditions | Fault modes | Test RMSE (cycles) | NASA Score |
|--------|-----------|-------------|-------------------|------------|
| FD001  | 1         | 1           | ~50               | ~15 000    |
| FD002  | 6         | 1           | —                 | —          |
| FD003  | 1         | 2           | —                 | —          |
| FD004  | 6         | 2           | —                 | —          |

> Lower is better for both metrics. FD001 trained; FD002–FD004 in progress.

---

## Architecture

```
Raw sensor data (21 sensors × N cycles)
        │
        ▼
  preprocess.py
  ├── Drop zero-variance sensors  (14 → 15 features incl. cycle_norm)
  ├── K-means operating condition clustering  (FD002/FD004 only)
  ├── Per-engine × condition Z-score normalisation
  └── Sliding window extraction  (seq_len=30)
        │
        ▼
  FaultSenseModel  (models/lstm_autoencoder.py)
  ├── Encoder:  BiLSTM(128) → Dropout → LSTM(64) → Dense(32, tanh)
  ├── Decoder:  RepeatVector → LSTM(64) → LSTM(128) → Dense(n_sensors)
  └── RUL head: MLP(32 → 16 → 1) attached to latent z
        │
  Two-phase training (train.py)
  ├── Phase 1: AE pre-training on healthy-only windows
  └── Phase 2: Joint AE + RUL fine-tuning with asymmetric NASA loss
        │
        ▼
  Inference
  ├── Anomaly score  =  MSE(input, reconstruction)
  ├── Alarm          =  score > μ + 2.5σ  (calibrated on healthy data)
  └── RUL prediction =  MLP(z) × 125 cycles

  EKFBaseline  (models/ekf_baseline.py)
  └── Per-sensor Kalman filter, Mahalanobis innovation score, EMA smoothing
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/momo-2609/faultsense
cd faultsense
pip install -r requirements.txt
```

### 2. Download data

Download the **Turbofan Engine Degradation Simulation Data Set** from the
[NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
and place the `.txt` files in `data/raw/`:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
└── ... (FD002, FD003, FD004)
```

### 3. Preprocess

```bash
python preprocess.py --subset FD001        # single subset
python preprocess.py                        # all 4 subsets
```

### 4. Train

```bash
python train.py --subset FD001             # train FD001
python train.py --all                      # train all 4 subsets
```

Training logs to MLflow automatically if the server is running (see below).

### 5. Launch the dashboard

```bash
streamlit run app.py
```

---

## REST API + MLflow

Start the full stack (API on `:8000`, MLflow on `:5000`) with Docker:

```bash
docker compose up -d
```

### Endpoints

| Method | Endpoint            | Description                              |
|--------|---------------------|------------------------------------------|
| GET    | `/health`           | Liveness check + loaded subsets          |
| GET    | `/models`           | List checkpoints with RMSE / NASA score  |
| POST   | `/predict`          | Single window → anomaly score + RUL      |
| POST   | `/predict/batch`    | Batch inference (up to 512 windows)      |
| GET    | `/metrics/{subset}` | Stored test metrics for a subset         |

Interactive docs: **`http://localhost:8000/docs`**

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subset": "FD001", "window": [[...30 timesteps × 15 sensors...]]}'
```

```json
{
  "anomaly_score": 0.312,
  "rul_cycles": 47.3,
  "is_anomaly": false,
  "threshold": 0.434,
  "per_sensor_mse": [0.21, 0.18, ...],
  "top_sensor_idx": 3,
  "latency_ms": 8.2
}
```

### MLflow experiment tracking

Open **`http://localhost:5000`** to compare runs side by side:

- All hyperparameters logged per run
- Per-epoch AE and RUL loss curves
- Final `test_rmse`, `test_nasa_score`, `threshold`, `train_time_min`

---

## Project structure

```
faultsense/
├── models/
│   ├── lstm_autoencoder.py   # FaultSenseModel, LSTMEncoder, RULHead
│   └── ekf_baseline.py       # EKFBaseline — per-sensor Kalman filter
├── api/
│   └── main.py               # FastAPI serving layer
├── faultsense/
│   ├── __init__.py           # Public package API
│   └── metrics.py            # RUL + anomaly detection metrics
├── tests/
│   └── test_metrics.py       # 35 pytest tests (all passing)
├── data/
│   ├── raw/                  # NASA .txt files (not tracked)
│   └── processed/            # Preprocessed .pkl files (not tracked)
├── checkpoints/              # Trained model .pt files (not tracked)
├── preprocess.py             # NASA CMAPSS preprocessing pipeline
├── train.py                  # Two-phase training with MLflow logging
├── app.py                    # Streamlit dashboard
├── metrics.py                # Evaluation utilities
├── Dockerfile
├── docker-compose.yml        # API + MLflow services
├── pyproject.toml            # Installable package config
└── requirements.txt
```

---

## Running tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ -v --cov=metrics --cov-report=term-missing
```

Expected: **35 passed** across 7 test classes.

---

## Key design decisions

**Two-phase training** — the autoencoder is pre-trained on healthy-only windows to establish a tight reconstruction manifold before RUL fine-tuning begins. Most published implementations skip this step.

**Condition-aware normalisation** — for FD002/FD004 (6 operating conditions), sensor readings are normalised within each engine × condition group after K-means clustering on the op-setting columns. The fitted K-means is saved in the preprocessed pickle and reused at test time to prevent leakage.

**Asymmetric NASA loss during training** — not just at evaluation. The model learns a conservative bias: predicting an engine will fail sooner than it will is safer than the reverse.

**Per-sensor attribution** — the `per_sensor_mse` field in every API response identifies which sensors drove the anomaly score, enabling targeted maintenance decisions.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | LSTM autoencoder + RUL head |
| NumPy | ≥ 1.24 | Numerical ops |
| scikit-learn | ≥ 1.3 | K-means clustering, preprocessing |
| pandas | ≥ 2.0 | Data loading |
| FastAPI | ≥ 0.111 | REST API |
| MLflow | ≥ 2.13 | Experiment tracking |
| Streamlit | ≥ 1.35 | Dashboard |

---

## References

- Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* IEEE PHM Conference.
- NASA Prognostics Data Repository — [CMAPSS dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

---

## Author

**Mohamed Becha** — M.Sc. Mechanical Engineering, ETH Zürich  
[GitHub](https://github.com/momo-2609) · [LinkedIn](https://linkedin.com/in/mohamed-becha-164bb61b9)
