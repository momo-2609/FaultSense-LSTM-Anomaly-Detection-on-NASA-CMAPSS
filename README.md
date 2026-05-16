# ⚡ FaultSense

[![CI](https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS/actions/workflows/ci.yml/badge.svg)](https://github.com/momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS/actions/workflows/ci.yml)

**LSTM Autoencoder for anomaly detection and Remaining Useful Life (RUL) prediction on NASA CMAPSS turbofan engine data.**

FaultSense trains an end-to-end deep learning pipeline that reconstructs sensor windows to detect degradation, predicts RUL in cycles, and surfaces results through an interactive fleet health dashboard and a production-ready REST API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-orange)

---

## Results

| Subset | Model | RMSE (↓) | NASA Score (↓) |
|--------|-------|----------|----------------|
| FD001 | **LSTM Autoencoder** | **14.85** | **311.6** |
| FD001 | Ridge (engineered) | 15.47 | 419.3 |
| FD001 | Ridge (raw) | 15.75 | 444.3 |
| FD003 | **LSTM Autoencoder** | **13.88** | **425.3** |
| FD003 | Ridge (engineered) | 16.59 | 482.3 |
| FD003 | Ridge (raw) | 17.16 | 573.5 |

The LSTM beats both Ridge baselines on every metric across both subsets. NASA Score uses the official asymmetric PHM scoring function — late predictions (predicting more life than remains) are penalised harder than early ones, reflecting real-world safety asymmetry.

<img width="950" height="476" alt="Capture d’écran 2026-05-16 155037" src="https://github.com/user-attachments/assets/2a38bd74-e179-4def-b5b7-e502eea7b0a9" />
<img width="752" height="422" alt="Capture d’écran 2026-05-16 155121" src="https://github.com/user-attachments/assets/e8593555-cc17-4ae3-94d9-b780c273a2ff" />

---

## Architecture

```
Raw sensor data (21 sensors)
        │
        ▼
  Preprocessing
  - Feature selection (14–15 sensors)
  - Min-max normalisation per operating condition
  - Sliding windows (seq_len=30)
        │
        ▼
  LSTM Autoencoder
  ┌─────────────────────────────────────┐
  │  Encoder  →  Latent space (dim=32)  │
  │  Decoder  →  Reconstruction         │
  │  RUL Head →  Predicted RUL (cycles) │
  └─────────────────────────────────────┘
        │
        ├── Anomaly score  = MSE reconstruction error
        ├── Threshold      = μ + 2.5σ  (calibrated on validation)
        └── RUL prediction = latent → linear head → cycles
```

---

## Project Structure

```
NASA/
├── app.py                  # Plotly Dash fleet health dashboard
├── train.py                # LSTM training pipeline + MLflow logging
├── train_baseline.py       # Ridge regression baselines
├── preprocess.py           # CMAPSS preprocessing (all 4 subsets)
├── models/
│   └── lstm_autoencoder.py # FaultSenseModel definition
├── api/
│   └── main.py             # FastAPI inference API
├── metrics.py              # RUL + anomaly detection metrics
├── tests/
│   └── test_metrics.py     # pytest suite
├── checkpoints/            # Trained model weights (.pt, .pkl)
├── data/
│   ├── raw/                # NASA CMAPSS text files
│   └── processed/          # Preprocessed pickle files
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/momo-2609/faultsense.git
cd faultsense
pip install -e ".[all]"
```

### 2. Preprocess

```bash
python preprocess.py --subset FD001
python preprocess.py --subset FD003
# or all at once:
python preprocess.py --all
```

### 3. Train

```bash
python train.py --subset FD001
python train.py --subset FD003
```

Checkpoints are saved to `checkpoints/faultsense_fd001.pt` etc. Training is logged to MLflow automatically.

### 4. Train baselines

```bash
python train_baseline.py --subset FD001 --both-modes
python train_baseline.py --subset FD003 --both-modes
```

### 5. Run the dashboard

```bash
python app.py
# → http://127.0.0.1:8888
```

### 6. Run the API

```bash
uvicorn api.main:app --reload --port 8000
# → http://127.0.0.1:8000/docs
```

---

## API

The FastAPI service exposes inference endpoints for real-time use.

### Single window prediction

```bash
POST /predict
{
  "subset": "FD001",
  "window": [[...], ...]   # shape (30, 14)
}
```

Response:
```json
{
  "rul_cycles": 47.3,
  "anomaly_score": 0.021,
  "is_anomaly": false,
  "threshold": 0.038,
  "per_sensor_mse": [...],
  "top_sensor_idx": 6,
  "latency_ms": 3.1
}
```

Other endpoints: `GET /health`, `GET /models`, `POST /predict/batch`, `GET /metrics/{subset}`

Full interactive docs at `http://localhost:8000/docs`.

---

## Docker

Runs the API + MLflow tracking server together:

```bash
docker compose up
# API    → http://localhost:8000
# MLflow → http://localhost:5000
```

Checkpoints and data are mounted as volumes — no rebuild needed when you retrain.

---

## Dataset

[NASA CMAPSS Turbofan Engine Degradation Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Four subsets with varying operating conditions and fault modes:

| Subset | Op. Conditions | Fault Modes | Train Engines |
|--------|---------------|-------------|---------------|
| FD001  | 1             | 1 (HPC)     | 100           |
| FD002  | 6             | 1 (HPC)     | 260           |
| FD003  | 1             | 2 (HPC+Fan) | 100           |
| FD004  | 6             | 2 (HPC+Fan) | 249           |

Place the raw `.txt` files in `data/raw/` before preprocessing.

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=metrics
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
