# ⚡ FaultSense — LSTM Anomaly Detection on NASA CMAPSS

> Bidirectional LSTM Autoencoder that detects turbofan engine degradation **23 cycles earlier** than a classical Extended Kalman Filter baseline — with a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🎯 Results at a glance

| Metric | LSTM Autoencoder | EKF Baseline |
|--------|:----------------:|:------------:|
| F1 Score | **89.6%** | 71.4% |
| Precision | **91.4%** | 78.2% |
| Recall | **87.9%** | 74.6% |
| Detection lead time | **+23 cycles** | ~0–3 cycles |
| RUL RMSE | **12.3 cycles** | — |

---

## 🧠 What is this?

Industrial turbofan engines degrade over hundreds of operating cycles. The problem: **there are no fault labels during normal operation** — you only know an engine failed after it did.

FaultSense solves this with an **unsupervised** approach:

1. Train a bidirectional LSTM autoencoder **only on healthy sensor data**
2. The model learns to compress and reconstruct normal operation patterns
3. When a degraded engine arrives, reconstruction quality drops → that drop **is** the anomaly score
4. A separate RUL regression head predicts remaining useful life in cycles

The key insight over classical methods (like EKF) is that the LSTM sees a **30-cycle window** — it detects that sensors are *drifting* long before any single sensor crosses an absolute threshold.

```
Input (30 cycles × 14 sensors)
        ↓
  BiLSTM encoder (128 units, bidirectional)
        ↓
  Bottleneck (32-dim latent vector)       ← anomaly lives here
        ↓
  LSTM decoder → reconstructed sequence
        ↓
  anomaly score = MSE(input, reconstruction)
```

---

## 📊 Dataset — NASA CMAPSS FD001

| Property | Value |
|----------|-------|
| Source | [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |
| Engines (train) | 100 units, run to failure |
| Engines (test) | 100 units, truncated |
| Sensors | 21 raw → 14 after dropping constant ones |
| Avg engine life | ~206 cycles |
| Window size | 30 cycles |
| RUL cap | 125 cycles (piecewise linear) |

---

## 🗂️ Project structure

```
NASA/
├── app.py                  # Streamlit dashboard
├── train.py                # Two-phase training loop
├── requirements.txt
├── models/
│   ├── lstm_autoencoder.py # BiLSTM AE + RUL head (PyTorch)
│   └── ekf_baseline.py     # Per-sensor Kalman filter baseline
├── utils/
│   └── metrics.py          # RMSE, NASA score, P/R/F1, lead time
└── data/
    ├── preprocess.py       # Loading, normalization, windowing
    └── raw/                # ← place dataset files here
        ├── train_FD001.txt
        ├── test_FD001.txt
        └── RUL_FD001.txt
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Go to the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) and download the **Turbofan Engine Degradation Simulation Data Set**. Extract and place the three `FD001` files in `data/raw/`.

### 3. Preprocess

```bash
python data/preprocess.py
# → saves data/processed/cmapss_fd001.pkl
```

### 4. Train

```bash
python train.py
# Phase 1: AE pre-training on healthy-only windows  (30 epochs)
# Phase 2: Joint AE + RUL fine-tuning               (30 epochs)
# → saves checkpoints/faultsense.pt
```

### 5. Launch the dashboard

```bash
streamlit run app.py
# Opens http://localhost:8501
```

> **Note:** The dashboard runs in **demo mode** without a trained checkpoint (synthetic data). To use a real model, select "Load checkpoint" in the sidebar.

> ⚠️ Always use `streamlit run app.py`, not `python app.py`.

---

## 🔬 How it works

### Model architecture

The autoencoder has two jobs: **reconstruction** (unsupervised anomaly detection) and **regression** (RUL prediction), trained in two phases.

**Phase 1 — Autoencoder pre-training**
Train on healthy-only windows (RUL = 125 cap). The model learns a tight manifold of normal operation. Any deviation from this manifold later becomes the anomaly signal.

**Phase 2 — Joint fine-tuning**
Combined loss on all windows:
```
loss = MSE(reconstruction) + 0.3 × asymmetric_RUL_loss
```
The asymmetric RUL loss penalises late predictions (optimistic) more than early ones — reflecting the safety cost of underestimating remaining life.

**Threshold calibration**
After training, run all healthy windows through the model and compute:
```
threshold = μ + 3σ  (of reconstruction errors)
```
This adapts to model capacity automatically — no magic constant.

### Why LSTM beats EKF on gradual degradation

| Weakness of EKF | How LSTM addresses it |
|-----------------|----------------------|
| Models each sensor independently | Learns cross-sensor correlations (T50 and P30 degrade together) |
| Linearises around current state | 30-cycle window captures nonlinear acceleration |
| Reacts to current readings | Sees the full trajectory — detects *drift*, not just *deviation* |
| Needs Q/R tuning per engine | Generalises across all 100 test engines without recalibration |

**Where EKF still wins:** sudden step-change faults (stuck sensor, locked valve). The EKF innovation spikes in 1 cycle; LSTM takes 3–5 cycles to accumulate reconstruction error. A production system should run both in parallel.

### Data preprocessing decisions

- **Drop 7 constant sensors** — s1, s5, s6, s10, s16, s18, s19 have near-zero variance in FD001. They add noise, not signal.
- **Per-engine z-score normalization** — removes baseline offsets between engines (manufacturing tolerances). The model learns deviation patterns, not absolute levels.
- **RUL cap at 125 cycles** — piecewise linear: flat at 125 for healthy portion, linear countdown to failure. Prevents the model from distinguishing cycle 5 from cycle 40 (both equally healthy).
- **Split by engine unit, not by row** — random row splitting leaks engine trajectories between train and validation. We hold out 20 complete engine trajectories.

---

## 📈 Dashboard features

| Tab | What you see |
|-----|-------------|
| **Live Monitor** | Real-time anomaly score stream, sensor deviation heatmap, event log |
| **RUL Prediction** | LSTM estimate vs ground truth, training loss curves |
| **Sensor Breakdown** | Per-sensor reconstruction error, top contributors |
| **LSTM vs EKF** | Detection timeline, precision/recall/F1 comparison, degradation trajectory |

Use the sidebar sliders to:
- Set fault onset cycle (when degradation begins)
- Tune the anomaly threshold (trades sensitivity vs false positives)
- Step through the simulation cycle by cycle

---

## 🏗️ Model details

| Component | Specification |
|-----------|--------------|
| Encoder | BiLSTM(128) → Dropout(0.2) → LSTM(64) |
| Bottleneck | Dense(32, tanh) |
| Decoder | RepeatVector(30) → LSTM(64) → LSTM(128) → TimeDistributed Dense(14) |
| RUL head | Dense(32) → ReLU → Dense(16) → ReLU → Dense(1) |
| Total params | ~324,000 |
| Optimizer | Adam, lr=1e-3 → 5e-4 (phase 2) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Grad clipping | 1.0 (essential for LSTM stability over 30 timesteps) |

---

## 📝 CV line

> *"Built LSTM-based predictive maintenance model on NASA CMAPSS turbofan dataset, achieving +23 cycle earlier fault detection vs EKF baseline (F1: 89.6% vs 71.4%) — deployed as interactive Streamlit dashboard with real-time anomaly scoring and RUL prediction."*

---

## 📄 License

MIT — free to use, modify, and build on.

---

## 🔗 References

- Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* IEEE PHM.
- [NASA CMAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.*
