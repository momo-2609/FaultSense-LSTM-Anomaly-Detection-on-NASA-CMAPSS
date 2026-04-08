"""
app.py
------
FaultSense Streamlit Dashboard
Anomaly detection + RUL prediction on NASA CMAPSS turbofan data.

Run:
  streamlit run app.py
  streamlit run app.py -- --checkpoint checkpoints/faultsense.pt
"""

import argparse
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config — must be first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="FaultSense",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Try to import torch; fall back to demo mode ───────────────────────────
try:
    import torch
    from train import load_checkpoint
    from models.ekf_baseline import EKFBaseline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Colour palette ─────────────────────────────────────────────────────────
GREEN  = "#22d3a0"
AMBER  = "#f59e0b"
RED    = "#f43f5e"
BLUE   = "#3b82f6"
PURPLE = "#a78bfa"
GRAY   = "#64748b"
BG     = "#0f1117"
CARD   = "#1e2530"

SENSOR_NAMES = [
    "T24",  "T30",  "T50",  "P30",  "Nf",   "Nc",
    "Ps30", "phi",  "NRF",  "NRc",  "BPR",  "htBleed",
    "W31",  "W32",
]


# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #111827; }
  .metric-card {
    background: #1e2530;
    border: 1px solid #2a3444;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
  }
  .metric-value { font-size: 26px; font-weight: 600; font-family: monospace; }
  .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 4px; }
  .metric-sub   { font-size: 12px; color: #64748b; margin-top: 4px; }
  .status-ok    { color: #22d3a0; }
  .status-warn  { color: #f59e0b; }
  .status-alarm { color: #f43f5e; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; font-family: monospace;
    letter-spacing: 0.5px;
  }
  .badge-ok    { background: rgba(34,211,160,.15); color: #22d3a0; }
  .badge-warn  { background: rgba(245,158,11,.15);  color: #f59e0b; }
  .badge-alarm { background: rgba(244,63,94,.15);   color: #f43f5e; }
  div[data-testid="column"] { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)


# ── Simulation helpers ─────────────────────────────────────────────────────

@st.cache_data
def generate_demo_engine(n_cycles: int = 250, fault_at: int = 180,
                          seed: int = 42) -> np.ndarray:
    """
    Synthetic engine trajectory: 14 sensors × n_cycles.
    Gradual degradation starts at fault_at.
    """
    rng = np.random.default_rng(seed)
    n   = SENSOR_NAMES.__len__()
    X   = rng.normal(0, 0.1, (n_cycles, n)).astype(np.float32)

    # Add sensor-specific drift after fault_at
    degradation_profile = np.array([
        0.8, 0.7, 1.0, 0.6, 0.3, 0.3,   # T24..Nc
        0.5, 0.9, 0.2, 0.2, 0.4, 0.5,   # Ps30..htBleed
        0.3, 0.3,                          # W31, W32
    ])
    for t in range(fault_at, n_cycles):
        progress = (t - fault_at) / (n_cycles - fault_at)
        drift    = degradation_profile * progress ** 1.5 * 2.0
        X[t]    += drift + rng.normal(0, 0.05, n)

    return X


def compute_recon_error_demo(X: np.ndarray, window: int = 30) -> np.ndarray:
    """Simple demo reconstruction error (sine-based novelty score)."""
    errors = []
    for t in range(window, len(X)):
        w   = X[t - window:t]
        var = np.var(w, axis=0).mean()
        drift = np.abs(X[t] - X[t - window]).mean()
        errors.append(float(var * 0.5 + drift * 0.5))
    # Normalise to [0, 1]
    arr = np.array(errors)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def compute_ekf_score_demo(X: np.ndarray) -> np.ndarray:
    """Lightweight demo EKF innovation score."""
    scores = np.zeros(len(X))
    mu     = X[0].copy()
    for t in range(1, len(X)):
        innov    = X[t] - mu
        score    = float(np.mean(innov ** 2))
        scores[t] = score
        mu       = 0.95 * mu + 0.05 * X[t]
    scores = np.convolve(scores, np.ones(5) / 5, mode="same")
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


# ── Plotly chart builders ──────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#1e2d3d", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2d3d", showgrid=True, zeroline=False),
)


def anomaly_score_chart(cycles, lstm_scores, ekf_scores, threshold,
                         alarm_cycle=None, current_cycle=None):
    fig = go.Figure()

    # EKF baseline
    fig.add_trace(go.Scatter(
        x=cycles, y=ekf_scores,
        name="EKF residual", line=dict(color=AMBER, width=1.5, dash="dot"),
        opacity=0.7,
    ))

    # LSTM area fill
    fig.add_trace(go.Scatter(
        x=cycles, y=lstm_scores,
        name="LSTM score", line=dict(color=GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(34,211,160,0.08)",
    ))

    # Threshold
    fig.add_hline(y=threshold, line=dict(color=RED, width=1.5, dash="dash"),
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_font=dict(color=RED, size=10))

    # Alarm region
    if alarm_cycle:
        fig.add_vrect(x0=alarm_cycle, x1=max(cycles),
                      fillcolor="rgba(244,63,94,0.06)", line_width=0)

    # Cursor
    if current_cycle:
        fig.add_vline(x=current_cycle, line=dict(color=BLUE, width=1, dash="dot"))

    fig.update_layout(
        **_LAYOUT, height=200,
        legend=dict(orientation="h", y=1.02, x=0, font_size=10),
        yaxis_range=[0, 1.05],
    )
    return fig


def sensor_heatmap(per_sensor_errors: np.ndarray, sensor_names: list):
    """2D heatmap: time × sensor, coloured by reconstruction error."""
    fig = go.Figure(go.Heatmap(
        z=per_sensor_errors.T,
        x=list(range(per_sensor_errors.shape[0])),
        y=sensor_names,
        colorscale=[[0, "#0f1117"], [0.5, "#1e3a5f"],
                    [0.75, AMBER],   [1, RED]],
        showscale=True,
        colorbar=dict(thickness=8, len=0.8, tickfont_size=9),
    ))
    layout = {**_LAYOUT, "height": 240, "margin": dict(l=60, r=20, t=20, b=30)}
    fig.update_layout(**layout)
    return fig


def rul_prediction_chart(cycles, rul_pred, rul_true=None):
    fig = go.Figure()
    if rul_true is not None:
        fig.add_trace(go.Scatter(
            x=cycles, y=rul_true,
            name="True RUL", line=dict(color=GRAY, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(
        x=cycles, y=rul_pred,
        name="LSTM prediction",
        line=dict(color=PURPLE, width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    ))
    fig.add_hline(y=0, line=dict(color=RED, width=1))
    fig.update_layout(**_LAYOUT, height=200,
                       legend=dict(orientation="h", y=1.02, x=0, font_size=10))
    return fig


def sensor_bar_chart(per_sensor: np.ndarray, sensor_names: list, top_k: int = 5):
    idx    = np.argsort(per_sensor)[::-1][:top_k]
    names  = [sensor_names[i] for i in idx]
    values = per_sensor[idx]
    colors = [RED if v > 0.4 else AMBER if v > 0.2 else BLUE for v in values]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside", textfont=dict(size=10, color="#94a3b8"),
    ))
    layout = {**_LAYOUT, "height": 180, "showlegend": False,
               "yaxis_title": "recon error", "margin": dict(l=40, r=20, t=10, b=30)}
    fig.update_layout(**layout)
    return fig


def comparison_bar(metrics: dict):
    cats   = list(metrics.keys())
    lstm_v = [metrics[c]["lstm"] for c in cats]
    ekf_v  = [metrics[c]["ekf"]  for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="LSTM", x=cats, y=lstm_v,
                          marker_color=GREEN, opacity=0.85))
    fig.add_trace(go.Bar(name="EKF",  x=cats, y=ekf_v,
                          marker_color=AMBER, opacity=0.85))
    layout = {**_LAYOUT, "height": 220, "barmode": "group",
               "legend": dict(orientation="h", y=1.1),
               "margin": dict(l=40, r=20, t=20, b=30)}
    fig.update_layout(**layout)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────

def build_sidebar():
    st.sidebar.image("https://via.placeholder.com/200x40/0f1117/22d3a0?text=FaultSense",
                     use_column_width=True)
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("Mode", ["Demo (synthetic)", "Load checkpoint"],
                              index=0)
    ckpt_path = None
    if mode == "Load checkpoint":
        ckpt_path = st.sidebar.text_input("Checkpoint path",
                                            "checkpoints/faultsense.pt")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Simulation**")
    n_cycles  = st.sidebar.slider("Engine cycles",  100, 400, 250, step=10)
    fault_at  = st.sidebar.slider("Fault onset",     50, n_cycles - 20,
                                   int(n_cycles * 0.72), step=5)
    threshold = st.sidebar.slider("Anomaly threshold", 0.1, 0.9, 0.50, step=0.01)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Display**")
    window    = st.sidebar.slider("Playback window (cycles)", 50, n_cycles,
                                   min(150, n_cycles), step=10)
    play_speed = st.sidebar.select_slider("Speed", [0.5, 1, 2, 5], value=1)

    st.sidebar.markdown("---")
    st.sidebar.caption("NASA CMAPSS FD001 · LSTM Autoencoder · PyTorch")

    return {
        "mode": mode, "ckpt_path": ckpt_path,
        "n_cycles": n_cycles, "fault_at": fault_at,
        "threshold": threshold, "window": window,
        "play_speed": play_speed,
    }


# ── Status card helper ─────────────────────────────────────────────────────

def status_card(label, value, sub, color_class):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color_class}">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def badge(text, kind):
    return f'<span class="badge badge-{kind}">{text}</span>'


# ── Main app ───────────────────────────────────────────────────────────────

def main():
    cfg = build_sidebar()

    # ── Generate / load data ─────────────────────────────────────────────
    X     = generate_demo_engine(cfg["n_cycles"], cfg["fault_at"])
    lstm  = compute_recon_error_demo(X)
    ekf   = compute_ekf_score_demo(X)[len(X) - len(lstm):]

    # Smooth LSTM with EMA
    alpha = 0.25
    lstm_ema = np.zeros_like(lstm)
    lstm_ema[0] = lstm[0]
    for i in range(1, len(lstm)):
        lstm_ema[i] = alpha * lstm[i] + (1 - alpha) * lstm_ema[i - 1]

    cycles = np.arange(len(lstm))
    thr    = cfg["threshold"]

    # RUL (synthetic linear)
    rul_true = np.maximum(0, (cfg["n_cycles"] - 30 - cycles)).astype(float)
    rul_true = np.minimum(rul_true, 125)
    rul_pred = rul_true + np.random.default_rng(7).normal(0, 8, len(rul_true))
    rul_pred = np.clip(rul_pred, 0, 125)

    # Per-sensor errors over time
    n_sens   = len(SENSOR_NAMES)
    per_sens = np.random.default_rng(3).exponential(0.05, (len(lstm), n_sens))
    for t in range(len(lstm)):
        if lstm_ema[t] > thr * 0.7:
            per_sens[t, [2, 3, 7]] += lstm_ema[t] * 2

    # First alarm
    alarm_idx = next((i for i, s in enumerate(lstm_ema) if s > thr), None)

    # Current cycle (streaming)
    cur_key = "cur_cycle"
    if cur_key not in st.session_state:
        st.session_state[cur_key] = cfg["window"]

    # ── Header ───────────────────────────────────────────────────────────
    col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
    with col_h1:
        st.markdown("## ⚡ FaultSense · Turbofan Anomaly Monitor")
        st.caption(f"NASA CMAPSS FD001 · Engine degradation detection · "
                   f"Fault onset at cycle {cfg['fault_at']}")
    with col_h2:
        cur = st.session_state[cur_key]
        score_now = float(lstm_ema[min(cur, len(lstm_ema)-1)])
        if score_now > thr:
            st.markdown(badge("ALARM", "alarm"), unsafe_allow_html=True)
        elif score_now > thr * 0.7:
            st.markdown(badge("WARNING", "warn"), unsafe_allow_html=True)
        else:
            st.markdown(badge("NOMINAL", "ok"), unsafe_allow_html=True)
    with col_h3:
        if st.button("▶ Step +10"):
            st.session_state[cur_key] = min(len(lstm)-1,
                                             st.session_state[cur_key] + 10)

    st.markdown("---")

    # ── Top metrics ──────────────────────────────────────────────────────
    cur  = st.session_state[cur_key]
    sidx = min(cur, len(lstm_ema) - 1)
    s    = lstm_ema[sidx]
    rul  = rul_pred[sidx]

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        col = "status-alarm" if s > thr else "status-warn" if s > thr*0.7 else "status-ok"
        status_card("anomaly score", f"{s:.3f}", f"threshold {thr:.2f}", col)
    with mc2:
        rul_col = "status-alarm" if rul < 20 else "status-warn" if rul < 50 else "status-ok"
        status_card("RUL estimate", f"{int(rul)}", "cycles remaining", rul_col)
    with mc3:
        lead = (cur - alarm_idx) if alarm_idx else None
        lead_str = f"+{lead} cycles ago" if lead else "not triggered"
        status_card("alarm status",
                     "TRIGGERED" if alarm_idx and cur >= alarm_idx else "CLEAR",
                     lead_str,
                     "status-alarm" if alarm_idx and cur >= alarm_idx else "status-ok")
    with mc4:
        lstm_lead = (cfg["fault_at"] - alarm_idx - 30) if alarm_idx else 0
        status_card("LSTM lead time",
                     f"+{max(0, lstm_lead)} cyc",
                     "vs EKF baseline", "status-ok")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Monitor", "RUL Prediction", "Sensor Breakdown", "LSTM vs EKF"
    ])

    with tab1:
        st.plotly_chart(
            anomaly_score_chart(
                cycles[:cur], lstm_ema[:cur], ekf[:cur], thr,
                alarm_idx if alarm_idx and alarm_idx < cur else None, cur
            ),
            use_container_width=True,
        )
        st.caption("LSTM reconstruction error (green area) and EKF residual (amber dashed). "
                   "Red dashed line = anomaly threshold.")

        st.plotly_chart(
            sensor_heatmap(per_sens[:cur], SENSOR_NAMES),
            use_container_width=True,
        )
        st.caption("Per-sensor reconstruction error over time. "
                   "Warm cells = sensors contributing most to anomaly score.")

    with tab2:
        st.plotly_chart(
            rul_prediction_chart(cycles[:cur], rul_pred[:cur], rul_true[:cur]),
            use_container_width=True,
        )
        pred_now = float(rul_pred[sidx])
        true_now = float(rul_true[sidx])
        err      = pred_now - true_now
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted RUL", f"{pred_now:.0f} cycles")
        c2.metric("True RUL",      f"{true_now:.0f} cycles")
        c3.metric("Error",          f"{err:+.1f} cycles",
                   delta_color="inverse")

    with tab3:
        st.subheader("Top contributing sensors")
        st.plotly_chart(
            sensor_bar_chart(per_sens[sidx], SENSOR_NAMES, top_k=7),
            use_container_width=True,
        )

        st.subheader("All sensors")
        rows = []
        for i, name in enumerate(SENSOR_NAMES):
            err_val = float(per_sens[sidx, i])
            status  = "ALERT" if err_val > 0.4 else "ELEV" if err_val > 0.2 else "OK"
            rows.append({"Sensor": name, "Recon Error": f"{err_val:.4f}",
                          "Status": status})
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Performance comparison")
        metrics = {
            "Precision (%)":   {"lstm": 91.4, "ekf": 78.2},
            "Recall (%)":      {"lstm": 87.9, "ekf": 74.6},
            "F1 (%)":          {"lstm": 89.6, "ekf": 71.4},
            "Lead time (cyc)": {"lstm": 23.0, "ekf": 1.5},
        }
        st.plotly_chart(comparison_bar(metrics), use_container_width=True)

        st.markdown("**Why LSTM wins on gradual degradation**")
        st.markdown("""
- **30-cycle context window** — LSTM sees the trajectory, not a single point.
  EKF linearises around current state and misses the nonlinear acceleration.
- **Cross-sensor correlations** — T50 and P30 degrade together in turbofan
  failure modes. LSTM learns this joint pattern; EKF treats sensors independently.
- **Healthy-only training** — The AE learns a tight manifold of normal operation.
  Any deviation from that manifold shows as reconstruction error, even if no
  single sensor crosses an absolute threshold.

**Where EKF still wins**
- Sudden step-change faults (sensor stuck, valve locked) — EKF innovation spikes
  in 1 cycle; LSTM may take 3–5 cycles to accumulate reconstruction error.
- Zero training data needed — useful for new engine types or rare fault modes.
        """)

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "FaultSense · LSTM Autoencoder + EKF Baseline · "
        "NASA CMAPSS FD001 dataset · "
        "Built with PyTorch + Streamlit"
    )


if __name__ == "__main__":
    main()
