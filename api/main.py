"""
api/main.py
-----------
FaultSense REST API — serves anomaly detection + RUL predictions.

Endpoints
  GET  /health          liveness check + model info
  GET  /models          list available checkpoints
  POST /predict         single window inference
  POST /predict/batch   batch inference (multiple windows)
  GET  /metrics         current model evaluation metrics

Run locally:
  uvicorn api.main:app --reload --port 8000

With Docker:
  docker compose up

Example request:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"subset": "FD001", "window": [[0.1, -0.2, ...], ...]}'
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FaultSense API",
    description="LSTM Autoencoder + EKF anomaly detection and RUL prediction on NASA CMAPSS turbofan data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model registry ────────────────────────────────────────────────────────────

_MODELS: dict = {}          # subset → FaultSenseModel
_CKPT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
_STARTUP_TIME = time.time()


def _load_model(subset: str):
    """Load and cache a FaultSense checkpoint."""
    if subset in _MODELS:
        return _MODELS[subset]

    ckpt_path = _CKPT_DIR / f"faultsense_{subset.lower()}.pt"
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint for subset '{subset}'. "
                   f"Run: python train.py --subset {subset}"
        )

    # Import here so the API still starts if torch isn't on PATH
    try:
        from train import load_checkpoint
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Model import failed: {e}")

    model, ckpt = load_checkpoint(str(ckpt_path), device="cpu")
    _MODELS[subset] = {"model": model, "ckpt": ckpt}
    return _MODELS[subset]


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    subset: str = Field("FD003", description="CMAPSS subset (FD001–FD004)")
    window: list[list[float]] = Field(
        ...,
        description="Sensor window — shape (seq_len, n_sensors). "
                    "Must match the subset's trained seq_len (default 30) "
                    "and n_sensors (default 14 after feature selection)."
    )

    @validator("subset")
    def subset_valid(cls, v):
        if v not in ("FD001", "FD002", "FD003", "FD004"):
            raise ValueError("subset must be one of FD001, FD002, FD003, FD004")
        return v

    @validator("window")
    def window_not_empty(cls, v):
        if not v or not v[0]:
            raise ValueError("window must be a non-empty 2D list")
        return v


class PredictResponse(BaseModel):
    subset: str
    anomaly_score: float = Field(..., description="MSE reconstruction error")
    rul_cycles: float    = Field(..., description="Predicted RUL in cycles")
    is_anomaly: bool     = Field(..., description="True if score > calibrated threshold")
    threshold: float     = Field(..., description="Calibrated anomaly threshold (μ + 3σ)")
    per_sensor_mse: list[float] = Field(..., description="Per-sensor reconstruction error")
    top_sensor_idx: int  = Field(..., description="Index of highest-error sensor")
    latency_ms: float    = Field(..., description="Inference time in milliseconds")


class BatchPredictRequest(BaseModel):
    subset: str = Field("FD003")
    windows: list[list[list[float]]] = Field(
        ..., description="Batch of windows — shape (B, seq_len, n_sensors)"
    )

    @validator("subset")
    def subset_valid(cls, v):
        if v not in ("FD001", "FD002", "FD003", "FD004"):
            raise ValueError("subset must be one of FD001, FD002, FD003, FD004")
        return v


class BatchPredictResponse(BaseModel):
    subset: str
    n_windows: int
    anomaly_scores: list[float]
    rul_cycles: list[float]
    is_anomaly: list[bool]
    top_sensor_indices: list[int]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    loaded_subsets: list[str]
    torch_available: bool
    device: str


class ModelInfoResponse(BaseModel):
    subset: str
    n_sensors: int
    seq_len: int
    hidden: int
    latent: int
    threshold: float
    test_rmse: Optional[float]
    test_nasa_score: Optional[float]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check. Returns loaded subsets and device info."""
    return HealthResponse(
        status="ok",
        uptime_s=round(time.time() - _STARTUP_TIME, 1),
        loaded_subsets=list(_MODELS.keys()),
        torch_available=torch.cuda.is_available() or True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.get("/models", response_model=list[ModelInfoResponse], tags=["Models"])
def list_models():
    """List all available checkpoints with their metrics."""
    results = []
    for ckpt_path in sorted(_CKPT_DIR.glob("faultsense_fd*.pt")):
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            cfg  = ckpt["config"]
            tr   = ckpt.get("test_results", {})
            results.append(ModelInfoResponse(
                subset=ckpt["subset"],
                n_sensors=len(ckpt["sensor_cols"]),
                seq_len=cfg["seq_len"],
                hidden=cfg["hidden"],
                latent=cfg["latent"],
                threshold=float(ckpt["threshold"]),
                test_rmse=round(float(tr["rmse"]), 2) if "rmse" in tr else None,
                test_nasa_score=round(float(tr["score"]), 1) if "score" in tr else None,
            ))
        except Exception:
            continue
    if not results:
        raise HTTPException(status_code=404, detail="No checkpoints found in checkpoint dir.")
    return results


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Single-window inference.

    Send a (seq_len × n_sensors) sensor window, receive:
    - anomaly_score   — MSE reconstruction error
    - rul_cycles      — predicted remaining useful life
    - is_anomaly      — whether score exceeds calibrated threshold
    - per_sensor_mse  — per-sensor breakdown for explainability
    """
    entry = _load_model(req.subset)
    model = entry["model"]
    cfg   = entry["ckpt"]["config"]

    X = np.array(req.window, dtype=np.float32)
    expected_seq = cfg["seq_len"]
    expected_sen = len(entry["ckpt"]["sensor_cols"])

    if X.shape != (expected_seq, expected_sen):
        raise HTTPException(
            status_code=422,
            detail=f"Window shape mismatch: got {X.shape}, "
                   f"expected ({expected_seq}, {expected_sen}) for {req.subset}."
        )

    t0     = time.perf_counter()
    result = model.predict(X, device="cpu")
    latency = (time.perf_counter() - t0) * 1000

    per_sensor = result["per_sensor_mse"].tolist()
    return PredictResponse(
        subset=req.subset,
        anomaly_score=round(float(result["anomaly_score"]), 6),
        rul_cycles=round(float(result["rul"]), 1),
        is_anomaly=bool(result["is_anomaly"]),
        threshold=round(float(model.threshold), 6),
        per_sensor_mse=[round(v, 6) for v in per_sensor],
        top_sensor_idx=int(np.argmax(per_sensor)),
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    """
    Batch inference — process multiple windows in one call.
    More efficient than calling /predict in a loop.
    Max batch size: 512 windows.
    """
    entry = _load_model(req.subset)
    model = entry["model"]
    cfg   = entry["ckpt"]["config"]

    if len(req.windows) > 512:
        raise HTTPException(status_code=422, detail="Max batch size is 512.")

    X = np.array(req.windows, dtype=np.float32)  # (B, T, S)
    expected_seq = cfg["seq_len"]
    expected_sen = len(entry["ckpt"]["sensor_cols"])

    if X.ndim != 3 or X.shape[1:] != (expected_seq, expected_sen):
        raise HTTPException(
            status_code=422,
            detail=f"Batch shape mismatch: got {X.shape}, "
                   f"expected (B, {expected_seq}, {expected_sen})."
        )

    t0     = time.perf_counter()
    result = model.predict(X, device="cpu")
    latency = (time.perf_counter() - t0) * 1000

    scores    = result["anomaly_score"].tolist()
    ruls      = result["rul"].tolist()
    per_sens  = result["per_sensor_mse"]
    threshold = float(model.threshold)

    return BatchPredictResponse(
        subset=req.subset,
        n_windows=len(req.windows),
        anomaly_scores=[round(s, 6) for s in scores],
        rul_cycles=[round(r, 1) for r in ruls],
        is_anomaly=[bool(s > threshold) for s in scores],
        top_sensor_indices=[int(np.argmax(row)) for row in per_sens],
        latency_ms=round(latency, 2),
    )


@app.get("/metrics/{subset}", response_model=ModelInfoResponse, tags=["Models"])
def get_metrics(subset: str):
    """Return stored test metrics for a trained subset."""
    entry = _load_model(subset)
    ckpt  = entry["ckpt"]
    cfg   = ckpt["config"]
    tr    = ckpt.get("test_results", {})
    return ModelInfoResponse(
        subset=subset,
        n_sensors=len(ckpt["sensor_cols"]),
        seq_len=cfg["seq_len"],
        hidden=cfg["hidden"],
        latent=cfg["latent"],
        threshold=round(float(ckpt["threshold"]), 6),
        test_rmse=round(float(tr["rmse"]), 2) if "rmse" in tr else None,
        test_nasa_score=round(float(tr["score"]), 1) if "score" in tr else None,
    )