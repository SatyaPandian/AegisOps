import asyncio
import logging
import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("aegisops.inference")

app = FastAPI(title="AegisOps Inference Service", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"

REQUESTS = Counter(
    "aegisops_inference_requests_total",
    "Inference requests grouped by result.",
    ["result"],
)
LATENCY = Histogram(
    "aegisops_inference_latency_seconds",
    "Inference request latency in seconds.",
)
CONFIDENCE = Gauge(
    "aegisops_last_prediction_confidence",
    "Confidence from the most recent prediction.",
)
DRIFT_SCORE = Gauge(
    "aegisops_input_drift_score",
    "Simulated input distribution drift score from zero to one.",
)


class FaultMode(str, Enum):
    none = "none"
    latency = "latency"
    errors = "errors"
    low_confidence = "low_confidence"
    drift = "drift"


class PredictionRequest(BaseModel):
    image_id: str = Field(min_length=1, examples=["factory-camera-17/frame-000241.jpg"])
    brightness: float = Field(ge=0, le=1, examples=[0.63])
    sharpness: float = Field(ge=0, le=1, examples=[0.81])


class FaultUpdate(BaseModel):
    mode: FaultMode


class RemediationAction(str, Enum):
    route_to_last_known_healthy_model = "route_to_last_known_healthy_model"
    scale_inference_service = "scale_inference_service"
    schedule_data_quality_review = "schedule_data_quality_review"


class RemediationRequest(BaseModel):
    action: RemediationAction
    requested_by: str = Field(min_length=2, examples=["satya.pandian"])
    rationale: str = Field(min_length=10, examples=["Error rate exceeded the 20% incident threshold."])


class ApprovalRequest(BaseModel):
    approved_by: str = Field(min_length=2, examples=["on-call-engineer"])


active_fault = FaultMode.none
events: deque[dict] = deque(maxlen=500)
events_lock = Lock()
remediation_requests: dict[str, dict] = {}
remediation_lock = Lock()


def record_event(level: str, event: str, **details: object) -> None:
    """Store recent, structured evidence for incident investigations."""
    item = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **details,
    }
    with events_lock:
        events.append(item)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_fault": active_fault}


@app.post("/admin/fault")
def set_fault(update: FaultUpdate) -> dict:
    global active_fault
    active_fault = update.mode
    DRIFT_SCORE.set(0.72 if active_fault == FaultMode.drift else 0.08)
    logger.warning("fault_mode_changed mode=%s", active_fault.value)
    record_event("WARNING", "fault_mode_changed", mode=active_fault.value)
    return {"active_fault": active_fault}


@app.get("/admin/events")
def get_events(limit: int = 50) -> dict:
    """Return the most recent structured events for tools and debugging."""
    safe_limit = max(1, min(limit, 500))
    with events_lock:
        recent_events = list(events)[-safe_limit:]
    return {"count": len(recent_events), "events": recent_events}


@app.post("/admin/remediations", status_code=201)
def create_remediation(request: RemediationRequest) -> dict:
    """Create a pending request; no operational change is made here."""
    request_id = str(uuid.uuid4())
    item = {
        "id": request_id,
        "action": request.action.value,
        "requested_by": request.requested_by,
        "rationale": request.rationale,
        "status": "pending_human_approval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": None,
        "approved_at": None,
    }
    with remediation_lock:
        remediation_requests[request_id] = item
    record_event("WARNING", "remediation_requested", request_id=request_id, action=request.action.value)
    return item


@app.get("/admin/remediations")
def list_remediations() -> dict:
    with remediation_lock:
        items = list(remediation_requests.values())
    return {"count": len(items), "remediations": items}


@app.post("/admin/remediations/{request_id}/approve")
def approve_remediation(request_id: str, approval: ApprovalRequest) -> dict:
    """Record human approval without directly executing a production action."""
    with remediation_lock:
        item = remediation_requests.get(request_id)
        if not item:
            raise HTTPException(status_code=404, detail="Remediation request not found")
        if item["status"] != "pending_human_approval":
            raise HTTPException(status_code=409, detail=f"Request is already {item['status']}")
        item["status"] = "approved_for_execution"
        item["approved_by"] = approval.approved_by
        item["approved_at"] = datetime.now(timezone.utc).isoformat()
    record_event(
        "WARNING",
        "remediation_approved",
        request_id=request_id,
        action=item["action"],
        approved_by=approval.approved_by,
    )
    return item


@app.post("/predict")
async def predict(request: PredictionRequest) -> dict:
    started_at = time.perf_counter()

    if active_fault == FaultMode.latency:
        await asyncio.sleep(1.5)

    if active_fault == FaultMode.errors and random.random() < 0.55:
        REQUESTS.labels(result="error").inc()
        logger.error("prediction_failed image_id=%s reason=simulated_upstream_error", request.image_id)
        record_event(
            "ERROR",
            "prediction_failed",
            image_id=request.image_id,
            reason="simulated_upstream_error",
            fault=active_fault.value,
        )
        raise HTTPException(status_code=503, detail="Simulated upstream model failure")

    confidence = random.uniform(0.35, 0.56) if active_fault == FaultMode.low_confidence else random.uniform(0.82, 0.98)
    prediction = "defect" if request.sharpness < 0.45 else "no_defect"
    elapsed = time.perf_counter() - started_at

    REQUESTS.labels(result="success").inc()
    LATENCY.observe(elapsed)
    CONFIDENCE.set(confidence)
    logger.info(
        "prediction_complete image_id=%s prediction=%s confidence=%.3f latency_seconds=%.3f fault=%s",
        request.image_id,
        prediction,
        confidence,
        elapsed,
        active_fault.value,
    )
    record_event(
        "INFO",
        "prediction_complete",
        image_id=request.image_id,
        prediction=prediction,
        confidence=round(confidence, 3),
        latency_seconds=round(elapsed, 3),
        fault=active_fault.value,
    )
    return {
        "image_id": request.image_id,
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "latency_seconds": round(elapsed, 3),
        "active_fault": active_fault,
    }


app.mount("/metrics", make_asgi_app())
app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")
