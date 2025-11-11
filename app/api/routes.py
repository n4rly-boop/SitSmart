from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import FeatureExtractionResponse, Notification, NotificationSeverity, ModelAnalysisRequest, ModelAnalysisResponse, FeatureVector, NotificationConfig, NotificationConfigUpdate
from app.services.history_service import HistoryService
from app.services.pose_service import PoseService, PoseServiceUnavailable
from app.services.notification_service import NotificationService
from app.services.rl_service import ThresholdTSAgent
from app.services.feature_aggregate_service import FeatureAggregateService
from app.services.calibration_service import CalibrationService
from app.services.ml_service import MLService

router = APIRouter()

_notification_ws_clients = set()
_last_notification: Notification | None = None
_agg = FeatureAggregateService.get_instance()


@router.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}

@router.post("/features", response_model=FeatureExtractionResponse, tags=["analysis"])
async def extract_features(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        # Use PoseService for both pose and features
        result = PoseService.get_instance().extract_features(image_bytes)
        return FeatureExtractionResponse.model_validate(result)
    except PoseServiceUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface error to client for now
        raise HTTPException(status_code=400, detail=f"Failed to extract features: {e}")




@router.get("/reminder", response_model=Notification, tags=["notifications"]) 
async def posture_reminder():
    # Static reminder payload for now; can be made dynamic based on recent analysis
    return Notification(
        title="Posture Check",
        message="Please straighten your back and align your head with your shoulders.",
        severity=NotificationSeverity.warning,
        suggested_action="Sit upright, relax shoulders, and keep screen at eye level.",
        ttl_seconds=10,
    )


# Webhook receiver for notifications (allows pluggable delivery channels)
@router.post("/notifications/webhook", tags=["notifications"])
async def notifications_webhook(payload: Notification):
    # In a production system, this would fan-out to subscribers (e.g., WS broadcast, push, email)
    # Broadcast to WS subscribers
    try:
        print(f"[routes:webhook] received notification to broadcast; clients={len(_notification_ws_clients)}")
    except Exception:
        pass
    global _last_notification
    _last_notification = payload
    dead = []
    for ws in list(_notification_ws_clients):
        try:
            await ws.send_json(payload.model_dump())
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            _notification_ws_clients.discard(ws)
            await ws.close()
        except Exception:
            pass
    return JSONResponse(status_code=200, content={"ok": True})


@router.websocket("/ws/notifications")
async def ws_notifications(websocket: WebSocket):
    await websocket.accept()
    _notification_ws_clients.add(websocket)
    try:
        print(f"[routes:ws_notifications] client connected; total={len(_notification_ws_clients)}")
    except Exception:
        pass
    # Send ready signal and last notification if any
    try:
        await websocket.send_json({"status": "ready"})
        if _last_notification is not None:
            await websocket.send_json(_last_notification.model_dump())
    except Exception:
        pass
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            # notifications are server-push only; ignore client messages
    except WebSocketDisconnect:
        try:
            print("[routes:ws_notifications] client disconnected (WebSocketDisconnect)")
        except Exception:
            pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        _notification_ws_clients.discard(websocket)
        try:
            print(f"[routes:ws_notifications] client removed; total={len(_notification_ws_clients)}")
        except Exception:
            pass


@router.get("/notifications/last", response_model=Notification | None, tags=["notifications"])
async def notifications_last():
    return _last_notification


# RL endpoints intentionally removed from main flow


# Future ML analyze stub with the same contract
@router.post("/ml/analyze", response_model=ModelAnalysisResponse, tags=["ml"])
async def ml_analyze(payload: ModelAnalysisRequest):
    resp = MLService.get_instance().analyze(payload)
    return resp


# Decide using the server buffer (mean over window) via ML, then notify if above threshold
@router.post("/decide/from_buffer", response_model=ModelAnalysisResponse, tags=["decision"])
async def decide_from_buffer():
    try:
        mean_features = _agg.mean_features()
    except Exception:
        mean_features = None
    if not mean_features:
        return ModelAnalysisResponse(bad_posture_prob=0, reason="no-features")
    # Call ML once, use score for notification decision
    resp = MLService.get_instance().analyze(ModelAnalysisRequest(features=FeatureVector(**mean_features)))
    try:
        # Pass f1 (mean_features) to history via notification service
        NotificationService.get_instance().maybe_notify_from_ml_response(resp, f1_features=mean_features)
    except Exception:
        pass
    return resp


# Notification configuration endpoints
@router.get("/notifications/config", response_model=NotificationConfig, tags=["notifications"])
async def notifications_config_get():
    svc = NotificationService.get_instance()
    opts = svc.options
    # Populate only the supported fields; others retain defaults
    return NotificationConfig(
        cooldown_seconds=int(opts.cooldown_seconds),
        ml_bad_prob_threshold=float(getattr(opts, "ml_bad_prob_threshold", 0.6)),
    )


@router.post("/notifications/config", response_model=NotificationConfig, tags=["notifications"])
async def notifications_config_update(payload: NotificationConfigUpdate):
    svc = NotificationService.get_instance()
    if payload.cooldown_seconds is not None:
        svc.options.cooldown_seconds = max(0, int(payload.cooldown_seconds))
    if payload.ml_bad_prob_threshold is not None:
        # Clamp to [0,1]
        val = float(payload.ml_bad_prob_threshold)
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        svc.options.ml_bad_prob_threshold = val
    return await notifications_config_get()

# Feature aggregation management endpoints
@router.post("/features/aggregate/add", tags=["aggregation"])
async def features_aggregate_add(payload: FeatureVector):
    try:
        feat = payload.model_dump()
        _agg.add_features(feat)
        # If calibrating, update calibration ranges
        try:
            CalibrationService.get_instance().update_from_features(feat)
        except Exception:
            pass
    except Exception:
        pass
    # History service will fetch live features via HTTP when needed
    return {"ok": True}


@router.get("/features", tags=["analysis"])
async def get_latest_features():
    # Provide the latest mean-less, single-sample features that were last added to the buffer
    # We approximate latest by returning the current buffer mean to keep a simple server-side contract
    # but the HistoryService will call this endpoint after delta_range seconds to sample f2.
    try:
        features_only = _agg.last_features()
    except Exception:
        features_only = None
    return {"features": features_only}


@router.get("/features/ranges", tags=["analysis"])
async def get_feature_ranges():
    try:
        snapshot = CalibrationService.get_instance().snapshot()
    except Exception:
        snapshot = {}
    return {"ranges": snapshot}


# Calibration control
@router.get("/calibration/status", tags=["analysis"])
async def calibration_status():
    try:
        on = CalibrationService.get_instance().is_calibrating()
    except Exception:
        on = False
    return {"calibrating": bool(on)}


@router.post("/calibration/start", tags=["analysis"])
async def calibration_start():
    try:
        CalibrationService.get_instance().start()
    except Exception:
        pass
    return {"ok": True, "calibrating": True}


@router.post("/calibration/stop", tags=["analysis"])
async def calibration_stop():
    try:
        CalibrationService.get_instance().stop()
    except Exception:
        pass
    return {"ok": True, "calibrating": False}


@router.get("/rl/threshold", tags=["rl"])
async def rl_threshold():
    try:
        thr = ThresholdTSAgent.get_instance().get_last_decision_threshold()
    except Exception:
        try:
            thr = float(NotificationService.get_instance().options.ml_bad_prob_threshold)
        except Exception:
            thr = 0.6
    return {"threshold": float(thr)}


@router.get("/rl/delta_baseline", tags=["rl"])
async def rl_delta_baseline():
    try:
        delta_baseline = ThresholdTSAgent.get_instance().get_delta_baseline()
        if delta_baseline is None:
            return {"delta_baseline": None}
        return {"delta_baseline": float(delta_baseline)}
    except Exception:
        return {"delta_baseline": None}


@router.get("/rl/band_bounds", tags=["rl"])
async def rl_band_bounds():
    try:
        L, H = ThresholdTSAgent.get_instance().get_band_bounds()
        return {"band_low": float(L), "band_high": float(H)}
    except Exception:
        return {"band_low": None, "band_high": None}


@router.get("/rl/history", tags=["rl"])
async def rl_history():
    hist = HistoryService.get_instance().get_notification_history()
    # Take last 5 entries, most recent first
    last5 = list(hist[-5:])[::-1]
    data = []
    for r in last5:
        try:
            data.append({
                "bad_posture_prob": float(r.bad_posture_prob),
                "delta": None if r.delta is None else float(r.delta),
                "threshold": float(r.threshold),
                "timestamp_ms": int(r.timestamp_ms),
            })
        except Exception:
            continue
    return {"history": data}


@router.post("/rl/threshold/decide", tags=["rl"])
async def rl_threshold_decide(payload: dict):
    try:
        bad_prob = float(payload.get("bad_posture_prob", 0.0))
    except Exception:
        bad_prob = 0.0
    agent = ThresholdTSAgent.get_instance()
    try:
        threshold = agent.estimate_threshold(bad_prob)
    except Exception:
        threshold = agent.get_current_threshold()
    return {"threshold": float(threshold)}


@router.get("/features/aggregate/mean", tags=["aggregation"])
async def features_aggregate_mean():
    mean = _agg.mean_features()
    return {"features": mean}


@router.post("/features/aggregate/clear", tags=["aggregation"])
async def features_aggregate_clear():
    _agg.clear()
    return {"ok": True}
