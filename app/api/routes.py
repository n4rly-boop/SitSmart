from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import FeatureExtractionResponse, Notification, NotificationSeverity, RLAgentState, ModelAnalysisRequest, ModelAnalysisResponse, FeatureVector
from app.services.pose_service import PoseService, PoseServiceUnavailable
from app.services.notification_service import NotificationService
from app.services.feature_buffer import FeatureBuffer
from app.services.rl_service import RLService
from app.services.ml_service import MLService

router = APIRouter()

_notification_ws_clients = set()
_last_notification: Notification | None = None
_buffer = FeatureBuffer()


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


@router.get("/rl/state", response_model=RLAgentState, tags=["rl"])
async def rl_state():
    # delegate to RL service
    return RLAgentState.model_validate(RLService.get_instance().get_state())


# RL analyze route (same contract as future ML analyze route)
@router.post("/rl/analyze", response_model=ModelAnalysisResponse, tags=["rl"])
async def rl_analyze(payload: ModelAnalysisRequest):
    resp = RLService.get_instance().analyze(payload)
    return resp


# Future ML analyze stub with the same contract
@router.post("/ml/analyze", response_model=ModelAnalysisResponse, tags=["ml"])
async def ml_analyze(payload: ModelAnalysisRequest):
    resp = MLService.get_instance().analyze(payload)
    return resp


# Decide using the server buffer (mean over window), then notify
@router.post("/decide/from_buffer", response_model=ModelAnalysisResponse, tags=["decision"])
async def decide_from_buffer(method: str = "rl"):
    try:
        mean_features = _buffer.mean()
    except Exception:
        mean_features = None
    if not mean_features:
        return ModelAnalysisResponse(should_notify=False, reason="no-features")
    # Toggle RL training according to method
    try:
        from app.services.rl_service import RLService as _RS
        _RS.get_instance().set_training_enabled(method == "rl")
    except Exception:
        pass
    NotificationService.get_instance().decide_and_notify(mean_features, method=method)
    # Also return model decision for clients who call this endpoint
    if method == "rl":
        resp = RLService.get_instance().analyze(ModelAnalysisRequest(features=FeatureVector(**mean_features)))
    else:
        resp = MLService.get_instance().analyze(
            ModelAnalysisRequest(features=FeatureVector(**mean_features))
        )
    return resp

# Feature aggregation management endpoints
@router.post("/features/aggregate/add", tags=["aggregation"])
async def features_aggregate_add(payload: FeatureVector):
    try:
        _buffer.add(payload.model_dump())
    except Exception:
        pass
    return {"ok": True}


@router.get("/features/aggregate/mean", tags=["aggregation"])
async def features_aggregate_mean():
    mean = _buffer.mean()
    return {"features": mean}


@router.post("/features/aggregate/clear", tags=["aggregation"])
async def features_aggregate_clear():
    from app.services.feature_buffer import FeatureBuffer as _FB
    # reinitialize buffer to clear it
    global _buffer
    _buffer = _FB(window_seconds=_buffer.window_seconds)
    return {"ok": True}
