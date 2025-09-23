from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, Request
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import AnalyzeResponse, FeatureExtractionResponse, Notification, NotificationSeverity
from app.services.pose_service import PoseService, PoseServiceUnavailable
from app.services.feature_extractor import PostureFeatureExtractor
from app.services.notification_service import NotificationService

router = APIRouter()

_notification_ws_clients = set()
_last_notification: Notification | None = None


@router.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        result = PoseService.get_instance().analyze_image(image_bytes)
        return AnalyzeResponse.model_validate(result)
    except PoseServiceUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface error to client for now
        raise HTTPException(status_code=400, detail=f"Failed to analyze image: {e}")


@router.post("/features", response_model=FeatureExtractionResponse, tags=["analysis"])
async def extract_features(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        extractor = PostureFeatureExtractor()
        result = extractor.extract_features(image_bytes)
        # Non-blocking notification check (fire-and-forget); do not raise if it fails
        try:
            features = result.get("features")
            should_notify = NotificationService.get_instance().maybe_notify(features)
            print(
                "[routes:/features] features present=",
                bool(features),
                "should_notify=",
                should_notify,
            )
        except Exception as e:
            print(f"[routes:/features] notification check failed: {e}")
        return FeatureExtractionResponse.model_validate(result)
    except PoseServiceUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface error to client for now
        raise HTTPException(status_code=400, detail=f"Failed to extract features: {e}")


@router.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    await websocket.accept()
    try:
        service = PoseService.get_instance()
    except PoseServiceUnavailable as e:
        try:
            await websocket.send_json({"error": str(e)})
        finally:
            # Closing once with a normal code; avoid double close
            await websocket.close(code=1013)
        return

    await websocket.send_json({"status": "ready"})

    try:
        while True:
            try:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if "text" in message and message["text"] is not None:
                    text = message["text"].strip().lower()
                    if text == "ping":
                        await websocket.send_json({"pong": True})
                    continue
                frame = message.get("bytes")
                if not frame:
                    continue
                try:
                    result = service.analyze_image(frame)
                    await websocket.send_json(result)
                except Exception as e:  # noqa: BLE001
                    await websocket.send_json({"error": f"analyze_failed: {e}"})
            except WebSocketDisconnect:
                break
    except Exception:
        # If something unexpected happens, attempt a graceful close
        try:
            await websocket.close()
        except Exception:
            pass


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
