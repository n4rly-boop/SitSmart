from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import AnalyzeResponse, FeatureExtractionResponse, Notification, NotificationSeverity
from app.services.pose_service import PoseService, PoseServiceUnavailable
from app.services.feature_extractor import PostureFeatureExtractor

router = APIRouter()


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
