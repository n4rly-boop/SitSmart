from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import AnalyzeResponse
from app.services.pose_service import PoseService, PoseServiceUnavailable

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
