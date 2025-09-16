from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import AnalyzeResponse
from app.services.pose_service import PoseServiceUnavailable, HalpeService

router = APIRouter()


@router.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"]) 
async def analyze(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        result = HalpeService.get_instance().analyze_image(image_bytes)
        return AnalyzeResponse.model_validate(result)
    except PoseServiceUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface error to client for now
        raise HTTPException(status_code=400, detail=f"Failed to analyze image: {e}")


@router.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    await websocket.accept()
    # Send ready immediately to avoid client timeouts on heavy model init
    try:
        await websocket.send_json({"status": "ready", "skeleton": "halpe26"})
    except Exception:
        try:
            await websocket.close()
        finally:
            return

    service: HalpeService | None = None

    try:
        while True:
            try:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                frame = message.get("bytes")
                if not frame:
                    continue
                try:
                    if service is None:
                        # Notify client about initialization since it may take a few seconds
                        try:
                            await websocket.send_json({"status": "initializing"})
                        except Exception:
                            pass
                        service = HalpeService.get_instance()
                    result = service.analyze_image(frame)
                    await websocket.send_json(result)
                except Exception as e:  # noqa: BLE001
                    await websocket.send_json({"error": f"analyze_failed: {e}"})
            except WebSocketDisconnect:
                break
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/halpe")
async def ws_halpe(websocket: WebSocket):
    await websocket.accept()
    try:
        service = HalpeService.get_instance()
    except PoseServiceUnavailable as e:
        try:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close(code=1013)
        return

    await websocket.send_json({"status": "ready", "skeleton": "halpe26"})

    try:
        while True:
            try:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                frame = message.get("bytes")
                if not frame:
                    continue
                try:
                    result = service.analyze_image(frame)
                    await websocket.send_json(result)
                except Exception as e:  # noqa: BLE001
                    await websocket.send_json({"error": f"halpe_failed: {e}"})
            except WebSocketDisconnect:
                break
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
