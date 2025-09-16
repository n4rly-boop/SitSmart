# SitSmart

Real-time posture detection using MMPose RTMPose Halpe‑26 (backend PyTorch) via FastAPI, with a web demo client.

## Modes
- Backend: Halpe‑26 skeleton via MMPose/Torch at `/api/ws/analyze`.

## Docker
### Build and run
```bash
docker compose up --build
# open http://localhost:8000/static/index.html
```

### Use Halpe‑26 backend (Torch)
1) Ensure you have your Halpe‑26 RTMPose config and checkpoint:
   - Config (example): `./models/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py`
   - Weights (example): `./models/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth`
2) Environment (optional):
   - `HALPE_CFG` path to config (defaults to the example above)
   - `HALPE_PTH` path to checkpoint (defaults to the example above)
   - `HALPE_DEVICE` set to `cpu` or `cuda`
3) Connect to WS `ws://localhost:8000/api/ws/analyze` and send JPEG frames; receive `{label, score, details, kpts}` where `kpts` are normalized to [0,1].

### Docker
Build and run with mounted models:
```bash
docker compose up --build
# open http://localhost:8000/static/index.html
```
The compose file mounts `./models` into `/models` and sets `HALPE_CFG`, `HALPE_PTH`, and `HALPE_DEVICE` accordingly.

## Local dev
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Notes
- Requires PyTorch, MMPose, MMDetection (for default detector). GPU optional.
- Client demo mirrors video; adjust as needed. The demo uses `/api/ws/halpe` previously; now it connects to `/api/ws/analyze`.