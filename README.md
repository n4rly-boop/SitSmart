# SitSmart

Real-time posture detection and feedback using MediaPipe Pose, served by FastAPI.

## Features
- Minimal static demo page to stream webcam frames over WebSocket (server version) and a client-side WASM demo
- Pose features extracted from MediaPipe Pose landmarks

## Quickstart
### Build and run
```
docker compose up --build
```
Open `http://localhost:8000/static/index.html`.

### HTTP API
- `POST /api/analyze` with form-data `image` (file): returns heuristic label and score.
- `POST /api/features` with form-data `image` (file): returns geometric posture features.

Example curl:
```
curl -s -X POST \
  -F "image=@/path/to/frame.jpg" \
  http://localhost:8000/api/features | jq .
```

## Project Structure
```
app/
  main.py
  api/
    routes.py
    schemas.py
  services/
    pose_service.py
    feature_extractor.py
  utils/
    image_io.py
static/
  index.html
```

## Notes
- The client-side demo (`static/index.html`) uses MediaPipe Tasks (WASM) and runs entirely in the browser.
- The server endpoints remain available for future needs; the demo page does not require them.