# SitSmart

Real-time posture detection and feedback using MediaPipe Pose, served by FastAPI.

## Features
- Minimal static demo page to stream webcam frames over WebSocket (server version) and a client-side WASM demo
- Pose features extracted from MediaPipe Pose landmarks

## Architecture (current)
- Images → `POST /api/features` → returns single-frame `FeatureVector` (no side effects)
- Optional server buffer:
  - `POST /api/features/aggregate/add` to append a sample
  - `GET /api/features/aggregate/mean` to read current time‑window mean
  - `POST /api/features/aggregate/clear` to reset
- Decision:
  - `POST /api/decide/from_buffer?method=rl|ml` → averages current buffer, calls selected model, may notify via webhook → broadcast on `WS /api/ws/notifications`
- RL introspection: `GET /api/rl/state`

## Quickstart
### Build and run
```
docker compose up --build
```
Open `http://localhost:8000/static/index.html`.

#### Environment variables
- `FEATURE_BUFFER_SECONDS` (default: `5`)
  - Time-window size (seconds) used by the server-side `FeatureBuffer` to compute mean features.
- `NOTIFICATION_WEBHOOK_URL` (default: `http://127.0.0.1:8000/api/notifications/webhook`)
  - Where `NotificationService` POSTs notifications. By default it hits the app’s own broadcasting endpoint.
- `RL_AGENT_STATE_PATH` (default: `./train/rl_agent_state.json`)
  - File where the RL prototype persists weights/epsilon/reward stats.

### HTTP API
- `POST /api/features` with form-data `image` (file): returns geometric posture features (pure extraction only).
- `POST /api/features/aggregate/add` JSON `FeatureVector`: append a sample to the server buffer.
- `GET /api/features/aggregate/mean`: get mean features over buffer window.
- `POST /api/features/aggregate/clear`: clear buffer.
- `POST /api/decide/from_buffer?method=rl|ml`: decide using buffer mean and optionally notify.

- `POST /api/rl/analyze` JSON: analyze features via RL service, response: `{ should_notify: bool, score?, reason?, details? }`.
- `POST /api/ml/analyze` JSON: stub ML analyze route with the same response contract.
- `GET /api/rl/state`: RL agent state for debugging.

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
    rl_service.py
    notification_service.py
    feature_buffer.py
  utils/
    image_io.py
static/
  index.html
```

## Where to implement ML (quick guide)
- Add a service at `app/services/ml_service.py` that exposes `MLService.analyze(ModelAnalysisRequest) -> ModelAnalysisResponse`.
- Wire the route in `app/api/routes.py` inside `@router.post("/ml/analyze")` to call your `MLService` (currently a stub).
- Update `POST /api/decide/from_buffer?method=ml` to call your `MLService` (currently a stub).
- Live path is `POST /api/decide/from_buffer?method=ml` (frontend method selector pauses RL training automatically).
- Contract: reuse `ModelAnalysisRequest`/`ModelAnalysisResponse` from `app/api/schemas.py` so ML and RL remain interchangeable.

## RL agent (prototype)
- Location: `app/services/rl_service.py` contains config, a simple contextual‑bandit agent, and service façade.
- Logic: epsilon‑greedy over a linear value function per action (0=don’t notify, 1=notify). Online TD‑style update after a delayed reward window.
- State vector: normalized posture features + bias.
- Persistence: JSON file defined by `RL_AGENT_STATE_PATH` (default `train/rl_agent_state.json`).
- Control: training is enabled when method=rl and paused when method=ml; pending evals are cancelled if training is disabled.
- Caveat: this is a prototype meant for iteration; expect tuning/feature additions for use.

## Notes
- The client-side demo (`static/index.html`) uses MediaPipe Tasks (WASM) and runs entirely in the browser.
- The server endpoints remain available for future needs; the demo page does not require them.