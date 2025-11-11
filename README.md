# SitSmart

Real-time posture detection and feedback using MediaPipe Pose, served by FastAPI.

## Pipeline summary
- Extract Mediapipe-based posture features over a short time window (server buffer averages them).
- Use ML to estimate `bad_pose_prob` from the averaged features.
- Compare `bad_pose_prob` to a threshold to decide whether to send a notification.
- The threshold is adapted online by the RL agent in `app/services/rl_service.py` to reduce spam while keeping useful alerts.

## Features
- Minimal static demo page to stream webcam frames over WebSocket (server version) and a client-side WASM demo
- Pose features extracted from MediaPipe Pose landmarks

## Architecture (current)
- Images → `POST /api/features` → returns single-frame `FeatureVector` (pure extraction)
- Optional server buffer:
  - `POST /api/features/aggregate/add` to append a sample
  - `GET /api/features/aggregate/mean` to read current time‑window mean
  - `POST /api/features/aggregate/clear` to reset
- Decision (ML + Adaptive Threshold):
  - `POST /api/decide/from_buffer` → averages current buffer, runs ML to get `bad_posture_prob`, compares against current threshold from RL agent, may notify via webhook → broadcast on `WS /api/ws/notifications`
- RL status: `GET /api/rl/status`

## Quickstart
### Build and run
```
docker compose up --build
```
Open `http://localhost:8000/static/index.html`.

#### Environment variables
- `FEATURE_BUFFER_SECONDS` (default: `12`)
  - Time-window size (seconds) used by the server-side `FeatureBuffer` to compute mean features (also acts as minimum notification cooldown).
- `NOTIFICATION_COOLDOWN_SECONDS` (default: `15`)
  - Minimum interval between notifications (effective cooldown is `max(FEATURE_BUFFER_SECONDS, NOTIFICATION_COOLDOWN_SECONDS)`).
- `NOTIFICATION_WEBHOOK_URL` (default: `http://127.0.0.1:8000/api/notifications/webhook`)
  - Where `NotificationService` POSTs notifications. By default it hits the app’s own broadcasting endpoint.
- `ML_BAD_PROB_THRESHOLD` (default: `0.6`)
  - Initial/static threshold used if RL is unavailable; also seeds the RL agent’s initial threshold.
- RL adaptive threshold parameters:
  - `RL_THRESHOLD_STEP` (default: `0.03`) – step size for threshold adjustments.
  - `RL_TAU_MIN` (default: `0.5`) / `RL_TAU_MAX` (default: `0.95`) – hard threshold bounds.
  - `RL_BAND_Q_LOW` (default: `0.3`) / `RL_BAND_Q_HIGH` (default: `0.6`) – target delta quantiles for band boundaries L/H.
  - `RL_BAND_Q_LR` (default: `0.03`) – learning rate for online quantile estimation.
  - `RL_DELTA_RANGE_SECONDS` (default: `8.0`) – delay between notification and delta measurement window.

### HTTP API
- `POST /api/features` with form-data `image` (file): returns geometric posture features (pure extraction only).
- `POST /api/features/aggregate/add` JSON `FeatureVector`: append a sample to the server buffer.
- `GET /api/features/aggregate/mean`: get mean features over buffer window.
- `POST /api/features/aggregate/clear`: clear buffer.
- `POST /api/decide/from_buffer`: decide using buffer mean; runs ML to produce `bad_posture_prob`, compares to the adaptive threshold, and may notify.
- `POST /api/ml/analyze` JSON: analyze features via ML service, response: `{ bad_posture_prob: float }`.
- `GET /api/rl/status`: RL agent status (current threshold, band bounds, and recent history sample).

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
  ml_training/
    data/
      features.csv
    training_of_classifier_model.py
  models/
    model.pkl
    scaler.pkl
  services/
    pose_service.py
    rl_service.py
    ml_service.py
    notification_service.py
    feature_buffer.py
    feature_aggregate_service.py
    history_service.py
    calibration_service.py
  utils/
    image_io.py
static/
  index.html
```

## RL adaptive threshold (how it works — simple version)

### The problem
- Notifications fire when `bad_prob ≥ threshold`
- Goal: adapt the threshold to reduce spam and keep useful notifications
- Delta (0–1) measures posture change after a notification

### The solution: adaptive band boundaries
The agent learns two boundaries (L and H) for each user:
1. L (low boundary): ~30th percentile of deltas
2. H (high boundary): ~60th percentile of deltas

### Action rules (deterministic)
When delta arrives:
- If `delta < L` → raise threshold (low delta = normal pose, reduce notifications)
- If `delta > H` → lower threshold (high delta = extreme pose, catch more)
- If `L ≤ delta ≤ H` → hold (good range, don't change)

### How L and H are learned
Uses Robbins-Monro quantile estimation:

![/meta/L_equation.png](/meta/L_equation.png)
![/meta/H_equation.png](/meta/H_equation.png)

Where the indicator is 1 if the condition holds, else 0.
- If many deltas are below L, L increases
- If many deltas are above H, H decreases
- Over time, L converges to the 30th percentile and H to the 60th percentile

### Complete flow
1. Notification decision:
   - Check `bad_prob ≥ threshold`
   - Fire notification if true
   - Store decision (wait for delta)
2. Delta arrives (after ~8 seconds):
   - Update L and H using the delta
   - Select action: raise/lower/hold based on delta vs bounds
   - Update threshold: `threshold = threshold + action`
3. Repeat

### Example
After 20 notifications, L ≈ 0.12, H ≈ 0.22:
- Delta = 0.10 (< 0.12) → raise threshold
- Delta = 0.15 (between 0.12 and 0.22) → hold
- Delta = 0.25 (> 0.22) → lower threshold

### Why it works
- Adapts to each user: L/H reflect their delta distribution
- Simple: deterministic rules, no complex models
- Efficient: O(1) memory and computation
- Self-correcting: quantiles adapt as behavior changes

In short: learn two boundaries (L, H) from user deltas, then use simple rules to adjust the threshold based on where delta falls relative to those boundaries.

### How delta is computed
Delta aggregates normalized per-feature changes between the notification time and a later snapshot:
![/meta/delta_equation.png](/meta/delta_equation.png)

where:
- Delta is computed using calibration ranges (angle features are scaled conservatively),
- n is the number of valid features
- b = 0.1 is a small bias,
- See `app/services/history_service.py` for details.

## Notes
- The client-side demo (`static/index.html`) uses MediaPipe Tasks (WASM) and runs entirely in the browser.
- The server endpoints remain available for future needs; the demo page does not require them.