# SitSmart

Real-time posture auditing pipeline built with FastAPI, MediaPipe Pose, a classical ML classifier, and an adaptive notification policy.

## What’s inside
- `app/` – FastAPI app, domain services (pose extraction, ML, RL, notification, calibration, history).
- `app/models/` – serialized logistic-regression classifier and scaler loaded by `MLService`.
- `app/ml_training/` – Jupyter-style script plus figures/metrics for training the classifier on extracted features.
- `dataset/` – feature CSV and raw frames gathered from the webcam/WebSocket demo.
- `scripts/` – data-collection helpers (e.g., `extract_features.sh` to populate `dataset/features.csv` from the API).
- `static/index.html` – Web demo that streams webcam frames via WebSocket for quick manual testing.

## System overview
1. Client captures webcam frames and either runs MediaPipe locally (demo page) or uploads frames to `POST /api/features`.
2. `PoseService` computes geometric features (shoulder angles, head tilt, offsets) from MediaPipe landmarks.
3. `FeatureAggregateService` buffers features for `FEATURE_BUFFER_SECONDS` to build a denoised mean vector.
4. `MLService` (LogReg + StandardScaler) outputs `bad_posture_prob` given the aggregated features.
5. `AdaptiveThresholdAgent` (RL) compares the score to a dynamic threshold and throttles notifications.
6. `NotificationService` enforces cooldowns, posts payloads to `NOTIFICATION_WEBHOOK_URL`, and broadcasts via `/ws/notifications`.
7. `HistoryService` stores each decision, waits `RL_DELTA_RANGE_SECONDS`, re-samples posture, and reports deltas back to the RL agent for learning.

## Architecture highlights
- **API layer:** `app/api/routes.py` exposes REST + WebSocket endpoints for feature extraction, aggregation, ML scoring, RL status, calibration, and notifications.
- **Pose extraction:** `PoseService` wraps MediaPipe Pose in static-image mode, returning normalized landmarks and engineered features (`PoseFeatures` dataclass).
- **Feature buffering:** `FeatureBuffer` keeps `(timestamp, features)` entries in a sliding window and returns means/last samples to smooth jitter before inference.
- **Decision core:** `MLService` loads `model.pkl` / `scaler.pkl` and always processes features in the fixed `FEATURE_ORDER`. Probabilities feed the RL policy.
- **Adaptive thresholding:** `AdaptiveThresholdAgent` is a singleton using Robbins–Monro quantile tracking to maintain low/high delta bands and choose deterministic actions (raise/lower/hold threshold).
- **History & calibration:** `HistoryService` holds all notifications with feature snapshots, computes deltas with calibration-aware normalization, and feeds RL. `CalibrationService` optionally records per-user feature ranges and resets aggregation/history when calibration stops.
- **Notification delivery:** `NotificationService` wraps cooldown logic, builds user-facing payloads, and reports every sent notification back to `HistoryService`.

## Machine learning pipeline
- **Feature space:** five interpretable signals – shoulder line angle, head tilt, perpendicular head-to-shoulder distance in px, the distance normalized by shoulder width, and shoulder width.
- **Dataset creation:** `scripts/extract_features.sh` iterates over labeled folders in `parsing/total_images/{good,bad}`, calls `/api/features`, and assembles `dataset/features.csv`. This keeps the schema identical to runtime extraction.
- **Training script:** `app/ml_training/training_of_classifier_model.py` cleans the CSV, enforces absolute angles, splits the data (stratified), scales with `StandardScaler`, and searches a logistic regression grid (C, solver, class weights) with 12-fold CV optimizing macro recall.
- **Artifacts:** the best estimator and scaler are serialized into `app/models/model.pkl` and `scaler.pkl` and later loaded by the API. The script also writes evaluation figures and `metrics_summary.csv` for traceability.

## Adaptive thresholding (RL) in practice
- Each notification stores `(bad_prob, threshold, timestamp, mean features at t₀)`. After `RL_DELTA_RANGE_SECONDS`, the system samples fresh features, normalizes per-dimension differences using calibration ranges, and collapses them into a single delta ∈ [0,1].
![delta.png](meta/delta_equation.png)
- **Quantile bands:** the agent learns low/high boundaries (targets ≈30% and ≈60%) through online Robbins–Monro updates so that the bands follow the user’s personal response distribution.
![L_equation.png](meta/L_equation.png)
![H_equation.png](meta/H_equation.png)
- **Policy:** 
  - `delta < L` → threshold += `eta` (fewer alerts after weak reactions).
  - `delta > H` → threshold -= `eta` (more alerts when posture worsens).
  - `L ≤ delta ≤ H` → no change.
- **Constraints:** thresholds are clamped to `[RL_TAU_MIN, RL_TAU_MAX]`, and the static `ML_BAD_PROB_THRESHOLD` is used whenever RL is unavailable. This keeps the system stable and spam-free even with noisy inputs.

## Installation & running
1. Clone the repository:
```bash
git clone https://github.com/n4rly-boop/SitSmart.git
```

2. Copy .env.example to .env and fill in the values:
```bash
cd SitSmart && cp .env.example .env
```
3. Build the Docker image:
```bash
docker compose up --build
```

This builds the FastAPI image (with MediaPipe system deps) and exposes the service on `localhost:8000`.
You can access frontend at `localhost:8000/static/index.html`.

### Key environment variables
- `FEATURE_BUFFER_SECONDS` (default `12`): sliding window for averaging pose features; also sets the minimum notification cadence.
- `NOTIFICATION_COOLDOWN_SECONDS` (default `15`): extra cooldown enforced by `NotificationService`.
- `NOTIFICATION_WEBHOOK_URL` (default internal webhook): destination for outbound notifications before they are rebroadcast over WebSocket.
- `ML_BAD_PROB_THRESHOLD` (default `0.6`): fallback threshold + RL seed.
- `RL_THRESHOLD_STEP`, `RL_TAU_MIN`, `RL_TAU_MAX`, `RL_BAND_Q_LOW`, `RL_BAND_Q_HIGH`, `RL_BAND_Q_LR`, `RL_DELTA_RANGE_SECONDS`, `RL_BIAS`: tune the adaptive policy’s step size, bounds, and delta computation.

## API & interfaces
- `POST /api/features` – upload a frame (`image` form-data) to get engineered features + normalized landmarks.
- `POST /api/features/aggregate/add` / `GET /api/features/aggregate/mean|clear` – manage the server-side feature buffer.
- `POST /api/decide/from_buffer` – run the full ML+RL decision pipeline on the current mean features; may trigger a notification through the webhook + `/ws/notifications`.
- `POST /api/ml/analyze` – pure ML scoring with explicit features provided in the request body.
- `GET /api/rl/status` – inspect the live threshold, low/high delta bands, and the most recent history entries.
- `GET /api/notifications/last`, `GET /api/notifications/config`, `POST /api/notifications/config` – interact with the notification subsystem.
- `GET /api/features/ranges`, `/api/calibration/*` – monitor or control calibration.
- `GET /api/health`, `POST /api/reset` – system housekeeping.

Example feature extraction:
```bash
curl -s -X POST -F "image=@/path/to/frame.jpg" http://localhost:8000/api/features | jq
```

## Utilities & data assets
- `scripts/extract_features.sh` automates dataset refreshes by running through labeled folders and writing directly to `dataset/features.csv`.
- `parsing/` keeps raw captured frames and augmentation scripts used for experimentation.
- `meta/` holds figures explaining the RL math (included in `README` as contextual references if needed).
- `train/` is a workspace for potential RL agent checkpoints (`RL_AGENT_STATE_PATH` hook).
- `static/index.html` lets you test posture detection fully in-browser using MediaPipe Tasks; the backend services remain ready for future integrations (mobile, desktop reminders, etc.).
