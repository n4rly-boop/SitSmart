# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OpenCV / PyTorch ecosystem
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 && \
    pip install --no-cache-dir openmim && \
    mim install "mmcv==2.1.0" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Optional warmup: pre-initialize MMPose and download default detector weights
# This avoids long delays and client disconnects on the first request.
RUN python - <<'PY'
import os, sys
try:
    import torch, numpy as np
    # Allow legacy numpy reconstruct during potential checkpoint load
    try:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    except Exception:
        pass
    from mmpose.apis import MMPoseInferencer
    cfg = os.environ.get('HALPE_CFG', 'models/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py')
    wts = os.environ.get('HALPE_PTH', 'models/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth')
    infer = MMPoseInferencer(pose2d=cfg, pose2d_weights=wts, device='cpu')
    # Trigger detector/model init by running one dummy frame
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    gen = infer(dummy, return_vis=False)
    next(gen)
    print('Warmup completed.')
except Exception as e:
    print(f'Warmup skipped: {e}', file=sys.stderr)
PY

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
