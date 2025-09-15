from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


def load_pil_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_resize_np(img: Image.Image, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr
