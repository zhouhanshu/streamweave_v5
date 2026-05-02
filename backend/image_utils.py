"""Shared image encoding helpers for multimodal backends."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def image_to_jpeg_bytes(path: str | Path, *, max_side: int, quality: int) -> bytes:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def image_to_data_url(path: str | Path, *, max_side: int, quality: int) -> str:
    encoded = base64.b64encode(image_to_jpeg_bytes(path, max_side=max_side, quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
