"""Frame extraction used before runtime starts."""

from __future__ import annotations

from pathlib import Path


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float,
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
    overwrite: bool = False,
    max_frames: int = 0,
) -> int:
    """Extract frames as 000000.jpg, 000001.jpg, ...

    This is preprocessing. Runtime code consumes the generated files only.
    """
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in target.glob(f"*.{image_ext}"):
            old.unlink()

    try:
        return _extract_with_decord(
            src,
            target,
            fps=fps,
            image_ext=image_ext,
            jpeg_quality=jpeg_quality,
            max_frames=max_frames,
        )
    except ImportError:
        return _extract_with_av(
            src,
            target,
            fps=fps,
            image_ext=image_ext,
            jpeg_quality=jpeg_quality,
            max_frames=max_frames,
        )


def _extract_with_decord(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float,
    image_ext: str,
    jpeg_quality: int,
    max_frames: int,
) -> int:
    import decord
    import numpy as np
    from PIL import Image

    reader = decord.VideoReader(str(video_path))
    total = len(reader)
    raw_fps = float(reader.get_avg_fps())
    duration = total / max(raw_fps, 1e-6)
    times = np.arange(0.0, duration, 1.0 / max(fps, 1e-6))
    if max_frames:
        times = times[:max_frames]
    if len(times) == 0:
        return 0
    indices = np.clip(np.round(times * raw_fps).astype(int), 0, total - 1)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    for idx, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        _save_image(image, output_dir / f"{idx:06d}.{image_ext}", jpeg_quality=jpeg_quality)
    return len(frames)


def _extract_with_av(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float,
    image_ext: str,
    jpeg_quality: int,
    max_frames: int,
) -> int:
    import av

    interval = 1.0 / max(fps, 1e-6)
    next_ts = 0.0
    count = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            ts = frame.time
            if ts is None:
                if frame.pts is None:
                    continue
                ts = float(frame.pts * stream.time_base)
            ts = float(ts)
            if ts < next_ts:
                continue
            image = frame.to_image().convert("RGB")
            while ts >= next_ts:
                _save_image(image, output_dir / f"{count:06d}.{image_ext}", jpeg_quality=jpeg_quality)
                count += 1
                if max_frames and count >= max_frames:
                    return count
                next_ts += interval
    return count


def _save_image(image: object, path: Path, *, jpeg_quality: int) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=jpeg_quality)
    else:
        image.save(path)

