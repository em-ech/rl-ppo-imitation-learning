"""Shared helpers for side-by-side policy comparison videos.

A title bar is drawn above each panel, panels are concatenated horizontally with
separators, and the result is written with imageio (moviepy's fps handling is
broken by a decorator-version mismatch in this environment).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLE_H = 34  # height of the title bar above each panel


def _load_font(size: int):
    """matplotlib ships DejaVuSans; fall back to PIL's tiny bitmap font."""
    try:
        import matplotlib
        from pathlib import Path
        p = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf"
        return ImageFont.truetype(str(p), size)
    except Exception:
        return ImageFont.load_default()


_FONT = _load_font(22)


def title_bar(width: int, text: str) -> np.ndarray:
    """A dark strip with centred white text, to sit above a panel's video."""
    img = Image.new("RGB", (width, TITLE_H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), text, font=_FONT)
    tw, th = box[2] - box[0], box[3] - box[1]
    d.text(((width - tw) // 2, (TITLE_H - th) // 2 - box[1]), text,
           fill=(255, 255, 255), font=_FONT)
    return np.asarray(img)


def titled_track(frames: list, text: str) -> list:
    """Stack the title bar above every frame of one panel."""
    bar = title_bar(frames[0].shape[1], text)
    return [np.vstack([bar, f]) for f in frames]


def stack_panels(tracks: list, sep_w: int = 6) -> list:
    """Horizontally concatenate equal-height panel tracks (truncating to the
    shortest) with white separators between them."""
    length = min(len(t) for t in tracks)
    full_h = tracks[0][0].shape[0]
    sep = np.full((full_h, sep_w, 3), 255, dtype=np.uint8)
    out = []
    for i in range(length):
        row = []
        for t, track in enumerate(tracks):
            if t:
                row.append(sep)
            row.append(track[i])
        out.append(np.concatenate(row, axis=1))
    return out


def write_mp4(frames: list, path, fps: int = 30):
    import imageio
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", macro_block_size=1)
    return path
