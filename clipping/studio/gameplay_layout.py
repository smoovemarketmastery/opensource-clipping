"""Gameplay layouts for vertical clips, built as one FFmpeg filter graph.

--gameplay-layout "x,y,w,h"  facecam box in the source (fractions of width/height).
                              Output = facecam on top, gameplay panel below.
--gameplay-layout fit         no facecam; the gameplay panel fills the canvas.

The gameplay panel shows --gameplay-crop (default: whole frame), either
"contain" (whole region, letterboxed over a blurred copy; keeps HUD and minimap)
or "cover" (fills the panel, cropped at --gameplay-anchor).

Pure string building lives here so it is testable without OpenCV; cv2 and the
watermark module are only touched inside render()/make_watermark_overlay().
"""

import os

DEFAULT_CAM_SHARE = 0.35

# Anchor -> fractional (x, y) position of the kept/placed area within the slack.
ANCHORS = {
    "center": (0.5, 0.5),
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
}

_warned_dev_mode = False


def active_layout(cfg, rasio):
    """The layout to render with, or None. Single predicate for renderer and B-roll gating.

    cfg may come from the Web/API adapter, which never sets the gameplay fields,
    so every read goes through getattr with a None default.
    """
    global _warned_dev_mode
    layout = getattr(cfg, "gameplay_layout", None)
    if not layout or rasio == "16:9":
        return None
    if getattr(cfg, "dev_mode", False):
        if not _warned_dev_mode:
            print("   ⚠️ --gameplay-layout is ignored in --dev-mode (dev output has its own fixed size).", flush=True)
            _warned_dev_mode = True
        return None
    return layout


def _even(n):
    return max(2, int(round(n / 2)) * 2)


def _crop_expr(box):
    x, y, w, h = box
    return f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"


def _panel(src_pad, out_pad, pw, ph, crop, fit, anchor):
    """Filter chain turning [src_pad] into a pw x ph gameplay panel [out_pad]."""
    ax, ay = ANCHORS[anchor]
    pre = f"{_crop_expr(crop)}," if crop else ""
    if fit == "cover":
        return (
            f"[{src_pad}]{pre}scale={pw}:{ph}:force_original_aspect_ratio=increase,"
            f"crop={pw}:{ph}:(iw-{pw})*{ax}:(ih-{ph})*{ay},setsar=1[{out_pad}]"
        )
    return (
        f"[{src_pad}]{pre}split[{out_pad}_s1][{out_pad}_s2];"
        f"[{out_pad}_s1]scale={pw}:{ph}:force_original_aspect_ratio=increase,"
        f"crop={pw}:{ph},boxblur=20:2[{out_pad}_bg];"
        f"[{out_pad}_s2]scale={pw}:{ph}:force_original_aspect_ratio=decrease:force_divisible_by=2[{out_pad}_fg];"
        f"[{out_pad}_bg][{out_pad}_fg]overlay=(W-w)*{ax}:(H-h)*{ay},setsar=1[{out_pad}]"
    )


def build_filter(layout, out_w, out_h, *, crop=None, fit="contain", anchor="center",
                 cam_share=DEFAULT_CAM_SHARE, watermark=False):
    """FFmpeg filter_complex string ending in [v]. Input 0 = video, input 1 = watermark overlay (if any)."""
    last = "v" if not watermark else "base"
    if layout == "fit":
        graph = _panel("0:v", last, out_w, out_h, crop, fit, anchor)
    else:
        cam_h = _even(out_h * cam_share)
        game_h = out_h - cam_h
        graph = (
            f"[0:v]split[cam_src][game_src];"
            f"[cam_src]{_crop_expr(layout)},"
            f"scale={out_w}:{cam_h}:force_original_aspect_ratio=increase,crop={out_w}:{cam_h},setsar=1[cam];"
            + _panel("game_src", "game", out_w, game_h, crop, fit, anchor)
            + f";[cam][game]vstack[{last}]"
        )
    if watermark:
        graph += ";[base][1:v]overlay=0:0:shortest=1[v]"
    return graph


def make_watermark_overlay(apply_watermark, cfg, out_w, out_h, path):
    """Render the configured watermark to a transparent full-canvas PNG.

    Reuses the frame renderer's own apply_watermark, so position, padding,
    scale and opacity match the other render modes exactly. Alpha is recovered
    by rendering onto black and white canvases.
    """
    import cv2
    import numpy as np

    black = apply_watermark(np.zeros((out_h, out_w, 3), np.uint8), cfg).astype(np.float32)
    white = apply_watermark(np.full((out_h, out_w, 3), 255, np.uint8), cfg).astype(np.float32)
    alpha = np.clip(1.0 - (white - black).mean(axis=2, keepdims=True) / 255.0, 0.0, 1.0)
    color = np.where(alpha > 0, black / np.maximum(alpha, 1e-6), 0)
    bgra = np.dstack([np.clip(color, 0, 255), alpha * 255]).round().astype(np.uint8)
    if not cv2.imwrite(path, bgra):
        raise RuntimeError(f"Could not write watermark overlay to {path}")
    return path


def render(input_video, output_video, start, end, layout, out_w, out_h, cfg, *,
           encode_args, run_ffmpeg, watermark_png=None, label="Gameplay"):
    """Render a silent clip with the layout (same output contract as buat_video_hybrid).

    encode_args: list of FFmpeg output encode args (codec, pix_fmt, fps).
    run_ffmpeg: callable(cmd, duration, label) that runs and reports progress.
    """
    graph = build_filter(
        layout, out_w, out_h,
        crop=getattr(cfg, "gameplay_crop", None),
        fit=getattr(cfg, "gameplay_fit", "contain") or "contain",
        anchor=getattr(cfg, "gameplay_anchor", "center") or "center",
        cam_share=getattr(cfg, "gameplay_cam_share", None) or DEFAULT_CAM_SHARE,
        watermark=bool(watermark_png),
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(start), "-to", str(end), "-i", input_video]
    if watermark_png:
        cmd += ["-loop", "1", "-i", watermark_png]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-an", "-t", str(end - start)] + encode_args
    run_ffmpeg(cmd, end - start, label)
    if not os.path.exists(output_video):
        raise RuntimeError(f"{label}: gameplay render produced no file at {output_video}")
