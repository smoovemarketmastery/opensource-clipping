"""Gameplay layouts for vertical clips, built as one FFmpeg filter graph.

--gameplay-layout "x,y,w,h"  facecam box in the source (fractions of width/height).
                              Output = facecam on top, gameplay panel below.
--gameplay-layout fit         no facecam; the gameplay panel fills the canvas.

The gameplay panel shows --gameplay-crop (default: whole frame), either
"contain" (whole region, letterboxed over a blurred copy; keeps HUD and minimap)
or "cover" (fills the panel, cropped). --gameplay-align-h/-v place the region
inside the panel (contain) or pick the kept part (cover). Only the gameplay
inside its panel moves; the panel itself has a fixed size, so the
facecam + panel stack always fills the canvas exactly.

With a cam box, the copy of the facecam still inside the gameplay is masked
(blur/darken) before scaling, so the streamer doesn't appear twice.

Pure string building lives here so it is testable without OpenCV; cv2 and
numpy are only imported inside make_watermark_overlay().
"""

import os

DEFAULT_CAM_SHARE = 0.35
ALIGN_H = {"left": 0.0, "center": 0.5, "right": 1.0}
ALIGN_V = {"top": 0.0, "center": 0.5, "bottom": 1.0}


def active_layout(cfg, rasio):
    """The layout to render with, or None. Single predicate for renderer and B-roll gating.

    Always returns None (never "", (), False) when no layout applies. cfg may come
    from the Web/API adapter, which never sets the gameplay fields.
    """
    layout = getattr(cfg, "gameplay_layout", None)
    if not layout or rasio == "16:9" or getattr(cfg, "dev_mode", False):
        return None
    return layout


def _even(n):
    return max(2, int(round(n / 2)) * 2)


def _crop_expr(box):
    x, y, w, h = box
    return f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"


def panel_size(layout, out_w, out_h, cam_share=DEFAULT_CAM_SHARE):
    """(width, height) of the gameplay panel."""
    if layout == "fit":
        return out_w, out_h
    return out_w, out_h - _even(out_h * cam_share)


def idle_alignments(src_w, src_h, crop, panel_w, panel_h, fit, align_h, align_v):
    """Alignment flags that can't move anything for this geometry (no slack on that axis)."""
    cw, ch = (crop[2], crop[3]) if crop else (1.0, 1.0)
    region = (src_w * cw) / (src_h * ch)
    panel = panel_w / panel_h
    if abs(region - panel) < 1e-3:
        h_slack = v_slack = False
    elif fit == "contain":
        h_slack, v_slack = region < panel, region > panel
    else:  # cover crops the overflowing axis
        h_slack, v_slack = region > panel, region < panel
    idle = []
    if align_h != "center" and not h_slack:
        idle.append(f"--gameplay-align-h {align_h}")
    if align_v != "center" and not v_slack:
        idle.append(f"--gameplay-align-v {align_v}")
    return idle


def warn_once(cfg, message):
    """Print a warning once per config (one job), not once per render call.

    The record lives on the cfg object, so a new job in a long-lived worker warns again.
    It also names its owner: a shallow copy (main.py clones per URL with copy.copy)
    would otherwise share the set, so a clone starts a fresh one.
    """
    owner, warned = cfg.__dict__.get("_gameplay_warned", (None, None))
    if owner is not cfg:
        warned = set()
        cfg._gameplay_warned = (cfg, warned)
    if message not in warned:
        warned.add(message)
        print(message, flush=True)


def _mask(src_pad, out_pad, cam, mode):
    """Hide the facecam copy inside the gameplay source. Returns (chain, pad)."""
    if not cam or mode == "none":
        return "", src_pad
    x, y, w, h = cam
    if mode == "darken":
        return (
            f"[{src_pad}]drawbox=x=iw*{x}:y=ih*{y}:w=iw*{w}:h=ih*{h}:color=black@0.85:t=fill[{out_pad}];",
            out_pad,
        )
    return (
        f"[{src_pad}]split[{out_pad}_m][{out_pad}_c];"
        f"[{out_pad}_c]{_crop_expr(cam)},boxblur=luma_radius=min(w\\,h)/6:luma_power=3[{out_pad}_b];"
        f"[{out_pad}_m][{out_pad}_b]overlay=main_w*{x}:main_h*{y}[{out_pad}];",
        out_pad,
    )


def _panel(src_pad, out_pad, pw, ph, crop, fit, align_h, align_v):
    """Filter chain turning [src_pad] into a pw x ph gameplay panel [out_pad]."""
    ax, ay = ALIGN_H[align_h], ALIGN_V[align_v]
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


def build_filter(layout, out_w, out_h, *, crop=None, fit="contain", align_h="center",
                 align_v="center", cam_share=DEFAULT_CAM_SHARE, mask_cam=None, watermark=False):
    """FFmpeg filter_complex string ending in [v]. Input 0 = video, input 1 = watermark overlay.

    mask_cam None means the default: blur when a cam box is given.
    """
    last = "base" if watermark else "v"
    pw, ph = panel_size(layout, out_w, out_h, cam_share)
    if layout == "fit":
        graph = _panel("0:v", last, pw, ph, crop, fit, align_h, align_v)
    else:
        cam_h = out_h - ph
        mask_chain, game_pad = _mask("game_src", "game_masked", layout, mask_cam or "blur")
        graph = (
            f"[0:v]split[cam_src][game_src];"
            f"[cam_src]{_crop_expr(layout)},"
            f"scale={out_w}:{cam_h}:force_original_aspect_ratio=increase,crop={out_w}:{cam_h},setsar=1[cam];"
            + mask_chain
            + _panel(game_pad, "game", pw, ph, crop, fit, align_h, align_v)
            + f";[cam][game]vstack[{last}]"
        )
    if watermark:
        # The looped PNG is infinite; shortest=1 lets the main input end the clip.
        graph += ";[base][1:v]overlay=0:0:shortest=1[v]"
    return graph


def make_watermark_overlay(apply_watermark, cfg, out_w, out_h, path):
    """Render the configured watermark to a transparent full-canvas PNG.

    apply_watermark takes and returns OpenCV BGR frames and depends only on cfg
    and the canvas size, so rendering onto black and white canvases recovers
    alpha exactly. The result is written with cv2 as BGRA, keeping BGR order.
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


def options(cfg):
    """Gameplay options from cfg, with defaults for anything missing (Web/API cfg)."""
    return {
        "crop": getattr(cfg, "gameplay_crop", None),
        "fit": getattr(cfg, "gameplay_fit", None) or "contain",
        "align_h": getattr(cfg, "gameplay_align_h", None) or "center",
        "align_v": getattr(cfg, "gameplay_align_v", None) or "center",
        "cam_share": getattr(cfg, "gameplay_cam_share", None) or DEFAULT_CAM_SHARE,
        "mask_cam": getattr(cfg, "gameplay_mask_cam", None),
    }


def render(input_video, output_video, start, end, layout, out_w, out_h, cfg, *,
           encode_args, run_ffmpeg, watermark_png=None, label="Gameplay"):
    """Render a silent clip with the layout (same output contract as buat_video_hybrid).

    encode_args: FFmpeg output encode args (codec, pix_fmt, fps).
    run_ffmpeg: callable(cmd, duration, label) -> (return_code, recent_errors), like
        ffmpeg_utils.run_ffmpeg_with_progress; cmd has no output path yet.
    Only -filter_complex is used (never -vf), so the graph owns every stream.
    FFmpeg creates the output when the muxer opens, so a mid-stream failure leaves a
    partial file: the exit code, not the file's existence, decides success.
    """
    graph = build_filter(layout, out_w, out_h, watermark=bool(watermark_png), **options(cfg))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(start), "-to", str(end), "-i", input_video]
    if watermark_png:
        cmd += ["-loop", "1", "-i", watermark_png]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-an", "-t", str(end - start)] + encode_args
    return_code, recent_errors = run_ffmpeg(cmd, end - start, label)
    if return_code != 0:
        raise RuntimeError(f"{label}: gameplay render failed (ffmpeg exit {return_code}):\n"
                           + "\n".join(recent_errors[-20:]))
    if not os.path.exists(output_video):
        raise RuntimeError(f"{label}: gameplay render produced no file at {output_video}")
