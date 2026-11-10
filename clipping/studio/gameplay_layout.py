"""Gameplay layouts for vertical clips, built as one FFmpeg filter graph.

Two modes, chosen by --gameplay-layout:
  "x,y,w,h"  facecam box in the source, as fractions of width/height.
             Output = facecam on top, center-cropped gameplay below.
  "fit"      no facecam. Whole gameplay frame scaled to width over a
             blurred, zoomed copy of itself.
"""

import subprocess

import cv2

# ponytail: fixed facecam share of output height; make it a flag if streamers' cams need different sizes
CAM_SHARE = 0.35


def parse_layout(value):
    """argparse type: returns "fit" or a validated (x, y, w, h) tuple of fractions."""
    if value is None or value.strip().lower() == "fit":
        return None if value is None else "fit"
    try:
        x, y, w, h = (float(p) for p in value.split(","))
    except ValueError:
        raise ValueError(
            f"--gameplay-layout '{value}': use 'fit' or four fractions 'x,y,w,h', e.g. 0.75,0.70,0.25,0.30"
        )
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.0001 and y + h <= 1.0001):
        raise ValueError(
            f"--gameplay-layout '{value}': x,y,w,h must be fractions 0-1 and the box must fit inside the frame"
        )
    return (x, y, w, h)


def _even(n):
    return max(2, int(round(n / 2)) * 2)


def build_filter(layout, out_w, out_h):
    """FFmpeg filter_complex string that ends in the [v] output pad."""
    if layout == "fit":
        return (
            f"[0:v]split[a][b];"
            f"[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=20:2[bg];"
            f"[b]scale={out_w}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
        )
    x, y, w, h = layout
    cam_h = _even(out_h * CAM_SHARE)
    game_h = out_h - cam_h
    return (
        f"[0:v]split[a][b];"
        f"[a]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},"
        f"scale={out_w}:{cam_h}:force_original_aspect_ratio=increase,crop={out_w}:{cam_h}[cam];"
        f"[b]crop='min(iw,ih*{out_w}/{game_h})':'min(ih,iw*{game_h}/{out_w})',"
        f"scale={out_w}:{game_h}[game];"
        f"[cam][game]vstack,setsar=1[v]"
    )


def render(input_video, output_video, start, end, layout, out_w, out_h,
           encode_args_fn, video_encoder, progress_cmd_fn, run_progress_fn, label):
    """Render a silent clip with the layout. Same contract as buat_video_hybrid's output file."""
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end), "-i", input_video,
        "-filter_complex", build_filter(layout, out_w, out_h),
        "-map", "[v]", "-an",
    ] + encode_args_fn(video_encoder, fps)
    run_progress_fn(progress_cmd_fn(cmd, output_video), end - start, label=label)


if __name__ == "__main__":
    # Self-check: parsing, validation, and that both graphs actually render.
    import os, tempfile
    assert parse_layout("fit") == "fit"
    assert parse_layout("0.75,0.7,0.25,0.3") == (0.75, 0.7, 0.25, 0.3)
    for bad in ["0.9,0.9,0.3,0.3", "1,2,3", "a,b,c,d", "0,0,0,0.5"]:
        try:
            parse_layout(bad)
            raise AssertionError(f"accepted bad layout {bad}")
        except ValueError:
            pass
    d = tempfile.mkdtemp()
    src = os.path.join(d, "src.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=3",
                    "-pix_fmt", "yuv420p", src], check=True)
    for name, lay in [("cam", (0.75, 0.7, 0.25, 0.3)), ("fit", "fit")]:
        out = os.path.join(d, f"{name}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-to", "2.5", "-i", src,
                        "-filter_complex", build_filter(lay, 1080, 1920), "-map", "[v]", "-an",
                        "-pix_fmt", "yuv420p", out], check=True)
        c = cv2.VideoCapture(out)
        wh = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        c.release()
        assert wh == (1080, 1920), (name, wh)
        print(f"ok {name} {wh} -> {out}")
    print("gameplay_layout self-check passed")
