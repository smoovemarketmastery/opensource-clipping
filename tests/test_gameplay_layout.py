"""Gameplay layout, Twitch/Kick sources and --source-section.

Pure tests run in the lightweight CI env. Render tests need ffmpeg, OpenCV,
NumPy and Pillow and skip otherwise.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipping import config

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "clipping" / "studio"


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, STUDIO / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


gl = _load("gameplay_layout.py", "test_gameplay_layout_mod")


@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    return lambda *args: config.build_config(["--url", "https://example.test/v", *args])


# --- CLI parsing and validation -------------------------------------------------

def test_flags_parse_with_defaults(cli):
    cfg = cli("--gameplay-layout", "0.75,0.70,0.25,0.30", "--source-section", "1:00:00-1:00:30")
    assert cfg.gameplay_layout == (0.75, 0.70, 0.25, 0.30)
    assert cfg.source_section == (3600.0, 3630.0)
    assert (cfg.gameplay_crop, cfg.gameplay_fit, cfg.gameplay_anchor, cfg.gameplay_cam_share) == (
        None, "contain", "center", 0.35)
    assert cli("--gameplay-layout", "FIT").gameplay_layout == "fit"
    off = cli()
    assert off.gameplay_layout is None and off.source_section is None


@pytest.mark.parametrize("args", [
    ["--gameplay-layout", "0.9,0.9,0.3,0.3"],
    ["--gameplay-layout", "1,2,3"],
    ["--gameplay-layout", "a,b,c,d"],
    ["--gameplay-crop", "0,0,0,0.5"],
    ["--gameplay-cam-share", "0.9"],
    ["--source-section", "3:00-2:00"],
    ["--source-section", "abc"],
    ["--gameplay-layout", "fit", "--split-screen"],
    ["--gameplay-layout", "fit", "--camera-switch"],
])
def test_invalid_values_and_conflicts_are_parser_errors(cli, args):
    with pytest.raises(SystemExit) as exc:
        cli(*args)
    assert exc.value.code == 2


def test_twitch_and_kick_are_sources(cli):
    assert cli("--source", "twitch").source_platform == "twitch"
    assert cli("--source", "kick").source_platform == "kick"


# --- Predicate shared by renderer and B-roll gating ----------------------------

def test_active_layout_predicate():
    on = SimpleNamespace(gameplay_layout="fit", dev_mode=False)
    assert gl.active_layout(on, "9:16") == "fit"
    assert gl.active_layout(on, "16:9") is None
    assert gl.active_layout(SimpleNamespace(gameplay_layout="fit", dev_mode=True), "9:16") is None
    assert gl.active_layout(SimpleNamespace(), "9:16") is None  # Web/API cfg has no gameplay fields


# --- Filter graph ---------------------------------------------------------------

def test_contain_is_default_and_keeps_whole_frame():
    graph = gl.build_filter("fit", 1080, 1920)
    assert "force_original_aspect_ratio=decrease" in graph
    assert "boxblur" in graph and graph.endswith("[v]")


def test_cover_crops_at_anchor():
    graph = gl.build_filter("fit", 1080, 1920, fit="cover", anchor="left")
    assert "force_original_aspect_ratio=increase" in graph
    assert "crop=1080:1920:(iw-1080)*0.0:(ih-1920)*0.5" in graph


def test_facecam_share_crop_and_watermark():
    graph = gl.build_filter((0.75, 0.7, 0.25, 0.3), 1080, 1920, crop=(0, 0, 0.75, 1),
                            cam_share=0.25, watermark=True)
    assert "scale=1080:480:" in graph           # 0.25 * 1920
    assert "scale=1080:1440:" in graph          # gameplay panel gets the rest
    assert "crop=iw*0.75:ih*1:iw*0:ih*0" in graph
    assert graph.endswith("[base][1:v]overlay=0:0:shortest=1[v]")


# --- Web/API config: new flags absent ------------------------------------------

def test_web_payload_cfg_renders_without_new_flags(monkeypatch, tmp_path):
    from web.api.config_adapter import build_config_from_payload
    cfg = build_config_from_payload({"url": "https://example.test/v", "ratio": "9:16"}, "job-gameplay")
    for name in ("gameplay_layout", "gameplay_crop", "gameplay_fit", "gameplay_anchor",
                 "gameplay_cam_share", "source_section"):
        assert not hasattr(cfg, name)
    assert gl.active_layout(cfg, "9:16") is None

    # An explicit layout through render() falls back to defaults for every missing field.
    calls = []
    out = tmp_path / "out.mp4"

    def fake_run(cmd, dur, label):
        calls.append(cmd)
        out.write_bytes(b"x")

    gl.render("in.mp4", str(out), 1.0, 3.0, "fit", 1080, 1920, cfg,
              encode_args=["-c:v", "libx264"], run_ffmpeg=fake_run)
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert graph == gl.build_filter("fit", 1080, 1920)

    # The runner passes source_section explicitly, and None for a Web/API cfg.
    import clipping
    seen = {}

    class Stop(Exception):
        pass

    def download(*args, **kwargs):
        seen.update(kwargs)
        raise Stop

    fake_engine = SimpleNamespace(download_video=download)
    monkeypatch.setattr(clipping, "engine", fake_engine, raising=False)
    monkeypatch.setitem(sys.modules, "clipping.engine", fake_engine)
    for name in ("metadata", "studio", "diarization", "hook_manager", "voiceover"):
        stub = SimpleNamespace()
        monkeypatch.setattr(clipping, name, stub, raising=False)
        monkeypatch.setitem(sys.modules, f"clipping.{name}", stub)
    monkeypatch.delitem(sys.modules, "clipping.runner", raising=False)
    from clipping import runner
    with pytest.raises(Stop):
        runner.run_pipeline(cfg)
    assert "source_section" in seen and seen["source_section"] is None


# --- Real renders (need ffmpeg + OpenCV + NumPy + Pillow) ----------------------

render_deps = pytest.mark.skipif(
    not shutil.which("ffmpeg")
    or any(importlib.util.find_spec(m) is None for m in ("cv2", "numpy", "PIL")),
    reason="needs ffmpeg, OpenCV, NumPy and Pillow",
)


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=3",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    return src


def _render(src, out, layout, cfg, wm=None):
    def run(cmd, dur, label):
        subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(out)], check=True)

    gl.render(str(src), str(out), 0.5, 2.5, layout, 1080, 1920, cfg,
              encode_args=["-c:v", "libx264", "-preset", "ultrafast"], run_ffmpeg=run, watermark_png=wm)
    import cv2
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    secs = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert ok
    return dims, round(secs, 1), frame


@render_deps
@pytest.mark.parametrize("layout,fit", [
    ((0.75, 0.7, 0.25, 0.3), "contain"), ("fit", "contain"), ("fit", "cover"),
])
def test_layouts_render_vertical(tmp_path, source, layout, fit):
    dims, secs, _ = _render(source, tmp_path / "o.mp4", layout, SimpleNamespace(gameplay_fit=fit))
    assert dims == (1080, 1920) and secs == 2.0


@render_deps
def test_watermark_is_burned_in(tmp_path, source):
    wm_mod = _load("watermark.py", "test_gameplay_watermark_mod")
    cfg = SimpleNamespace(watermark_enabled=True, watermark_text="SMOOVE", watermark_image=None,
                          watermark_opacity=100, watermark_position="top-left", watermark_padding=40,
                          watermark_font_size=120, watermark_scale=15)
    png = gl.make_watermark_overlay(wm_mod.apply_watermark, cfg, 1080, 1920, str(tmp_path / "wm.png"))
    import cv2
    overlay = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    assert overlay.shape == (1920, 1080, 4) and overlay[:, :, 3].max() > 200
    _, _, plain = _render(source, tmp_path / "plain.mp4", "fit", SimpleNamespace())
    dims, secs, marked = _render(source, tmp_path / "wm.mp4", "fit", SimpleNamespace(), wm=png)
    assert dims == (1080, 1920) and secs == 2.0
    near = (slice(0, 400), slice(0, 800))
    assert abs(marked[near].astype(int) - plain[near].astype(int)).mean() > 5
    far = (slice(1500, 1900), slice(300, 800))
    assert abs(marked[far].astype(int) - plain[far].astype(int)).mean() < 2
