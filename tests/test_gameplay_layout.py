"""Gameplay layout, Twitch/Kick sources and --source-section.

Pure tests run in the lightweight CI env. Render tests need ffmpeg, OpenCV,
NumPy and Pillow and skip otherwise.
"""

import importlib
import importlib.util
import json
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
    assert (cfg.gameplay_crop, cfg.gameplay_fit, cfg.gameplay_align_h, cfg.gameplay_align_v,
            cfg.gameplay_cam_share, cfg.gameplay_mask_cam) == (None, "contain", "center", "center", 0.35, "blur")
    fit = cli("--gameplay-layout", "FIT")
    assert fit.gameplay_layout == "fit" and fit.gameplay_mask_cam == "none"
    assert cli("--gameplay-layout", "0.75,0.70,0.25,0.30", "--gameplay-mask-cam", "darken").gameplay_mask_cam == "darken"
    off = cli()
    assert off.gameplay_layout is None and off.source_section is None


def test_dev_mode_warning_is_per_config_not_per_process(cli, capsys):
    for _ in range(2):  # e.g. two jobs in one long-lived process
        cli("--gameplay-layout", "fit", "--dev-mode")
        assert "ignored in --dev-mode" in capsys.readouterr().out


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
    # Axis/value mismatch and removed single-anchor flag.
    ["--gameplay-layout", "fit", "--gameplay-align-h", "top"],
    ["--gameplay-layout", "fit", "--gameplay-align-v", "left"],
    ["--gameplay-layout", "fit", "--gameplay-anchor", "left"],
    # Mask needs a cam box; sub-options need a layout.
    ["--gameplay-layout", "fit", "--gameplay-mask-cam", "blur"],
    ["--gameplay-fit", "cover"],
    ["--gameplay-align-v", "top"],
    ["--gameplay-mask-cam", "none"],
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
    for empty in ("", (), False, None):
        assert gl.active_layout(SimpleNamespace(gameplay_layout=empty), "9:16") is None


def test_idle_alignment_is_reported_per_axis():
    # 16:9 source into a 1080x1248 panel: contain has only vertical slack, cover only horizontal.
    assert gl.idle_alignments(1920, 1080, None, 1080, 1248, "contain", "left", "top") == [
        "--gameplay-align-h left"]
    assert gl.idle_alignments(1920, 1080, None, 1080, 1248, "cover", "left", "top") == [
        "--gameplay-align-v top"]
    assert gl.idle_alignments(1920, 1080, None, 1080, 1248, "contain", "center", "bottom") == []
    # A tall crop flips it: contain now has horizontal slack.
    assert gl.idle_alignments(1920, 1080, (0, 0, 0.2, 1), 1080, 1248, "contain", "right", "center") == []


# --- Filter graph ---------------------------------------------------------------

def test_contain_is_default_and_keeps_whole_frame():
    graph = gl.build_filter("fit", 1080, 1920)
    assert "force_original_aspect_ratio=decrease" in graph
    assert "boxblur" in graph and graph.endswith("[v]")


def test_alignment_moves_gameplay_inside_fixed_panel():
    cover = gl.build_filter("fit", 1080, 1920, fit="cover", align_h="left", align_v="bottom")
    assert "force_original_aspect_ratio=increase" in cover
    assert "crop=1080:1920:(iw-1080)*0.0:(ih-1920)*1.0" in cover
    contain = gl.build_filter((0.75, 0.7, 0.25, 0.3), 1080, 1920, align_v="top")
    assert "overlay=(W-w)*0.5:(H-h)*0.0" in contain
    # Panel size is fixed regardless of alignment, so cam + panel always fill the canvas.
    assert "crop=1080:672" in contain and "scale=1080:1248:" in contain  # 672 + 1248 = 1920


def test_cam_mask_modes():
    cam = (0.75, 0.7, 0.25, 0.3)
    blur = gl.build_filter(cam, 1080, 1920)  # default = blur
    assert "[game_src]split[game_masked_m][game_masked_c]" in blur
    assert "boxblur=luma_radius=min(w\\,h)/6" in blur
    assert "overlay=main_w*0.75:main_h*0.7[game_masked]" in blur
    assert "drawbox=x=iw*0.75:y=ih*0.7:w=iw*0.25:h=ih*0.3:color=black@0.85:t=fill" in gl.build_filter(
        cam, 1080, 1920, mask_cam="darken")
    none = gl.build_filter(cam, 1080, 1920, mask_cam="none")
    assert "game_masked" not in none and "[game_src]split[game_s1]" in none
    # Blur with no cam box (fit) is a no-op, not a crash.
    assert gl.build_filter("fit", 1080, 1920, mask_cam="blur") == gl.build_filter("fit", 1080, 1920)


def test_warn_once_is_per_config(capsys):
    job1, job2 = SimpleNamespace(), SimpleNamespace()
    for _ in range(5):  # renderer runs several times per clip
        gl.warn_once(job1, "align warning")
    gl.warn_once(job2, "align warning")
    assert capsys.readouterr().out.count("align warning") == 2


def test_overlay_input_is_looped_before_its_input():
    cmd = []

    def capture(c, d, l):
        cmd.extend(c)
        return 0, []

    try:
        gl.render("in.mp4", "missing.mp4", 0, 1, "fit", 1080, 1920, SimpleNamespace(),
                  encode_args=[], run_ffmpeg=capture, watermark_png="wm.png")
    except RuntimeError:
        pass  # no output file: only the command matters here
    i = cmd.index("wm.png")
    assert cmd[i - 3:i + 1] == ["-loop", "1", "-i", "wm.png"]
    assert "-vf" not in cmd and cmd.index("-filter_complex") > i


def test_failed_ffmpeg_with_partial_output_raises(tmp_path):
    out = tmp_path / "out.mp4"

    def failing(cmd, dur, label):
        out.write_bytes(b"x")  # ffmpeg opens the muxer before it fails mid-stream
        return 1, ["boom"]

    with pytest.raises(RuntimeError, match="(?s)ffmpeg exit 1.*boom"):
        gl.render("in.mp4", str(out), 0, 1, "fit", 1080, 1920, SimpleNamespace(),
                  encode_args=[], run_ffmpeg=failing)


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
    for name in ("gameplay_layout", "gameplay_crop", "gameplay_fit", "gameplay_align_h",
                 "gameplay_align_v", "gameplay_cam_share", "gameplay_mask_cam", "source_section"):
        assert not hasattr(cfg, name)
    assert gl.active_layout(cfg, "9:16") is None

    # An explicit layout through render() falls back to defaults for every missing field.
    calls = []
    out = tmp_path / "out.mp4"

    def fake_run(cmd, dur, label):
        calls.append(cmd)
        out.write_bytes(b"x")
        return 0, []

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


# --- Import smoke test: every module in clipping/ ------------------------------

@pytest.fixture
def isolated_modules():
    """Remove the stubs and project modules a smoke import added.

    Real third-party packages that got imported stay loaded: some (OpenCV)
    can't be imported a second time in one process once their submodules
    are removed.
    """
    from unittest.mock import MagicMock
    before = dict(sys.modules)
    yield
    for name in set(sys.modules) - set(before):
        mod = sys.modules[name]
        if isinstance(mod, MagicMock) or name.split(".")[0] in ("clipping", "web", "scripts") \
                or name.startswith("smoke_"):
            del sys.modules[name]
    sys.modules.update(before)


def _import_with_stubs(import_fn, max_stubs=60):
    """Import, stubbing each missing third-party dependency until it loads.

    Only absent packages are stubbed, so installed ones are exercised for real.
    Syntax errors and missing project modules still fail.
    """
    from unittest.mock import MagicMock
    for _ in range(max_stubs):
        try:
            return import_fn()
        except ModuleNotFoundError as exc:
            if not exc.name or exc.name.split(".")[0] in ("clipping", "web", "scripts"):
                raise
            parts = exc.name.split(".")
            for i in range(1, len(parts) + 1):
                sys.modules.setdefault(".".join(parts[:i]), MagicMock())
    raise AssertionError("too many missing dependencies")


CLIPPING_MODULES = sorted(
    p.relative_to(ROOT).with_suffix("").as_posix()
    for p in (ROOT / "clipping").rglob("*.py")
    if "__pycache__" not in p.parts
)


@pytest.mark.parametrize("module", CLIPPING_MODULES)
def test_every_clipping_module_imports(module, isolated_modules, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if module.startswith("clipping/studio/"):
        # Loaded by file path in production (clipping/studio.py shadows the folder).
        def load():
            spec = importlib.util.spec_from_file_location(
                "smoke_" + module.replace("/", "_"), ROOT / f"{module}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    else:
        def load():
            return importlib.import_module(module.replace("/", ".").removesuffix(".__init__"))
    assert _import_with_stubs(load) is not None


# --- Engine: source cache keyed to the request ---------------------------------

def _engine(monkeypatch, downloads=None, fail=None, instances=None, offline=None):
    """engine.py with a fake yt-dlp.

    fail (list): the next download raises mid-way. instances (list): records the
    opts of every YoutubeDL created. offline (list): while non-empty, every
    yt-dlp network call raises.
    """
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            if instances is not None:
                instances.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            if offline:
                raise RuntimeError("Unable to download JSON metadata: offline")
            vid = url.rstrip("/").split("/")[-1].split("?")[0]  # like yt-dlp: id ignores query params
            return {"height": 1080, "vcodec": "avc1", "extractor_key": "TwitchVod", "id": "v" + vid}

        def download(self, urls):
            if offline:
                raise RuntimeError("offline")
            if self.opts.get("skip_download"):
                return  # subtitle-only pass
            out = Path(self.opts["outtmpl"])
            if out.exists():
                return  # real yt-dlp skips an existing file
            ranges = self.opts.get("download_ranges")
            if fail:
                out.write_text("partial", encoding="utf-8")
                raise RuntimeError(fail.pop())
            out.write_text(f"{urls[0]} {ranges}", encoding="utf-8")
            if downloads is not None:
                downloads.append(ranges)

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setitem(sys.modules, "yt_dlp.utils",
                        SimpleNamespace(download_range_func=lambda chapters, ranges: ranges))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=None))
    spec = importlib.util.spec_from_file_location("test_gameplay_engine", ROOT / "clipping" / "engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_probe_duration", lambda path: 38.0)  # same length for every section
    return module


URL = "https://www.twitch.tv/videos/1"


def _sidecar(tmp_path):
    return json.loads((tmp_path / "video_asli.source.json").read_text(encoding="utf-8"))


def test_same_length_sections_do_not_share_the_cache(monkeypatch, tmp_path, capsys):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(500.0, 530.0))
    assert downloads == [[(100.0, 130.0)], [(500.0, 530.0)]]
    assert "(500.0, 530.0)" in out.read_text(encoding="utf-8")
    assert _sidecar(tmp_path) == {"source_key": "TwitchVod:v1", "url": URL, "normalized_urls": [URL],
                                  "section": [500.0, 530.0], "actual_duration": 38.0}
    assert "holds section 100-130s" in capsys.readouterr().out
    assert not (tmp_path / "video_asli.download.mp4").exists()


def test_cache_reuse_rules(monkeypatch, tmp_path):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    sidecar = tmp_path / "video_asli.source.json"
    # Same source and section: reused.
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert len(downloads) == 1
    # Same VOD with ?t= and tracking params: same extractor id, reused.
    engine.download_video(URL + "?t=1h2m&utm_source=x", str(out), source_platform="twitch",
                          source_section=(100.0, 130.0))
    assert len(downloads) == 1
    # Full video after a section: redownloaded, and its own record written.
    engine.download_video(URL, str(out), source_platform="twitch")
    assert downloads[-1] is None and _sidecar(tmp_path)["section"] is None
    # Section after full: redownloaded.
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert downloads[-1] == [(100.0, 130.0)] and len(downloads) == 3
    # Legacy file (no sidecar), full request: reused as before. Section request: redownloaded.
    sidecar.unlink()
    engine.download_video(URL, str(out), source_platform="twitch")
    assert len(downloads) == 3 and not sidecar.exists()  # reuse writes no record
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert len(downloads) == 4
    # Empty file is never trusted, whatever the sidecar says.
    out.write_bytes(b"")
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert len(downloads) == 5


def test_same_url_cache_hit_needs_no_network(monkeypatch, tmp_path):
    instances, offline = [], []
    engine = _engine(monkeypatch, instances=instances, offline=offline)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    before = out.read_text(encoding="utf-8")
    offline.append(True)   # no connection / rate limited: any yt-dlp call would raise
    instances.clear()
    for url in (URL, URL + "?t=1h2m&utm_source=x#chat"):  # normalized URL matches offline
        engine.download_video(url, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert instances == [] and out.read_text(encoding="utf-8") == before


def test_different_url_spelling_falls_back_to_extractor_id(monkeypatch, tmp_path):
    downloads, instances = [], []
    engine = _engine(monkeypatch, downloads, instances=instances)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    instances.clear()
    engine.download_video("https://twitch.tv/videos/1/", str(out), source_platform="twitch",
                          source_section=(100.0, 130.0))  # no www: URL differs, same video id
    assert len(downloads) == 1 and len(instances) == 1  # one metadata call, no media download


def test_offline_with_unmatched_url_fails_clearly(monkeypatch, tmp_path):
    instances, offline = [], []
    engine = _engine(monkeypatch, instances=instances, offline=offline)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    before = out.read_text(encoding="utf-8")
    offline.append(True)
    with pytest.raises(RuntimeError) as exc:
        engine.download_video("https://twitch.tv/videos/1/", str(out), source_platform="twitch",
                              source_section=(100.0, 130.0))
    msg = str(exc.value)
    assert "while offline" in msg and str(out) in msg and URL in msg
    assert "different URL" in msg and "not reused or replaced" in msg
    assert out.read_text(encoding="utf-8") == before  # no silent reuse, no clobber
    # Same URL, different section, offline: the message names the section instead.
    with pytest.raises(RuntimeError, match="holds section 100-130s"):
        engine.download_video(URL, str(out), source_platform="twitch", source_section=(500.0, 530.0))


@pytest.mark.parametrize("old_sidecar,reused", [
    # Round 4: id recorded, no URL list -> metadata check confirms the id, cache reused.
    ({"source_key": "TwitchVod:v1", "url": URL, "section": [100.0, 130.0], "actual_duration": 38.0}, True),
    # Round 3: no id at all -> metadata check can't confirm it, so it is replaced.
    ({"url": URL, "section": [100.0, 130.0], "actual_duration": 38.0}, False),
])
def test_older_sidecars_fall_through_to_metadata_check(monkeypatch, tmp_path, old_sidecar, reused):
    downloads, instances = [], []
    engine = _engine(monkeypatch, downloads, instances=instances)
    out = tmp_path / "video_asli.mp4"
    out.write_text("cached", encoding="utf-8")
    (tmp_path / "video_asli.source.json").write_text(json.dumps(old_sidecar), encoding="utf-8")
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert instances  # no KeyError; the URL check missed and the metadata check ran
    assert (downloads == []) is reused
    assert URL in _sidecar(tmp_path)["normalized_urls"]  # either way, the next run is offline


def test_malformed_sidecar_fields_are_unknown_not_errors(monkeypatch, tmp_path, capsys):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    out.write_text("cached", encoding="utf-8")
    (tmp_path / "video_asli.source.json").write_text(json.dumps(
        {"source_key": "TwitchVod:v1", "normalized_urls": "oops", "section": [1],
         "actual_duration": "long"}), encoding="utf-8")
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert len(downloads) == 1 and "record is malformed" in capsys.readouterr().out


def test_id_match_remembers_the_new_spelling(monkeypatch, tmp_path):
    instances = []
    engine = _engine(monkeypatch, instances=instances)
    out = tmp_path / "video_asli.mp4"
    alt = "https://twitch.tv/videos/1/"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    engine.download_video(alt, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert _sidecar(tmp_path)["normalized_urls"] == [URL, "https://twitch.tv/videos/1"]
    instances.clear()
    for u in (alt, URL):  # both spellings are now offline hits
        engine.download_video(u, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert instances == []


@pytest.mark.parametrize("bad_section", [[9000, 7200], [7200, 7200], [-5, 10], [0, float("inf")],
                                         [float("nan"), 10]])
def test_out_of_order_or_out_of_range_section_is_malformed(monkeypatch, tmp_path, capsys, bad_section):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    out.write_text("cached", encoding="utf-8")
    (tmp_path / "video_asli.source.json").write_text(json.dumps(
        {"source_key": "TwitchVod:v1", "url": URL, "normalized_urls": [URL], "section": bad_section}),
        encoding="utf-8")
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(9000.0, 9030.0))
    printed = capsys.readouterr().out
    assert len(downloads) == 1 and "record is malformed" in printed and "9000-7200s" not in printed


def test_known_urls_are_capped(monkeypatch, tmp_path):
    engine = _engine(monkeypatch)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    # Unknown per-share params survive normalization, so each run is a new spelling.
    for i in range(12):
        engine.download_video(f"{URL}?share={i}", str(out), source_platform="twitch",
                              source_section=(100.0, 130.0))
    urls = _sidecar(tmp_path)["normalized_urls"]
    assert len(urls) == engine.MAX_KNOWN_URLS == 8
    assert urls[-1] == f"{URL}?share=11" and len(set(urls)) == 8


def test_logic_errors_surface_instead_of_looking_malformed(monkeypatch, tmp_path):
    engine = _engine(monkeypatch)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))

    def broken(section):
        raise TypeError("bug in decision logic")

    monkeypatch.setattr(engine, "_normalize_section", broken)
    with pytest.raises(TypeError, match="bug in decision logic"):
        engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))


def test_section_never_runs_the_subtitle_pass(monkeypatch, tmp_path):
    instances = []
    engine = _engine(monkeypatch, instances=instances)
    engine.download_video("https://www.youtube.com/watch?v=abc", str(tmp_path / "video_asli.mp4"),
                          use_dlp_subs=True, source_platform="youtube", source_section=(100.0, 130.0))
    assert instances and not any(o.get("skip_download") or o.get("writesubtitles") for o in instances)
    # Control: without a section the subtitle pass does run.
    instances.clear()
    engine.download_video("https://www.youtube.com/watch?v=def", str(tmp_path / "full.mp4"),
                          use_dlp_subs=True, source_platform="youtube")
    assert any(o.get("writesubtitles") for o in instances)


def test_runner_does_not_request_subtitles_for_a_section(monkeypatch, cli):
    cfg = cli("--use-dlp-subs", "--source-section", "100-130")
    import clipping
    seen = []

    class Stop(Exception):
        pass

    def download(*args, **kwargs):
        seen.append(args[2])  # use_dlp_subs, positional
        raise Stop

    fake_engine = SimpleNamespace(download_video=download)
    monkeypatch.setattr(clipping, "engine", fake_engine, raising=False)
    monkeypatch.setitem(sys.modules, "clipping.engine", fake_engine)
    for name in ("metadata", "studio", "diarization", "hook_manager", "voiceover"):
        stub = SimpleNamespace()
        monkeypatch.setattr(clipping, name, stub, raising=False)
        monkeypatch.setitem(sys.modules, f"clipping.{name}", stub)
    monkeypatch.delitem(sys.modules, "clipping.runner", raising=False)
    monkeypatch.delattr(clipping, "runner", raising=False)  # package attr survives the delitem
    from clipping import runner
    with pytest.raises(Stop):
        runner.run_pipeline(cfg)
    cfg.source_section = None
    with pytest.raises(Stop):
        runner.run_pipeline(cfg)
    assert seen == [False, True]


def test_three_section_spellings_share_one_cache_entry(monkeypatch, tmp_path, cli):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    for spelling in ("7200-7230", "02:00:00-02:00:30", "7200.0-7230.0"):
        section = cli("--source-section", spelling).source_section
        engine.download_video(URL, str(out), source_platform="twitch", source_section=section)
    assert len(downloads) == 1


def test_failed_download_keeps_previous_source(monkeypatch, tmp_path):
    fail = []
    engine = _engine(monkeypatch, fail=fail)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    before_video, before_sidecar = out.read_text(encoding="utf-8"), _sidecar(tmp_path)
    fail.append("HTTP Error 403: Forbidden")  # cache miss, then the new download dies
    with pytest.raises(RuntimeError, match="403"):
        engine.download_video(URL, str(out), source_platform="twitch", source_section=(500.0, 530.0))
    assert out.read_text(encoding="utf-8") == before_video
    assert _sidecar(tmp_path) == before_sidecar
    assert not (tmp_path / "video_asli.download.mp4").exists()  # partial file cleaned up


def test_corrupt_sidecar_redownloads_without_error(monkeypatch, tmp_path, capsys):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    (tmp_path / "video_asli.source.json").write_text('{"source_key": "Twit', encoding="utf-8")
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert len(downloads) == 2 and "record is unreadable" in capsys.readouterr().out
    assert _sidecar(tmp_path)["section"] == [100.0, 130.0]


def test_truncated_cache_is_redownloaded(monkeypatch, tmp_path, capsys):
    downloads = []
    engine = _engine(monkeypatch, downloads)
    out = tmp_path / "video_asli.mp4"
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    monkeypatch.setattr(engine, "_probe_duration", lambda path: 12.0 if Path(path) == out else 38.0)
    engine.download_video(URL, str(out), source_platform="twitch", source_section=(100.0, 130.0))
    assert len(downloads) == 2 and "length changed from 38.0s to 12.0s" in capsys.readouterr().out


def test_section_drops_full_video_subtitles(monkeypatch, tmp_path):
    engine = _engine(monkeypatch)
    out = tmp_path / "video_asli.mp4"
    stale = tmp_path / "video_asli.en.json3"
    stale.write_text("{}", encoding="utf-8")
    engine.download_video(URL, str(out), use_dlp_subs=True, source_platform="twitch",
                          source_section=(100.0, 130.0))
    assert not stale.exists()


def test_cleanup_source_removes_video_and_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import main
    video = tmp_path / "video_asli.mp4"
    for name in ("video_asli.mp4", "video_asli.source.json", "video_asli.download.mp4"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    main._cleanup_source_video(SimpleNamespace(file_video_asli=str(video)))
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_batch_urls_each_emit_the_alignment_warning(monkeypatch, tmp_path, capsys):
    """Runs main.py's real batch loop (copy.copy per URL) with the pipeline stubbed."""
    monkeypatch.chdir(tmp_path)
    for name in ("NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # main() exits without one; never sent anywhere
    monkeypatch.setattr(sys, "argv", ["main.py", "--url", "https://example.test/a", "https://example.test/b",
                                      "--gameplay-layout", "fit", "--gameplay-align-h", "left"])
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=lambda *a, **k: None))

    def run_pipeline(cfg, whisper_model_instance=None):
        for _ in range(4):  # several renderer calls per clip
            gl.warn_once(cfg, "ALIGN-WARNING")
        return []

    fake_runner = SimpleNamespace(run_pipeline=run_pipeline)
    import clipping
    monkeypatch.setattr(clipping, "runner", fake_runner, raising=False)
    monkeypatch.setitem(sys.modules, "clipping.runner", fake_runner)
    import main
    main.main()
    assert capsys.readouterr().out.count("ALIGN-WARNING") == 2


def test_warn_once_survives_clone_after_warning(capsys):
    import copy
    cfg = SimpleNamespace()
    gl.warn_once(cfg, "W")
    clone = copy.copy(cfg)  # shares the stored set, but not its owner
    gl.warn_once(clone, "W")
    gl.warn_once(cfg, "W")
    assert capsys.readouterr().out.count("W") == 2


def test_section_length_mismatch_warns(monkeypatch, capsys):
    engine = _engine(monkeypatch)
    monkeypatch.setattr(engine, "_probe_duration", lambda path: 38.0)
    engine._report_section_duration("x.mp4", (7200.0, 7230.0))
    out = capsys.readouterr().out
    assert "Section length: 38.0s (requested 30.0s)" in out and "approximate" in out
    monkeypatch.setattr(engine, "_probe_duration", lambda path: 30.4)
    engine._report_section_duration("x.mp4", (7200.0, 7230.0))
    assert "approximate" not in capsys.readouterr().out


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
        done = subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(out)], capture_output=True, text=True)
        return done.returncode, done.stderr.splitlines()[-20:]

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


def _has_scalable_font():
    """Whether the text watermark can get a real TrueType font, as TextWatermarkRenderer looks it up.

    Without one it falls back to Pillow's ~10px bitmap font, too small for the
    region assertion below. The image-watermark tests cover the alpha/color math
    on any machine; this one only checks text placement.
    """
    from PIL import ImageFont
    font_dir = Path.cwd() / "custom_fonts"
    for candidate in [font_dir / "Montserrat-Black.ttf", font_dir / "Montserrat-Regular.ttf", "arial.ttf"]:
        if isinstance(candidate, Path) and not candidate.exists():
            continue
        try:
            ImageFont.truetype(str(candidate), 40)
            return True
        except OSError:
            continue
    return False


@render_deps
def test_watermark_is_burned_in(tmp_path, source):
    if not _has_scalable_font():
        pytest.skip("no TrueType font for the text watermark (custom_fonts/Montserrat-*.ttf or arial.ttf)")
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


@render_deps
def test_colored_image_watermark_keeps_bgr_order(tmp_path, source):
    from PIL import Image
    red = tmp_path / "red.png"
    Image.new("RGBA", (200, 200), (255, 0, 0, 255)).save(red)  # pure red, RGB
    wm_mod = _load("watermark.py", "test_gameplay_watermark_color_mod")
    cfg = SimpleNamespace(watermark_enabled=True, watermark_text=None, watermark_image=str(red),
                          watermark_opacity=100, watermark_position="center", watermark_padding=0,
                          watermark_font_size=0, watermark_scale=20)
    png = gl.make_watermark_overlay(wm_mod.apply_watermark, cfg, 1080, 1920, str(tmp_path / "wm.png"))
    import cv2
    b, g, r, a = cv2.imread(png, cv2.IMREAD_UNCHANGED)[960, 540]
    assert (b, g, r, a) == (0, 0, 255, 255)
    _, _, frame = _render(source, tmp_path / "wm.mp4", "fit", SimpleNamespace(), wm=png)
    b, g, r = frame[900:1020, 480:600].reshape(-1, 3).mean(axis=0)
    assert r > 200 and g < 50 and b < 50, (b, g, r)


@render_deps
def test_half_opacity_watermark_keeps_alpha_and_color(tmp_path):
    from PIL import Image
    red = tmp_path / "red.png"
    Image.new("RGBA", (200, 200), (255, 0, 0, 255)).save(red)
    wm_mod = _load("watermark.py", "test_gameplay_watermark_half_mod")
    cfg = SimpleNamespace(watermark_enabled=True, watermark_text=None, watermark_image=str(red),
                          watermark_opacity=50, watermark_position="center", watermark_padding=0,
                          watermark_font_size=0, watermark_scale=20)
    png = gl.make_watermark_overlay(wm_mod.apply_watermark, cfg, 1080, 1920, str(tmp_path / "wm.png"))
    import cv2
    overlay = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    b, g, r, a = overlay[900:1020, 480:600].reshape(-1, 4).mean(axis=0)
    assert abs(a - 127.5) <= 2, a              # 50% opacity survives
    assert r >= 250 and g <= 3 and b <= 3      # and color is unpremultiplied back to pure red
    assert overlay[:200, :200, 3].max() == 0   # outside the watermark stays transparent


@render_deps
def test_cam_mask_changes_only_the_duplicate(tmp_path, source):
    cam = (0.75, 0.7, 0.25, 0.3)
    _, _, unmasked = _render(source, tmp_path / "none.mp4", cam, SimpleNamespace(gameplay_mask_cam="none"))
    _, _, blurred = _render(source, tmp_path / "blur.mp4", cam, SimpleNamespace(gameplay_mask_cam="blur"))
    diff = abs(blurred.astype(int) - unmasked.astype(int)).mean(axis=2)
    assert diff[:672].mean() < 1                    # facecam itself is untouched
    # Contain: 1920x1080 -> 1080x608 inside the 1248-tall panel, centered (top 320).
    # The cam copy sits at x 75-100%, y 70-100% of that image.
    copy = diff[672 + 320 + int(608 * 0.72):672 + 320 + 606, 1080 * 3 // 4 + 10:1070]
    assert copy.mean() > 5


@render_deps
def test_top_aligned_contain_still_fills_the_stack(tmp_path, source):
    dims, _, frame = _render(source, tmp_path / "top.mp4", (0.75, 0.7, 0.25, 0.3),
                             SimpleNamespace(gameplay_align_v="top"))
    assert dims == (1080, 1920)
    assert frame[-60:].mean() > 20  # bottom of the panel is the blurred fill, not an empty bar
