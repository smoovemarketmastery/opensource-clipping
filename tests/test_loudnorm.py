"""--loudnorm: finished clips are normalized to about -14 LUFS; Kick needs yt-dlp's curl-cffi extra."""

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from clipping import config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ffmpeg_utils_t", ROOT / "clipping" / "studio" / "ffmpeg_utils.py")
fu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fu)

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")


def _make_clip(path, audio_volume=None, seconds=4):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}"]
    if audio_volume is not None:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds},volume={audio_volume}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio_volume is not None else []) + [str(path)]
    subprocess.run(cmd, check=True)


def _integrated_lufs(path):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1])


def _video_codec(path):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()


def test_flag_is_off_by_default_and_parses():
    assert config.build_config(["--url", "https://example.test/v"]).loudnorm is False
    assert config.build_config(["--url", "https://example.test/v", "--loudnorm"]).loudnorm is True


@needs_ffmpeg
def test_quiet_clip_is_raised_to_about_minus_14_lufs(tmp_path):
    clip = tmp_path / "quiet.mp4"
    _make_clip(clip, audio_volume=0.02)
    before = _integrated_lufs(clip)
    assert before < -25
    assert fu.normalize_loudness(str(clip)) is True
    after = _integrated_lufs(clip)
    assert -16.0 <= after <= -12.0, (before, after)
    assert _video_codec(clip) == "h264"                     # video kept
    assert not (tmp_path / "quiet.mp4.loudnorm.mp4").exists()  # temp file cleaned up


@needs_ffmpeg
def test_clip_without_audio_is_left_alone(tmp_path):
    clip = tmp_path / "silent.mp4"
    _make_clip(clip)
    before = clip.read_bytes()
    assert fu.normalize_loudness(str(clip)) is False
    assert clip.read_bytes() == before


@needs_ffmpeg
def test_ffmpeg_failure_keeps_the_original(tmp_path, monkeypatch):
    clip = tmp_path / "keep.mp4"
    _make_clip(clip, audio_volume=0.02)
    before = clip.read_bytes()
    monkeypatch.setattr(fu, "LOUDNORM_FILTER", "loudnorm=I=not_a_number")
    with pytest.raises(subprocess.CalledProcessError):
        fu.normalize_loudness(str(clip))
    assert clip.read_bytes() == before
    assert not (tmp_path / "keep.mp4.loudnorm.mp4").exists()


def test_requirements_install_yt_dlp_with_curl_cffi():
    first = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first.split("#")[0].strip() == "yt-dlp[curl-cffi]"
