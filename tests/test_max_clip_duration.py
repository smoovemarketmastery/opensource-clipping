"""--max-clip-duration: prompt limit, config validation, and the post-AI cap."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from clipping import config, metadata

ROOT = Path(__file__).resolve().parents[1]


def test_long_clip_is_cut_at_the_end_and_input_is_untouched():
    clip = {"rank": 1, "start_time": 100.0, "end_time": 189.0}
    [out] = metadata.cap_clip_durations([clip], 58)
    assert (out["start_time"], out["end_time"]) == (100.0, 158.0)
    assert clip["end_time"] == 189.0


def test_short_clip_is_unchanged(capsys):
    clip = {"rank": 1, "start_time": 10.0, "end_time": 40.0,
            "keep_segments": [{"start_time": "10", "end_time": "40"}]}
    [out] = metadata.cap_clip_durations([clip], 58)
    assert out["end_time"] == 40.0
    assert out["keep_segments"] == [{"start_time": 10.0, "end_time": 40.0}]
    assert "cut to" not in capsys.readouterr().out


def test_keep_segments_share_one_budget(capsys):
    clip = {"rank": 2, "start_time": 0.0, "end_time": 120.0, "keep_segments": [
        {"start_time": 0.0, "end_time": 30.0},
        {"start_time": 50.0, "end_time": 80.0},   # only 28s of this fits
        {"start_time": 90.0, "end_time": 120.0},  # dropped
    ]}
    [out] = metadata.cap_clip_durations([clip], 58)
    assert out["keep_segments"] == [{"start_time": 0.0, "end_time": 30.0},
                                    {"start_time": 50.0, "end_time": 78.0}]
    assert sum(s["end_time"] - s["start_time"] for s in out["keep_segments"]) == 58
    assert "Rank 2: cut to 58s" in capsys.readouterr().out


def test_flag_default_and_bounds(capsys):
    assert config.build_config(["--url", "https://example.test/v"]).max_clip_duration == 179
    assert config.build_config(["--url", "https://example.test/v", "--max-clip-duration", "58"]).max_clip_duration == 58
    for bad in ("19", "180"):
        with pytest.raises(SystemExit):
            config.build_config(["--url", "https://example.test/v", "--max-clip-duration", bad])
    assert "--max-clip-duration must be 20-179" in capsys.readouterr().err


def test_prompt_uses_the_run_limit(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=None))
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", SimpleNamespace(download_range_func=None))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=None))
    spec = importlib.util.spec_from_file_location("test_max_dur_engine", ROOT / "clipping" / "engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    prompt = engine.get_analysis_prompt("[0 - 1] hi", 3, 2, cfg=SimpleNamespace(max_clip_duration=58))
    assert "20-58 detik" in prompt and "179" not in prompt
    assert "20-179 detik" in engine.get_analysis_prompt("[0 - 1] hi", 3, 2)
