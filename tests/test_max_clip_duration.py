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
    assert "Rank 2: cut to 58s cap (--max-clip-duration)" in capsys.readouterr().out


def test_string_clip_times_come_back_as_floats():
    [out] = metadata.cap_clip_durations([{"start_time": "100", "end_time": "189"}], 58)
    assert out["end_time"] - out["start_time"] == 58


def test_cap_to_one_segment_renders_that_segment_window():
    clip = {"start_time": 0.0, "end_time": 120.0, "keep_segments": [
        {"start_time": 5.0, "end_time": 45.0}, {"start_time": 55.0, "end_time": 85.0}]}
    [out] = metadata.cap_clip_durations([clip], 40)
    assert (out["start_time"], out["end_time"]) == (5.0, 45.0)
    assert out["keep_segments"] == [{"start_time": 5.0, "end_time": 45.0}]


def test_existing_single_segment_clip_keeps_its_window():
    clip = {"start_time": 0.0, "end_time": 100.0, "keep_segments": [{"start_time": 5.0, "end_time": 50.0}]}
    [out] = metadata.cap_clip_durations([clip], 58)
    assert (out["start_time"], out["end_time"]) == (0.0, 58.0)


def test_malformed_segments_are_skipped_without_changing_the_budget():
    clip = {"start_time": 0.0, "end_time": 200.0, "keep_segments": [
        "junk", {"start_time": "1:40", "end_time": 110}, {"end_time": 5},
        {"start_time": 80.0, "end_time": 20.0},   # reversed: dropped, must not add budget
        {"start_time": 0.0, "end_time": 40.0},
        {"start_time": 100.0, "end_time": 140.0},  # only 18s fits
        {"start_time": 150.0, "end_time": 190.0},  # over budget: dropped
    ]}
    [out] = metadata.cap_clip_durations([clip], 58)
    assert out["keep_segments"] == [{"start_time": 0.0, "end_time": 40.0},
                                    {"start_time": 100.0, "end_time": 118.0}]


def test_segment_dict_instead_of_list_is_accepted():
    clip = {"start_time": 0.0, "end_time": 90.0, "keep_segments": {"start_time": 0.0, "end_time": 90.0}}
    [out] = metadata.cap_clip_durations([clip], 58)
    assert out["keep_segments"] == [{"start_time": 0.0, "end_time": 58.0}]


def test_notice_reports_the_segment_total_when_it_is_shorter(capsys):
    clip = {"rank": 3, "start_time": 0.0, "end_time": 120.0, "keep_segments": [
        {"start_time": 0.0, "end_time": 10.0}, {"start_time": 20.0, "end_time": 30.0}]}
    metadata.cap_clip_durations([clip], 58)
    assert "Rank 3: cut to 58s cap (20s after segment trim)" in capsys.readouterr().out


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
