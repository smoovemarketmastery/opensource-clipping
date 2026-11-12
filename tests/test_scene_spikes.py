"""scripts/scene_spikes.py: pick --source-section windows from scene cuts and loudness."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from clipping import config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scene_spikes", ROOT / "scripts" / "scene_spikes.py")
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def test_cuts_and_loudness_both_raise_the_score():
    scores = ss.second_scores(cuts=[5.2, 5.8, 20.1], levels=[-40.0] * 10 + [-16.0] * 5 + [-40.0] * 16, length=30)
    assert scores[5] == 2.0                     # two cuts in second 5
    assert scores[10] == pytest.approx(4.0)     # 24 dB above the -40 median = 4 points
    assert scores[0] == 0.0


def test_best_windows_are_ranked_non_overlapping_and_parse_as_sections():
    scores = [0.0] * 120
    for i in range(30, 40):
        scores[i] = 3.0                          # strongest burst
    for i in range(80, 85):
        scores[i] = 2.0
    picks = ss.best_windows(scores, cuts=[31, 32, 81], window=20, top=3)
    assert picks[0]["start"] <= 30 and picks[0]["end"] >= 40 and picks[0]["cuts"] == 2
    assert picks[1]["start"] <= 80 < picks[1]["end"]
    for a in picks:
        for b in picks:
            if a is not b:
                assert a["end"] <= b["start"] or b["end"] <= a["start"]
    start, end = config._parse_section(f"{ss.hms(picks[0]['start'])}-{ss.hms(picks[0]['end'])}")
    assert end - start == 20


def test_window_longer_than_video_returns_the_whole_video():
    picks = ss.best_windows([1.0] * 31, cuts=[], window=600, top=3)
    assert len(picks) == 1 and (picks[0]["start"], picks[0]["end"]) == (0.0, 30.0)


def test_nothing_happening_returns_nothing():
    assert ss.best_windows([0.0] * 60, cuts=[], window=10) == []
    assert ss.second_scores([], [], 0) == [0.0]


def test_duration_parsing():
    assert ss.parse_duration("40s") == 40 and ss.parse_duration("2m") == 120
    assert ss.parse_duration("0:01:30") == 90 and ss.parse_duration("90") == 90
    with pytest.raises(ValueError):
        ss.parse_duration("a while")


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
def test_real_video_action_burst_is_found(tmp_path, capsys):
    pytest.importorskip("scenedetect")
    colors = ["red", "blue", "green", "yellow", "white", "purple", "orange", "cyan", "magenta", "lime"]
    burst = "".join(f"color=c={c}:s=160x90:d=1[c{i}];" for i, c in enumerate(colors))
    graph = (f"color=c=black:s=160x90:d=10[a];{burst}color=c=gray:s=160x90:d=10[z];"
             f"[a]{''.join(f'[c{i}]' for i in range(10))}[z]concat=n=12:v=1:a=0[v];"
             "sine=f=300:d=10,volume=0.01[q1];sine=f=600:d=10,volume=0.9[l];sine=f=300:d=10,volume=0.01[q2];"
             "[q1][l][q2]concat=n=3:v=0:a=1[aud]")
    video = tmp_path / "action.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph,
                    "-map", "[v]", "-map", "[aud]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(video)], check=True)
    assert ss.main([str(video), "--window", "10s", "--top", "1"]) == 0
    start, end = config._parse_section(capsys.readouterr().out.split()[0])
    assert 8 <= start <= 12 and end - start == 10   # the 10-20 s burst of cuts + loud audio
