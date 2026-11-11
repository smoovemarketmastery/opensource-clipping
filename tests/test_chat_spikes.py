"""scripts/chat_spikes.py: pick --source-section windows from a Twitch chat JSON."""

import importlib.util
import json
from pathlib import Path

import pytest

from clipping import config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("chat_spikes", ROOT / "scripts" / "chat_spikes.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _chat(tmp_path, messages, length=None, name="chat.json"):
    root = {"comments": [{"content_offset_seconds": t, "message": {"body": b}} for t, b in messages]}
    if length is not None:
        root["video"] = {"length": length}
    path = tmp_path / name
    path.write_text(json.dumps(root), encoding="utf-8")
    return path


def _steady(length, every=20, body="hello"):
    return [(float(t), body) for t in range(0, int(length), every)]


def test_spike_lands_in_the_best_window_and_parses_as_a_section(tmp_path):
    length = 4 * 3600
    spike = [(3 * 3600 + 10 * 60 + i * 0.5, "KEKW LMAO") for i in range(600)]  # 5 min of chaos at 3:10
    messages, loaded_length = cs.load_chat(_chat(tmp_path, _steady(length) + spike, length))
    assert loaded_length == length
    best = cs.best_sections(messages, length, window=1800, top=1)[0]
    assert best["start"] <= 3 * 3600 + 10 * 60 and best["end"] >= 3 * 3600 + 15 * 60
    assert best["end"] - best["start"] == 1800
    assert 3 * 3600 + 10 * 60 <= best["peak"] <= 3 * 3600 + 15 * 60
    assert best["hype"][0][0] in ("KEKW", "LMAO")
    start, end = config._parse_section(f"{cs.hms(best['start'])}-{cs.hms(best['end'])}")
    assert (start, end) == (best["start"], best["end"])


def test_windows_do_not_overlap_and_are_ranked():
    length = 3 * 3600
    a = [(600 + i, "W") for i in range(300)]          # big spike near 0:10
    b = [(2 * 3600 + i, "lol") for i in range(120)]    # smaller spike near 2:00
    secs = cs.best_sections(_steady(length) + a + b, length, window=1800, top=3)
    assert len(secs) == 3 and secs[0]["score"] >= secs[1]["score"] >= secs[2]["score"]
    for x in secs:
        for y in secs:
            if x is not y:
                assert x["end"] <= y["start"] or y["end"] <= x["start"]
    assert secs[0]["start"] <= 600 < secs[0]["end"]
    assert secs[1]["start"] <= 2 * 3600 < secs[1]["end"]


def test_hype_messages_outweigh_plain_ones():
    plain = [(300 + i, "hello") for i in range(100)]
    hype = [(2400 + i, "OMEGALUL") for i in range(100)]
    best = cs.best_sections(plain + hype, 3600, window=600, top=1)[0]
    assert best["start"] <= 2400 < best["end"]


def test_short_vod_returns_the_whole_vod():
    secs = cs.best_sections([(10.0, "hi"), (50.0, "W")], 120, window=1800, top=3)
    assert len(secs) == 1 and (secs[0]["start"], secs[0]["end"]) == (0.0, 120.0)


def test_missing_length_falls_back_to_last_message(tmp_path):
    messages, length = cs.load_chat(_chat(tmp_path, [(5.0, "a"), (725.5, "b")]))
    assert length == 725.5 and len(messages) == 2


def test_bad_input_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"messages": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'comments' list"):
        cs.load_chat(bad)
    assert cs.best_sections([], 3600) == []


def test_duration_parsing():
    assert cs.parse_duration("30m") == 1800 and cs.parse_duration("1h") == 3600
    assert cs.parse_duration("90s") == 90 and cs.parse_duration("0:30:00") == 1800
    assert cs.parse_duration("1800") == 1800
    with pytest.raises(ValueError):
        cs.parse_duration("half an hour")


def test_cli_prints_sections_first(tmp_path, capsys):
    path = _chat(tmp_path, _steady(7200) + [(3600 + i, "KEKW") for i in range(200)], 7200)
    assert cs.main([str(path), "--window", "20m", "--top", "2"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    start, end = config._parse_section(lines[0].split()[0])
    assert end - start == 1200 and start <= 3600 < end
