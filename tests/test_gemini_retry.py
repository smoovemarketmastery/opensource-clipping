"""Gemini retry: transient network errors must retry the main model, not jump to the fallback."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=None))
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", SimpleNamespace(download_range_func=None))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=None))
    spec = importlib.util.spec_from_file_location("test_retry_engine", ROOT / "clipping" / "engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)
    return engine


def test_read_timeout_is_retryable(monkeypatch):
    engine = _engine(monkeypatch)
    assert engine._is_retryable(Exception("The read operation timed out"))
    assert not engine._is_retryable(Exception("404 NOT_FOUND model is no longer available"))


def test_timeout_retries_main_model_before_fallback(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        if len(calls) == 1:
            raise Exception("The read operation timed out")
        return SimpleNamespace(text='[{"rank": 1}]')

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    assert engine._generate_json_with_retry(client, "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main", "main"]


def _client(fail_main, calls):
    def generate_content(model, contents, config):
        calls.append(model)
        if model == "main" and fail_main(calls.count("main")):
            raise fail_main.error
        return SimpleNamespace(text='[{"rank": 1}]')
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_persistent_timeout_stops_after_the_cap_then_uses_fallback(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = Exception("The read operation timed out")
    assert engine._generate_json_with_retry(_client(always, calls), "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main"] * engine.TIMEOUT_MAX_ATTEMPTS + ["fallback"]


def test_busy_errors_still_get_more_than_the_timeout_cap(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    first_three = lambda n: n <= 3   # 3 > TIMEOUT_MAX_ATTEMPTS would already have stopped a timeout
    first_three.error = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")
    assert engine._generate_json_with_retry(_client(first_three, calls), "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main"] * 4


BUSY_503 = Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")


def test_busy_streak_switches_to_a_different_fallback_after_the_busy_cap(monkeypatch, capsys):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = BUSY_503
    assert engine._generate_json_with_retry(_client(always, calls), "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main"] * engine.BUSY_MAX_ATTEMPTS + ["fallback"]
    assert "sibuk 4x" in capsys.readouterr().out


def test_rate_limit_429_counts_as_busy(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = type("E", (Exception,), {"code": 429})("RESOURCE_EXHAUSTED")
    engine._generate_json_with_retry(_client(always, calls), "main", "fallback", "x", None)
    assert calls == ["main"] * engine.BUSY_MAX_ATTEMPTS + ["fallback"]


def test_busy_keeps_all_attempts_when_fallback_is_the_same_model_or_missing(monkeypatch):
    engine = _engine(monkeypatch)
    always = lambda n: True
    always.error = BUSY_503
    calls = []
    with pytest.raises(RuntimeError, match="utama & fallback"):   # the same busy model fails again
        engine._generate_json_with_retry(_client(always, calls), "main", "main", "x", None)
    assert calls == ["main"] * (engine.MAX_ATTEMPTS + 1)   # 10 tries + the same-model fallback call
    calls = []
    with pytest.raises(RuntimeError, match="setelah 10 percobaan"):
        engine._generate_json_with_retry(_client(always, calls), "main", None, "x", None)
    assert calls == ["main"] * engine.MAX_ATTEMPTS


@pytest.mark.parametrize("message", [
    "500 INTERNAL. An internal error has occurred.",
    "502 Bad Gateway",
    "504 DEADLINE_EXCEEDED gateway",   # also timeout-class by design: stops at the timeout cap
])
def test_other_5xx_errors_do_not_count_as_busy(monkeypatch, message):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = Exception(message)
    engine._generate_json_with_retry(_client(always, calls), "main", "fallback", "x", None)
    expected = engine.TIMEOUT_MAX_ATTEMPTS if "deadline" in message.lower() else engine.MAX_ATTEMPTS
    assert calls == ["main"] * expected + ["fallback"]


def test_busy_errors_are_counted_across_the_run_not_only_consecutive(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    errors = [BUSY_503, Exception("500 INTERNAL"), BUSY_503, BUSY_503, BUSY_503]

    def generate_content(model, contents, config):
        calls.append(model)
        if model == "main":
            raise errors[min(calls.count("main"), len(errors)) - 1]
        return SimpleNamespace(text='[{"rank": 1}]')

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    assert engine._generate_json_with_retry(client, "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main"] * 5 + ["fallback"]   # the 4th busy error is the 5th attempt


def test_not_found_goes_straight_to_fallback(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = Exception("404 NOT_FOUND. This model is no longer available to new users.")
    assert engine._generate_json_with_retry(_client(always, calls), "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main", "fallback"]


def test_bare_timeout_error_is_retryable(monkeypatch):
    engine = _engine(monkeypatch)
    assert engine._is_retryable(TimeoutError())


def test_bare_timeout_error_also_stops_after_the_cap(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = TimeoutError()
    assert engine._generate_json_with_retry(_client(always, calls), "main", "fallback", "x", None) == [{"rank": 1}]
    assert calls == ["main"] * 3 + ["fallback"]


def test_status_408_counts_as_timeout_and_error_reports_real_attempts(monkeypatch):
    engine = _engine(monkeypatch)
    calls = []
    always = lambda n: True
    always.error = type("E", (Exception,), {"code": 408})("request expired")
    with pytest.raises(RuntimeError, match="setelah 3 percobaan"):
        engine._generate_json_with_retry(_client(always, calls), "main", None, "x", None)
    assert calls == ["main"] * 3
