"""Gemini retry: transient network errors must retry the main model, not jump to the fallback."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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
