"""Default Gemini models: one source of truth, and no retired model names in code."""

from pathlib import Path
import re

from clipping import config

ROOT = Path(__file__).resolve().parents[1]
RETIRED = re.compile(r"gemini-(?:1\.5|2\.0|2\.5)-flash")  # 404 or limited to prior users


def test_fallback_default_is_not_a_retired_model():
    assert config.GEMINI_FALLBACK_MODEL == "gemini-3.5-flash-lite"
    assert config.build_config(["--url", "https://example.test/v"]).gemini_fallback_model == "gemini-3.5-flash-lite"


def test_no_hardcoded_model_names_outside_config():
    offenders = []
    for path in [*(ROOT / "clipping").rglob("*.py"), *(ROOT / "web").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if RETIRED.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_voiceover_reads_models_from_config():
    text = (ROOT / "clipping" / "voiceover.py").read_text(encoding="utf-8")
    assert '"gemini_model", GEMINI_MODEL' in text and '"gemini_fallback_model", GEMINI_FALLBACK_MODEL' in text
