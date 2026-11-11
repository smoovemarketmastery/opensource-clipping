#!/usr/bin/env python3
"""Find the busiest stretches of a Twitch VOD from its chat, for --source-section.

Input is a chat JSON from TwitchDownloaderCLI (https://github.com/lay295/TwitchDownloader):

    TwitchDownloaderCLI chatdownload --id <VOD id or URL> -o chat.json --collision Overwrite
    python scripts/chat_spikes.py chat.json --window 30m --top 3

Each output line starts with a section ready for --source-section, e.g.
    2:13:00-2:43:00  score 3120  peak 2:27:40  hype: LMAO x84, KEKW x40

Scoring: chat is bucketed into 10-second slots; every message scores 1, plus 1 if it has a
hype reaction (LUL, KEKW, LMAO, W, POG, "clip it", ...). Windows are ranked by total score and
picked greedily without overlap. Stdlib only, so it runs in any environment.
"""

import argparse
import json
import re
import sys
from collections import Counter

BUCKET = 10  # seconds
STEP = 60    # window start resolution, seconds
# ponytail: raw message counts favor late-stream peaks when the audience grows; divide by a
# rolling baseline if that starts picking slow-but-crowded stretches.
HYPE = re.compile(
    r"\b(?:LULW?|KEKW?|OMEGALUL|LMAO+|LMFAO+|LOL+|ICANT|W+|POG(?:GERS|CHAMP|U)?|POGGERS|PogChamp|"
    r"monkaS|monkaW|HOLY|NO WAY|WTF|CLIP(?: IT|PED)?|SHEESH|DEAD)\b|\?{3,}|!{3,}|💀|😂|🤣",
    re.I,
)


def parse_duration(text):
    """'30m', '1h', '90s', '1800' or '0:30:00' -> seconds."""
    text = text.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([hms]?)", text)
    if m:
        return float(m.group(1)) * {"h": 3600, "m": 60, "s": 1, "": 1}[m.group(2)]
    parts = text.split(":")
    if 1 < len(parts) <= 3 and all(p.replace(".", "", 1).isdigit() for p in parts):
        total = 0.0
        for p in parts:
            total = total * 60 + float(p)
        return total
    raise ValueError(f"bad duration {text!r}; use 30m, 1h, 90s or H:MM:SS")


def hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def load_chat(path):
    """(messages [(offset_seconds, body)], length_seconds) from a TwitchDownloader chat JSON."""
    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    comments = root.get("comments") if isinstance(root, dict) else None
    if not isinstance(comments, list):
        raise ValueError(f"{path}: no 'comments' list; is this a TwitchDownloader chat JSON?")
    messages = []
    for c in comments:
        try:
            messages.append((float(c["content_offset_seconds"]), str((c.get("message") or {}).get("body") or "")))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    video = root.get("video") or {}
    length = video.get("length") if isinstance(video, dict) else None
    if not isinstance(length, (int, float)) or length <= 0:
        length = max((t for t, _ in messages), default=0.0)
    return messages, float(length)


def score_buckets(messages, length):
    n = int(length // BUCKET) + 1
    scores = [0] * n
    hype_words = [Counter() for _ in range(n)]
    for t, body in messages:
        if not 0 <= t <= length:
            continue
        i = int(t // BUCKET)
        hits = HYPE.findall(body)
        scores[i] += 1 + (1 if hits else 0)
        for h in hits:
            hype_words[i][h.upper()] += 1
    return scores, hype_words


def best_sections(messages, length, window=1800.0, top=3):
    """Top non-overlapping windows as dicts: start, end, score, peak, hype [(word, count)]."""
    if length <= 0 or not messages:
        return []
    scores, hype_words = score_buckets(messages, length)
    window = min(window, length)
    per = max(1, int(round(window / BUCKET)))
    prefix = [0]
    for s in scores:
        prefix.append(prefix[-1] + s)
    candidates = []
    last_start = max(0.0, length - window)
    start = 0.0
    while True:
        i = int(start // BUCKET)
        candidates.append((prefix[min(i + per, len(scores))] - prefix[i], start))
        if start >= last_start:
            break
        start = min(start + STEP, last_start)
    picked = []
    for score, start in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if score <= 0 or len(picked) >= top:
            break
        end = start + window
        if any(start < p["end"] and p["start"] < end for p in picked):
            continue
        i, j = int(start // BUCKET), min(int(start // BUCKET) + per, len(scores))
        peak_i = max(range(i, j), key=lambda k: scores[k])
        hype = sum((hype_words[k] for k in range(i, j)), Counter()).most_common(3)
        picked.append({"start": start, "end": end, "score": score, "peak": peak_i * BUCKET, "hype": hype})
    return sorted(picked, key=lambda p: -p["score"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("chat_json", help="chat JSON from TwitchDownloaderCLI chatdownload")
    p.add_argument("--window", default="30m", help="section length: 30m, 1h, 90s or H:MM:SS (default 30m)")
    p.add_argument("--top", type=int, default=3, help="how many non-overlapping sections (default 3)")
    args = p.parse_args(argv)
    try:
        window = parse_duration(args.window)
        messages, length = load_chat(args.chat_json)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if window <= 0 or args.top < 1:
        p.error("--window and --top must be positive")
    sections = best_sections(messages, length, window, args.top)
    if not sections:
        print("no chat messages found", file=sys.stderr)
        return 1
    for s in sections:
        hype = ", ".join(f"{w} x{n}" for w, n in s["hype"]) or "-"
        print(f"{hms(s['start'])}-{hms(s['end'])}  score {s['score']}  peak {hms(s['peak'])}  hype: {hype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
