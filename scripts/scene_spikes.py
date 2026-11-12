#!/usr/bin/env python3
"""Find the most action-packed stretches of a video (no chat needed), for --source-section.

For footage where the transcript says little: trailers, gameplay without commentary, Kick or
YouTube streams (no chat replay). Scores each second by scene cuts (PySceneDetect) and loudness
spikes (ffmpeg), then prints the best non-overlapping windows, like scripts/chat_spikes.py:

    python scripts/scene_spikes.py trailer.mp4 --window 40s --top 3
    0:00:42-0:01:22  score 61.5  peak 0:01:05  cuts 18

Needs ffmpeg on PATH and `pip install scenedetect` (OpenCV is already a clipper dependency).
"""

import argparse
import re
import statistics
import subprocess
import sys

STEP = 1  # seconds; window start resolution
CUT_WEIGHT = 1.0
LOUD_DB_PER_POINT = 6.0  # every 6 dB above the video's median loudness scores 1 point


def parse_duration(text):
    """'40s', '2m', '1h', '90' or '0:01:30' -> seconds."""
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
    raise ValueError(f"bad duration {text!r}; use 40s, 2m, 1h or H:MM:SS")


def hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def scene_cuts(path, threshold=27.0):
    """Seconds at which a new scene starts (the first scene's start is not a cut)."""
    from scenedetect import ContentDetector, detect
    starts = [start for start, _ in detect(path, ContentDetector(threshold=threshold))][1:]
    return [s.seconds if hasattr(s, "seconds") else s.get_seconds() for s in starts]  # 0.7 renamed it


def loudness_per_second(path):
    """RMS level in dB for each second of audio (empty list if the video has no audio)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
           "-af", "aresample=8000,asetnsamples=8000,astats=metadata=1:reset=1,"
                  "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
           "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    levels = []
    for value in re.findall(r"RMS_level=(-?[\d.]+|-inf)", out):
        levels.append(-120.0 if value == "-inf" else float(value))
    return levels


def duration_of(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def second_scores(cuts, levels, length):
    """Score per second: CUT_WEIGHT per scene cut plus loudness above the median, in LOUD_DB_PER_POINT steps."""
    n = int(length) + 1
    scores = [0.0] * n
    for t in cuts:
        if 0 <= t < n:
            scores[int(t)] += CUT_WEIGHT
    audible = [db for db in levels if db > -90]  # at or below -90 dB is silence, not a loudness spike
    if audible:
        median = statistics.median(audible)
        for i, db in enumerate(levels[:n]):
            if db > -90:
                scores[i] += max(0.0, (db - median) / LOUD_DB_PER_POINT)
    return scores


def best_windows(scores, cuts, window, top=3):
    """Top non-overlapping windows as dicts: start, end, score, peak, cuts."""
    length = len(scores) - 1
    if length <= 0:
        return []
    window = min(window, float(length))
    per = max(1, int(round(window)))
    prefix = [0.0]
    for s in scores:
        prefix.append(prefix[-1] + s)
    last_start = max(0, int(length - window))
    candidates = [(prefix[min(i + per, len(scores))] - prefix[i], i) for i in range(0, last_start + 1, STEP)]
    picked = []
    for score, start in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if score <= 0 or len(picked) >= top:
            break
        end = start + window
        if any(start < p["end"] and p["start"] < end for p in picked):
            continue
        j = min(start + per, len(scores))
        peak = max(range(start, j), key=lambda k: scores[k])
        n_cuts = sum(1 for t in cuts if start <= t < end)
        picked.append({"start": float(start), "end": float(end), "score": round(score, 1), "peak": float(peak), "cuts": n_cuts})
    return sorted(picked, key=lambda p: -p["score"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", help="local video file")
    p.add_argument("--window", default="40s", help="section length: 40s, 2m, 30m or H:MM:SS (default 40s)")
    p.add_argument("--top", type=int, default=3, help="how many non-overlapping sections (default 3)")
    p.add_argument("--threshold", type=float, default=27.0, help="PySceneDetect ContentDetector threshold")
    args = p.parse_args(argv)
    try:
        window = parse_duration(args.window)
        length = duration_of(args.video)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))
    if window <= 0 or args.top < 1:
        p.error("--window and --top must be positive")
    cuts = scene_cuts(args.video, args.threshold)
    levels = loudness_per_second(args.video)
    picks = best_windows(second_scores(cuts, levels, length), cuts, window, args.top)
    if not picks:
        print("no scene cuts or loudness changes found", file=sys.stderr)
        return 1
    for s in picks:
        print(f"{hms(s['start'])}-{hms(s['end'])}  score {s['score']}  peak {hms(s['peak'])}  cuts {s['cuts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
