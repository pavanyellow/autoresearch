#!/usr/bin/env python3
"""
Extract bilingual STT events from call logs for offline replay/analysis.

Downloads call logs from S3 and extracts all STT events (interims + finals
from both EN and ES models), routing decisions, and tool calls into a
structured JSON file that can be replayed against different routing strategies.

Usage:
    # Extract events for canary calls since a timestamp
    python extract_stt_events.py \
        --since 2026-03-19T15:40:00+00:00 \
        --version taylor-fresh-canary \
        --max-per-group 10 \
        --output bilingual_stt_events.json

    # Append new calls to an existing file (skips already-processed call IDs)
    python extract_stt_events.py \
        --since 2026-03-19T15:40:00+00:00 \
        --version taylor-fresh-canary \
        --max-per-group 10 \
        --output bilingual_stt_events.json \
        --append
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dfdvsmtmyhsqslvcvpcl.supabase.co")
SUPABASE_KEY = os.environ.get(
    "supabase_key",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmZHZzbXRteWhzcXNsdmN2cGNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTg3NTQxMTAsImV4cCI6MjAzNDMzMDExMH0.-talIg9F__vDpC7cDfz5MTsYsj7cZqCaLAeVeflNB-Y",
)
TABLE = "call_westlake"


def query_calls(since: str, version: str) -> list[dict]:
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE}",
        headers=headers,
        params={
            "select": "id,result,duration,language,language_switched,creation_date",
            "version": f"eq.{version}",
            "creation_date": f"gte.{since}",
            "account_number": "neq.RUOK_TEST_ACCOUNT",
            "order": "creation_date.desc",
            "limit": "5000",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def sample_calls(calls: list[dict], max_per_group: int) -> list[dict]:
    groups = defaultdict(list)
    for c in calls:
        key = (c["result"], c.get("language", "en"))
        groups[key].append(c)

    random.seed(42)
    sampled = []
    for calls_in_group in groups.values():
        n = min(max_per_group, len(calls_in_group))
        sampled.extend(random.sample(calls_in_group, n))
    return sampled


def download_log(call_id: str) -> bool:
    log_path = f"s3_logs/{call_id}.log"
    if os.path.exists(log_path):
        return True
    result = subprocess.run(
        [sys.executable, "-m", "utils.s3_call_logs", call_id],
        capture_output=True, text=True, timeout=30,
    )
    return "downloaded" in result.stdout.lower()


def extract_events(call_id: str, meta: dict) -> dict:
    log_path = f"s3_logs/{call_id}.log"
    with open(log_path) as f:
        lines = f.readlines()

    call_events = {
        "call_id": call_id,
        "result": meta["result"],
        "duration": meta["duration"],
        "language": meta.get("language", "en"),
        "language_switched": meta.get("language_switched"),
        "stt_events": [],
        "routing_events": [],
        "tool_events": [],
    }

    for i, line in enumerate(lines):
        ts_match = re.search(r"\[(\d+\.\d+)\]", line)
        if not ts_match:
            continue
        ts = float(ts_match.group(1))

        # NemotronSTT events
        if "NemotronSTT" in line and ("INTERIM" in line or "FINAL" in line):
            conf_match = re.search(r"conf=(\d+\.\d+)", line)
            text_match = re.search(r"text='(.+)'", line)
            audio_ts_match = re.search(r"audio_ts=(\d+\.\d+)s", line)
            ev_type = "FINAL" if "FINAL" in line else "INTERIM"
            conf = float(conf_match.group(1)) if conf_match else None
            text = text_match.group(1) if text_match else ""
            audio_ts = float(audio_ts_match.group(1)) if audio_ts_match else None

            stream = None
            for check_line in lines[i + 1 : i + 5]:
                if "transcript_language=en" in check_line:
                    stream = 0
                    break
                elif "transcript_language=es" in check_line:
                    stream = 1
                    break
                elif "Dropped non-base interim EN" in check_line or "INTERIM EN" in check_line or "FINAL EN" in check_line:
                    stream = 0
                    break
                elif "Dropped non-base interim ES" in check_line or "INTERIM ES" in check_line or "FINAL ES" in check_line:
                    stream = 1
                    break

            call_events["stt_events"].append({
                "ts": ts,
                "audio_ts": audio_ts,
                "type": ev_type,
                "stream": stream,
                "lang": "en" if stream == 0 else ("es" if stream == 1 else "?"),
                "conf": conf,
                "text": text,
            })

        # Routing decisions
        if "BilingualSTT" in line and any(
            kw in line
            for kw in [
                "Base language set", "Dropped non-base", "LANGUAGE SWITCH",
                "Rejected non-base", "Blocked cross-stream", "Blocking non-primary",
                "Updated", "Reset", "FINAL EN", "FINAL ES", "INTERIM EN", "INTERIM ES",
            ]
        ):
            call_events["routing_events"].append({
                "ts": ts,
                "event": line.strip().split("} ")[-1] if "} " in line else line.strip(),
            })

        # Tool calls
        if "change_language" in line and ("tool called" in line or "tool_name" in line):
            call_events["tool_events"].append({"ts": ts, "event": "change_language"})

        # What LLM received
        if "user_input_transcribed" in line:
            transcript_match = re.search(r"transcript='(.+?)' is_final=(\w+)", line)
            if transcript_match:
                call_events["routing_events"].append({
                    "ts": ts,
                    "event": f"LLM_RECEIVED {'FINAL' if transcript_match.group(2) == 'True' else 'interim'}: \"{transcript_match.group(1)[:80]}\"",
                })

    return call_events


def main():
    parser = argparse.ArgumentParser(description="Extract bilingual STT events from call logs")
    parser.add_argument("--since", required=True, help="ISO timestamp (e.g. 2026-03-19T15:40:00+00:00)")
    parser.add_argument("--version", default="taylor-fresh-canary", help="Call version filter")
    parser.add_argument("--max-per-group", type=int, default=10, help="Max calls per (dispo, language) group")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--append", action="store_true", help="Append to existing file, skip processed call IDs")
    args = parser.parse_args()

    # Load existing if appending
    existing = []
    existing_ids = set()
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        existing_ids = {c["call_id"] for c in existing}
        print(f"Existing calls: {len(existing_ids)}")

    # Query and sample
    all_calls = query_calls(args.since, args.version)
    print(f"Total calls available: {len(all_calls)}")
    sampled = sample_calls(all_calls, args.max_per_group)
    new_calls = [c for c in sampled if c["id"] not in existing_ids]
    print(f"New calls to process: {len(new_calls)}")

    if not new_calls:
        print("No new calls.")
        return

    # Download and extract
    new_events = []
    for i, c in enumerate(new_calls):
        if download_log(c["id"]):
            events = extract_events(c["id"], c)
            new_events.append(events)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(new_calls)} processed")

    # Merge and save
    all_events = existing + new_events
    with open(args.output, "w") as f:
        json.dump(all_events, f, indent=2)

    # Summary
    dispo_counts = Counter(c["result"] for c in all_events)
    lang_counts = Counter(c["language"] for c in all_events)
    total_stt = sum(len(c["stt_events"]) for c in all_events)

    print(f"\nTotal: {len(all_events)} calls, {total_stt} STT events")
    print(f"Dispos: {dict(dispo_counts)}")
    print(f"Languages: {dict(lang_counts)}")
    print(f"Switched: {sum(1 for c in all_events if c['language_switched'])}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
