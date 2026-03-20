#!/usr/bin/env python3
"""
Generate side-by-side spotcheck txt comparing BilingualSTT vs Oracle events.

Usage:
    python scripts/eval_bilingual_stt/spotcheck.py \
        --events bilingual_stt_events.json \
        --oracle-dir scripts/eval_bilingual_stt/eval_results/oracle/ \
        --call-ids 5c7c8787,de68583a \
        --output /Users/pavan/Downloads/spotcheck.txt

    # All calls with oracle data
    python scripts/eval_bilingual_stt/spotcheck.py \
        --events bilingual_stt_events.json \
        --oracle-dir scripts/eval_bilingual_stt/eval_results/oracle/
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def generate_spotcheck(events_path: str, oracle_dir: str, call_ids: list[str] | None, output: str | None):
    with open(events_path) as f:
        data = json.load(f)

    # Filter calls
    if call_ids:
        # Support partial IDs
        data = [c for c in data if any(c["call_id"].startswith(prefix) for prefix in call_ids)]

    # Only include calls that have oracle data
    data = [c for c in data if os.path.exists(os.path.join(oracle_dir, f"{c['call_id']}.json"))]

    if not data:
        print("No calls found with oracle data.")
        return

    lines = []
    for call in data:
        cid = call["call_id"]
        with open(os.path.join(oracle_dir, f"{cid}.json")) as f:
            oracle = json.load(f)

        events = []
        for stt in call["stt_events"]:
            if stt["type"] == "FINAL" and stt.get("audio_ts"):
                conf = stt.get("conf", 0)
                events.append((stt["audio_ts"], "BILINGUAL", f'lang={stt["lang"]}  conf={conf:.3f}  "{stt["text"]}"'))

        for u in oracle:
            wl = u["word_langs"]
            dg = wl.get("deepgram", {})
            ft = wl.get("fasttext", "?")
            events.append((u["start"], "ORACLE  ", f'lang={u["lang"]:5s}  dg={dg} ft={ft}  "{u["text"][:70]}"'))

        events.sort(key=lambda x: x[0])

        lines.append(f"=== {cid} ===")
        lines.append(f"Call language: {call['language']}  |  switched: {call.get('language_switched', False)}")
        lines.append("")
        lines.append(f"{'TIME':>7s}  {'SOURCE':8s}  DETAIL")
        lines.append("-" * 115)
        for ts, src, detail in events:
            lines.append(f"{ts:7.1f}s  {src}  {detail}")
        lines.append("")
        lines.append("")

    text = "\n".join(lines)
    if output:
        with open(output, "w") as f:
            f.write(text)
        print(f"Written to {output} ({len(data)} calls)")
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description="Generate spotcheck txt for BilingualSTT vs Oracle")
    parser.add_argument("--events", required=True, help="Path to bilingual_stt_events.json")
    parser.add_argument("--oracle-dir", required=True, help="Directory with oracle JSON files")
    parser.add_argument("--call-ids", help="Comma-separated call ID prefixes (default: all with oracle data)")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    args = parser.parse_args()

    call_ids = args.call_ids.split(",") if args.call_ids else None
    generate_spotcheck(args.events, args.oracle_dir, call_ids, args.output)


if __name__ == "__main__":
    main()
