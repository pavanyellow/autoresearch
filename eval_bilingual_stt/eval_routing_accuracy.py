#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluate BilingualSTT routing accuracy against Nova-3 multi oracle.

Takes bilingual_stt_events.json + oracle transcription JSONs as inputs.
No downloading or transcribing — run transcribe_oracle.py first.

Metrics:
- switch_delay: bilingual events before first ES forward after oracle detects Spanish
- false_es_events: spurious Spanish forwarding during oracle-EN regions

Usage:
    python scripts/eval_bilingual_stt/eval_routing_accuracy.py \
        --events scripts/eval_bilingual_stt/bilingual_stt_events.json \
        --oracle-dir scripts/eval_bilingual_stt/eval_results/oracle/ \
        --output-dir scripts/eval_bilingual_stt/eval_results/
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class OracleUtterance:
    start: float
    end: float
    text: str
    lang: str
    word_langs: dict


@dataclass
class BilingualEvent:
    ts: float
    text: str
    lang: str  # "en", "es", or "?"
    is_final: bool


@dataclass
class CallResult:
    call_id: str
    call_language: str
    language_switched: bool
    first_oracle_es_time: float | None = None
    first_oracle_es_text: str | None = None
    oracle_utterance_count: int = 0
    mixed_utterances: int = 0
    switch_delay: int | None = None
    switch_delay_events: list[dict] = field(default_factory=list)
    false_es_events: int = 0
    false_es_finals: int = 0
    false_es_detail: list[dict] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Extract bilingual LLM_RECEIVED events
# ---------------------------------------------------------------------------


def extract_bilingual_events(call: dict) -> list[BilingualEvent]:
    """Extract all LLM_RECEIVED events (interim + final) with inferred language."""
    events = []
    routing_events = call.get("routing_events", [])

    for i, rev in enumerate(routing_events):
        event_str = rev.get("event", "")

        if "LLM_RECEIVED FINAL:" in event_str:
            text = event_str.split('FINAL: "', 1)[-1].rstrip('"')
            is_final = True
        elif "LLM_RECEIVED interim:" in event_str:
            text = event_str.split('interim: "', 1)[-1].rstrip('"')
            is_final = False
        else:
            continue

        ts = rev.get("ts", 0)
        lang = _infer_lang_from_context(routing_events, ts)
        events.append(BilingualEvent(ts=ts, text=text, lang=lang, is_final=is_final))

    return events


def _infer_lang_from_context(routing_events: list[dict], ts: float) -> str:
    """Infer language of an LLM_RECEIVED event from surrounding routing events."""
    for rev in routing_events:
        rev_ts = rev.get("ts", 0)
        if abs(rev_ts - ts) > 0.5:
            continue
        ev = rev.get("event", "")
        if "FINAL EN" in ev or "INTERIM EN" in ev:
            return "en"
        if "FINAL ES" in ev or "INTERIM ES" in ev:
            return "es"
    return "?"


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------


def compute_switch_metrics(
    oracle_utts: list[OracleUtterance], bilingual_events: list[BilingualEvent], call: dict
) -> CallResult:
    """Compute switch delay and false ES forward metrics."""
    result = CallResult(
        call_id=call["call_id"],
        call_language=call.get("language", "en"),
        language_switched=call.get("language_switched", False),
        oracle_utterance_count=len(oracle_utts),
        mixed_utterances=sum(1 for u in oracle_utts if u.lang == "mixed"),
    )

    # Map oracle audio timestamps to wall-clock
    stt_events = call.get("stt_events", [])
    offset = _estimate_wallclock_offset(stt_events)
    if offset is None:
        result.error = "no_timestamp_mapping"
        return result

    # Find first non-mixed oracle ES utterance
    first_oracle_es = next((u for u in oracle_utts if u.lang == "es"), None)

    # False ES forwards during oracle-EN region
    en_region_end = (first_oracle_es.start + offset) if first_oracle_es else float("inf")

    for ev in bilingual_events:
        if ev.ts >= en_region_end:
            break
        if ev.lang == "es":
            result.false_es_events += 1
            if ev.is_final:
                result.false_es_finals += 1
            if len(result.false_es_detail) < 10:
                result.false_es_detail.append({"ts": ev.ts, "text": ev.text, "is_final": ev.is_final})

    if first_oracle_es is None:
        return result

    result.first_oracle_es_time = first_oracle_es.start
    result.first_oracle_es_text = first_oracle_es.text

    # Switch delay: bilingual events after first oracle ES before first bilingual ES
    oracle_es_wallclock = first_oracle_es.start + offset
    delay_events = []
    for ev in bilingual_events:
        if ev.ts < oracle_es_wallclock:
            continue
        if ev.lang == "es":
            break
        if ev.lang == "en":
            delay_events.append({"ts": ev.ts, "text": ev.text, "is_final": ev.is_final})

    result.switch_delay = len(delay_events)
    result.switch_delay_events = delay_events

    return result


def _estimate_wallclock_offset(stt_events: list[dict]) -> float | None:
    """Estimate wall-clock offset: wall_ts = audio_ts + offset."""
    for ev in stt_events:
        if ev.get("audio_ts") and ev.get("ts"):
            return ev["ts"] - ev["audio_ts"]
    return None


# ---------------------------------------------------------------------------
# Summary + output
# ---------------------------------------------------------------------------


def print_summary(results: list[CallResult]):
    """Print human-readable summary to stdout."""
    processed = [r for r in results if r.error is None]
    errored = [r for r in results if r.error is not None]
    error_counts = Counter(r.error for r in errored)

    switched = [r for r in processed if r.first_oracle_es_time is not None]
    en_only = [r for r in processed if r.first_oracle_es_time is None]

    print(f"\n{'=' * 60}")
    print(f"  BILINGUAL STT ROUTING EVAL ({len(results)} calls)")
    print(f"{'=' * 60}")
    print(f"Calls processed:     {len(processed):>4} / {len(results)}")
    if errored:
        print(f"Errors:              {len(errored):>4}  {dict(error_counts)}")

    if switched:
        delays = [r.switch_delay for r in switched if r.switch_delay is not None]
        zero_delay = sum(1 for d in delays if d == 0)
        print(f"\n--- Spanish-switched calls ({len(switched)}) ---")
        print(f"Switch delay (bilingual events before first ES forward):")
        print(f"  delay=0 (instant):  {zero_delay:>3} calls")
        for d in sorted(set(delays)):
            if d > 0:
                count = sum(1 for x in delays if x == d)
                print(f"  delay={d}:            {count:>3} calls")
        if delays:
            print(f"  avg delay:          {sum(delays) / len(delays):.1f} events")

        delayed = [r for r in switched if r.switch_delay and r.switch_delay > 0]
        if delayed:
            print(f"\n  Calls with delay > 0:")
            for r in sorted(delayed, key=lambda r: r.switch_delay, reverse=True):
                print(
                    f'    {r.call_id}  delay={r.switch_delay}  first_es_oracle={r.first_oracle_es_time:.1f}s "{r.first_oracle_es_text[:50]}"'
                )
                for ev in r.switch_delay_events:
                    tag = "FINAL" if ev["is_final"] else "interim"
                    print(f'      [{tag}] "{ev["text"][:60]}"')

    total_false_es = sum(r.false_es_events for r in processed)
    total_false_es_finals = sum(r.false_es_finals for r in processed)
    calls_with_false_es = [r for r in processed if r.false_es_events > 0]

    print(f"\n--- False ES forwards (during oracle-EN regions) ---")
    print(f"Calls affected:      {len(calls_with_false_es):>3} / {len(processed)}")
    print(f"Total events:        {total_false_es:>3}  ({total_false_es_finals} finals)")
    if calls_with_false_es:
        for r in sorted(calls_with_false_es, key=lambda r: r.false_es_events, reverse=True):
            print(f"  {r.call_id}  events={r.false_es_events} finals={r.false_es_finals}")
            for ev in r.false_es_detail:
                tag = "FINAL" if ev["is_final"] else "interim"
                print(f'    [{tag}] "{ev["text"][:60]}"')

    total_mixed = sum(r.mixed_utterances for r in processed)
    total_utts = sum(r.oracle_utterance_count for r in processed)
    if total_mixed > 0:
        print(f"\nMixed utterances:    {total_mixed:>4} / {total_utts}  (excluded)")


def write_results(results: list[CallResult], output_dir: str):
    """Write aggregate and per-call results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    processed = [r for r in results if r.error is None]
    switched = [r for r in processed if r.first_oracle_es_time is not None]
    en_only = [r for r in processed if r.first_oracle_es_time is None]
    delays = [r.switch_delay for r in switched if r.switch_delay is not None]

    aggregate = {
        "total_calls": len(results),
        "processed": len(processed),
        "errors": len(results) - len(processed),
        "error_breakdown": dict(Counter(r.error for r in results if r.error)),
        "switched_calls": len(switched),
        "avg_switch_delay": round(sum(delays) / len(delays), 1) if delays else None,
        "zero_delay_calls": sum(1 for d in delays if d == 0),
        "en_only_calls": len(en_only),
        "false_es_events": sum(r.false_es_events for r in processed),
        "false_es_finals": sum(r.false_es_finals for r in processed),
        "calls_with_false_es": sum(1 for r in processed if r.false_es_events > 0),
    }

    with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    per_call = [asdict(r) for r in results]
    with open(os.path.join(output_dir, "per_call.json"), "w") as f:
        json.dump(per_call, f, indent=2)

    print(f"\nResults written to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate BilingualSTT routing accuracy against Nova-3 multi oracle")
    parser.add_argument("--events", required=True, help="Path to bilingual_stt_events.json")
    parser.add_argument(
        "--oracle-dir", required=True, help="Directory with oracle JSON files from transcribe_oracle.py"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for eval results")
    parser.add_argument("--call-ids", help="Comma-separated call IDs to evaluate (default: all with oracle data)")
    args = parser.parse_args()

    with open(args.events) as f:
        all_calls = json.load(f)

    if args.call_ids:
        target_ids = set(args.call_ids.split(","))
        all_calls = [c for c in all_calls if c["call_id"] in target_ids]

    # Only evaluate calls that have oracle data
    all_calls = [c for c in all_calls if os.path.exists(os.path.join(args.oracle_dir, f"{c['call_id']}.json"))]

    print(f"Evaluating {len(all_calls)} calls")

    results = []
    for call in all_calls:
        call_id = call["call_id"]
        oracle_path = os.path.join(args.oracle_dir, f"{call_id}.json")

        with open(oracle_path) as f:
            oracle_utts = [OracleUtterance(**u) for u in json.load(f)]

        bilingual_events = extract_bilingual_events(call)
        result = compute_switch_metrics(oracle_utts, bilingual_events, call)
        results.append(result)

    print_summary(results)
    write_results(results, args.output_dir)


if __name__ == "__main__":
    main()
