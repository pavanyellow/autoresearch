"""
Root runner for bilingual STT forwarding replay.

The default replay path reconstructs the logged prod-forwarded `LLM_RECEIVED`
events so the root baseline matches the black-box evaluator. This gives the
next iteration loop a correct baseline before replacing the replay function
with an editable routing/session simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare import (
    CallInput,
    build_report,
    evaluate_replay,
    extract_logged_forwarded_events,
    load_calls,
)


def replay_call(call: CallInput):
    return extract_logged_forwarded_events(call)


def _format_metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _select_calls(calls: list[CallInput], call_ids: str | None) -> list[CallInput]:
    if not call_ids:
        return calls
    targets = set(call_ids.split(","))
    return [call for call in calls if call.call_id in targets or any(call.call_id.startswith(target) for target in targets)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay bilingual forwarded events against the oracle.")
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--oracle-dir", default=None)
    parser.add_argument("--call-ids", default=None)
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    calls = load_calls(args.events_path, args.oracle_dir)
    calls = _select_calls(calls, args.call_ids)
    metrics, scores = evaluate_replay(replay_call, calls)

    print("---")
    print(f"text_avg_switch_delay: {_format_metric(metrics['text_based']['avg_switch_delay'])}")
    print(
        "text_false_es_rate: "
        f"{metrics['text_based']['false_es_rate']:.2f}% "
        f"({metrics['text_based']['false_es_events']}/{metrics['text_based']['false_es_denominator']})"
    )
    print(f"text_zero_delay_calls: {_format_metric(metrics['text_based']['zero_delay_calls'])}")
    print(f"stream_avg_switch_delay: {_format_metric(metrics['stream_based']['avg_switch_delay'])}")
    print(
        "stream_false_es_rate: "
        f"{metrics['stream_based']['false_es_rate']:.2f}% "
        f"({metrics['stream_based']['false_es_events']}/{metrics['stream_based']['false_es_denominator']})"
    )
    print(f"stream_zero_delay_calls: {_format_metric(metrics['stream_based']['zero_delay_calls'])}")
    print(f"processed_calls: {_format_metric(metrics['processed'])}")
    print(f"switched_calls: {_format_metric(metrics['switched_calls'])}")
    print(f"errors: {_format_metric(metrics['errors'])}")

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = build_report(
            metrics=metrics,
            scores=scores,
            events_path=args.events_path or "default",
            oracle_dir=args.oracle_dir or "default",
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
