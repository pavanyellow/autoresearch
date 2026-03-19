"""
Fixed offline evaluation harness for bilingual STT switch-policy experiments.

This module is intentionally read-only during the autoresearch loop.
It loads the production event log, builds deterministic splits, and
evaluates a policy by replaying STT events call by call.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EVENTS_PATH = "/Users/pavan/Downloads/bilingual_stt_events.json"
EVENTS_PATH_ENV = "AUTORESEARCH_EVENTS_PATH"
SPLIT_SEED = 42
SWITCH_ROUTING_PREFIX = "[BilingualSTT] LANGUAGE SWITCH:"
CHANGE_LANGUAGE_EVENT = "change_language"

REQUIRED_CALL_KEYS = {
    "call_id",
    "result",
    "duration",
    "language",
    "language_switched",
    "stt_events",
    "routing_events",
    "tool_events",
}
REQUIRED_STT_EVENT_KEYS = {"ts", "audio_ts", "type", "stream", "lang", "conf", "text"}
REQUIRED_LOG_EVENT_KEYS = {"ts", "event"}


@dataclass(frozen=True)
class STTEvent:
    ts: float
    audio_ts: float
    type: str
    stream: int | None
    lang: str
    conf: float
    text: str


@dataclass(frozen=True)
class LoggedEvent:
    ts: float
    event: str


@dataclass(frozen=True)
class CallRecord:
    call_id: str
    result: str
    duration: int
    language: str
    language_switched: bool
    stt_events: tuple[STTEvent, ...]
    routing_events: tuple[LoggedEvent, ...]
    tool_events: tuple[LoggedEvent, ...]


@dataclass(frozen=True)
class Prediction:
    switch_to_es: bool
    decision_ts: float
    decision_audio_ts: float
    trigger_text: str
    reason: str


@dataclass(frozen=True)
class CallOutcome:
    call_id: str
    split: str
    gold_switch: bool
    gold_switch_ts: float | None
    predicted_switch: bool
    decision_ts: float | None
    decision_audio_ts: float | None
    latency_s: float | None
    result: str
    prediction_reason: str | None
    trigger_text: str | None


def resolve_events_path(events_path: str | None = None) -> Path:
    path = Path(events_path or os.environ.get(EVENTS_PATH_ENV) or DEFAULT_EVENTS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Event log not found at {path}. "
            f"Set {EVENTS_PATH_ENV} or place the file at {DEFAULT_EVENTS_PATH}."
        )
    return path


def _require_keys(obj: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(obj))
    if missing:
        raise ValueError(f"{context} is missing keys: {missing}")


def _load_logged_events(events: list[dict[str, Any]], context: str) -> tuple[LoggedEvent, ...]:
    parsed = []
    for index, event in enumerate(events):
        _require_keys(event, REQUIRED_LOG_EVENT_KEYS, f"{context}[{index}]")
        parsed.append(
            LoggedEvent(
                ts=float(event["ts"]),
                event=str(event["event"]),
            )
        )
    return tuple(sorted(parsed, key=lambda item: item.ts))


def _load_stt_events(events: list[dict[str, Any]], context: str) -> tuple[STTEvent, ...]:
    parsed = []
    for index, event in enumerate(events):
        _require_keys(event, REQUIRED_STT_EVENT_KEYS, f"{context}[{index}]")
        parsed.append(
            STTEvent(
                ts=float(event["ts"]),
                audio_ts=float(event["audio_ts"]),
                type=str(event["type"]).upper(),
                stream=int(event["stream"]) if event["stream"] is not None else None,
                lang=str(event["lang"]),
                conf=float(event["conf"]),
                text=str(event["text"]),
            )
        )
    return tuple(
        sorted(
            parsed,
            key=lambda item: (item.ts, item.audio_ts, -1 if item.stream is None else item.stream, item.type),
        )
    )


def load_calls(events_path: str | None = None) -> list[CallRecord]:
    path = resolve_events_path(events_path)
    with path.open() as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError("Top-level event log must be a list of calls")

    calls = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Call entry {index} must be a dict")
        _require_keys(item, REQUIRED_CALL_KEYS, f"call[{index}]")

        calls.append(
            CallRecord(
                call_id=str(item["call_id"]),
                result=str(item["result"]),
                duration=int(item["duration"]),
                language=str(item["language"]),
                language_switched=bool(item["language_switched"]),
                stt_events=_load_stt_events(item["stt_events"], f"call[{index}].stt_events"),
                routing_events=_load_logged_events(item["routing_events"], f"call[{index}].routing_events"),
                tool_events=_load_logged_events(item["tool_events"], f"call[{index}].tool_events"),
            )
        )
    return calls


def _class_split_counts(total: int) -> tuple[int, int, int]:
    dev = round(total * 0.2)
    test = round(total * 0.2)
    train = total - dev - test
    if min(train, dev, test) < 0:
        raise ValueError(f"Invalid split counts for class size {total}")
    return train, dev, test


def get_splits(calls: list[CallRecord], seed: int = SPLIT_SEED) -> dict[str, list[CallRecord]]:
    switched = sorted((call for call in calls if call.language_switched), key=lambda call: call.call_id)
    non_switched = sorted((call for call in calls if not call.language_switched), key=lambda call: call.call_id)

    randomizer = random.Random(seed)
    randomizer.shuffle(switched)
    randomizer.shuffle(non_switched)

    split_map = {"train": [], "dev": [], "test": []}
    for bucket in (switched, non_switched):
        train_count, dev_count, test_count = _class_split_counts(len(bucket))
        split_map["train"].extend(bucket[:train_count])
        split_map["dev"].extend(bucket[train_count : train_count + dev_count])
        split_map["test"].extend(bucket[train_count + dev_count : train_count + dev_count + test_count])

    for split_name in split_map:
        split_map[split_name] = sorted(split_map[split_name], key=lambda call: call.call_id)

    seen: set[str] = set()
    for split_name, split_calls in split_map.items():
        split_ids = {call.call_id for call in split_calls}
        if seen & split_ids:
            raise ValueError(f"Overlapping calls detected in split {split_name}")
        seen.update(split_ids)

    return split_map


def get_gold_switch_ts(call: CallRecord) -> float | None:
    for event in call.tool_events:
        if event.event == CHANGE_LANGUAGE_EVENT:
            return event.ts
    for event in call.routing_events:
        if SWITCH_ROUTING_PREFIX in event.event:
            return event.ts
    return None


def replay_call(call: CallRecord, policy: Any) -> Prediction | None:
    for event in call.stt_events:
        prediction = policy.observe_event(event)
        if prediction is not None:
            return prediction
    return policy.finalize()


def evaluate_policy(
    policy_factory: Any,
    calls: list[CallRecord],
    split_name: str,
) -> tuple[dict[str, float | int | None], list[CallOutcome]]:
    outcomes: list[CallOutcome] = []
    true_positive = false_positive = true_negative = false_negative = 0
    latencies: list[float] = []

    for call in calls:
        policy = policy_factory()
        prediction = replay_call(call, policy)
        predicted_switch = bool(prediction and prediction.switch_to_es)
        gold_switch = call.language_switched

        if gold_switch and predicted_switch:
            true_positive += 1
        elif gold_switch and not predicted_switch:
            false_negative += 1
        elif not gold_switch and predicted_switch:
            false_positive += 1
        else:
            true_negative += 1

        gold_switch_ts = get_gold_switch_ts(call)
        latency = None
        if predicted_switch and gold_switch_ts is not None and prediction is not None:
            latency = prediction.decision_ts - gold_switch_ts
            latencies.append(latency)

        outcomes.append(
            CallOutcome(
                call_id=call.call_id,
                split=split_name,
                gold_switch=gold_switch,
                gold_switch_ts=gold_switch_ts,
                predicted_switch=predicted_switch,
                decision_ts=prediction.decision_ts if prediction else None,
                decision_audio_ts=prediction.decision_audio_ts if prediction else None,
                latency_s=latency,
                result=call.result,
                prediction_reason=prediction.reason if prediction else None,
                trigger_text=prediction.trigger_text if prediction else None,
            )
        )

    switched_total = true_positive + false_negative
    non_switched_total = true_negative + false_positive
    recall_switched = true_positive / switched_total if switched_total else 0.0
    recall_non_switched = true_negative / non_switched_total if non_switched_total else 0.0
    precision_switched = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    balanced_accuracy = (recall_switched + recall_non_switched) / 2.0
    median_latency = statistics.median(latencies) if latencies else None
    median_abs_latency = statistics.median(abs(value) for value in latencies) if latencies else None

    metrics: dict[str, float | int | None] = {
        f"{split_name}_bal_acc": balanced_accuracy,
        f"{split_name}_precision_switched": precision_switched,
        f"{split_name}_recall_switched": recall_switched,
        f"{split_name}_fp": false_positive,
        f"{split_name}_fn": false_negative,
        f"{split_name}_tp": true_positive,
        f"{split_name}_tn": true_negative,
        f"{split_name}_median_latency_s": median_latency,
        f"{split_name}_median_abs_latency_s": median_abs_latency,
        f"num_{split_name}_calls": len(calls),
    }
    return metrics, outcomes


def summarize_dataset(calls: list[CallRecord]) -> dict[str, Any]:
    switched_calls = sum(call.language_switched for call in calls)
    return {
        "num_calls": len(calls),
        "num_switched_calls": switched_calls,
        "num_non_switched_calls": len(calls) - switched_calls,
        "default_events_path": DEFAULT_EVENTS_PATH,
    }


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate the bilingual STT backtest dataset.")
    parser.add_argument("--events-path", default=None, help=f"Override {EVENTS_PATH_ENV}")
    args = parser.parse_args()

    calls = load_calls(args.events_path)
    splits = get_splits(calls)
    summary = summarize_dataset(calls)
    summary["splits"] = {
        split_name: {
            "count": len(split_calls),
            "switched": sum(call.language_switched for call in split_calls),
            "non_switched": sum(not call.language_switched for call in split_calls),
        }
        for split_name, split_calls in splits.items()
    }
    _print_json(summary)


if __name__ == "__main__":
    main()
