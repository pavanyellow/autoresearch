"""
Editable EN->ES switch-policy runner for autoresearch experiments.

Run with:
    python train.py --split dev
or:
    uv run train.py --split dev
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prepare import CallRecord, Prediction, STTEvent, evaluate_policy, get_splits, load_calls, resolve_events_path

# ---------------------------------------------------------------------------
# Editable policy constants
# ---------------------------------------------------------------------------

ARRIVAL_TIME_GROUPING_WINDOW = 0.8
CONFIDENCE_DECAY_FACTOR = 0.7
MIN_CONFIDENCE_SWITCH_DELTA = 0.05
MIN_NON_PRIMARY_CONFIDENCE = 0.5
PRIMARY_LANGUAGE = "en"
TARGET_LANGUAGE = "es"

# Words that should not trigger a language switch on their own.
# These are common English words or language-ambiguous words that
# the Spanish STT stream can pick up with high confidence from English speakers.
ENGLISH_TRIGGER_BLOCKLIST = {
    "no", "hello", "yes", "hi", "hey", "okay", "ok", "yeah", "yep",
    "nope", "sure", "right", "well", "oh", "uh", "um",
}


class BaselineLanguageSwitchPolicy:
    """
    Simplified EN->ES port of the arrival-time bilingual routing logic.

    This policy only emits one decision:
        switch_to_es = True

    No-switch is represented by returning no decision for the full call.
    """

    def __init__(self) -> None:
        self._current_language = PRIMARY_LANGUAGE
        self._current_language_confidence = 0.0
        self._current_text = ""
        self._last_event_ts = 0.0
        self._switched = False

    def observe_event(self, event: STTEvent) -> Prediction | None:
        if self._switched:
            return None

        if event.stream not in (0, 1):
            return None

        text = event.text.strip()
        if not text:
            return None

        language = PRIMARY_LANGUAGE if event.stream == 0 else TARGET_LANGUAGE
        is_final = event.type == "FINAL"
        confidence = event.conf

        is_same_group = self._last_event_ts > 0 and (event.ts - self._last_event_ts) < ARRIVAL_TIME_GROUPING_WINDOW
        self._last_event_ts = event.ts

        if not is_same_group:
            self._current_language_confidence *= CONFIDENCE_DECAY_FACTOR
            self._current_text = ""

        if confidence < MIN_NON_PRIMARY_CONFIDENCE and language != PRIMARY_LANGUAGE:
            return None

        if language == self._current_language:
            self._observe_current_language(text=text, confidence=confidence, is_same_group=is_same_group)
            return None

        if not is_final:
            return None

        if language != TARGET_LANGUAGE:
            return None

        if text.lower() in ENGLISH_TRIGGER_BLOCKLIST:
            return None

        should_switch, reason = self._should_switch_to_target(
            confidence=confidence,
            is_same_group=is_same_group,
        )
        if not should_switch:
            return None

        self._current_language = TARGET_LANGUAGE
        self._current_language_confidence = confidence
        self._current_text = text
        self._switched = True
        return Prediction(
            switch_to_es=True,
            decision_ts=event.ts,
            decision_audio_ts=event.audio_ts,
            trigger_text=text,
            reason=reason,
        )

    def finalize(self) -> Prediction | None:
        return None

    def _observe_current_language(self, text: str, confidence: float, is_same_group: bool) -> None:
        if is_same_group:
            if confidence > self._current_language_confidence or len(text) > len(self._current_text):
                self._current_language_confidence = max(confidence, self._current_language_confidence)
                if len(text) >= len(self._current_text):
                    self._current_text = text
            return

        self._current_language_confidence = confidence
        self._current_text = text

    def _should_switch_to_target(self, confidence: float, is_same_group: bool) -> tuple[bool, str]:
        if self._current_language != PRIMARY_LANGUAGE:
            return False, "already_switched"

        if is_same_group:
            conf_delta = confidence - self._current_language_confidence
            if conf_delta > MIN_CONFIDENCE_SWITCH_DELTA:
                return True, f"same_group_delta_{conf_delta:.3f}"
            return False, f"same_group_delta_too_small_{conf_delta:.3f}"

        if confidence > self._current_language_confidence:
            return True, f"new_group_higher_conf_{confidence:.3f}"
        return False, f"new_group_not_higher_{confidence:.3f}"


def build_policy() -> BaselineLanguageSwitchPolicy:
    return BaselineLanguageSwitchPolicy()


def _select_calls(split: str, split_map: dict[str, list[CallRecord]]) -> list[CallRecord]:
    if split == "full":
        full = []
        for split_name in ("train", "dev", "test"):
            full.extend(split_map[split_name])
        return sorted(full, key=lambda call: call.call_id)
    return split_map[split]


def _format_metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _build_report(
    metrics: dict[str, float | int | None],
    outcomes: list[Any],
    split: str,
    events_path: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "events_path": events_path,
        "metrics": metrics,
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bilingual STT language-switch backtest.")
    parser.add_argument("--split", choices=("train", "dev", "test", "full"), default="dev")
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    resolved_events_path = resolve_events_path(args.events_path)
    calls = load_calls(str(resolved_events_path))
    split_map = get_splits(calls)
    selected_calls = _select_calls(args.split, split_map)

    metrics, outcomes = evaluate_policy(
        policy_factory=build_policy,
        calls=selected_calls,
        split_name=args.split,
    )

    print("---")
    ordered_keys = [
        f"{args.split}_bal_acc",
        f"{args.split}_precision_switched",
        f"{args.split}_recall_switched",
        f"{args.split}_fp",
        f"{args.split}_fn",
        f"{args.split}_median_latency_s",
        f"num_{args.split}_calls",
    ]
    for key in ordered_keys:
        print(f"{key}: {_format_metric(metrics[key])}")

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = _build_report(
            metrics=metrics,
            outcomes=outcomes,
            split=args.split,
            events_path=str(resolved_events_path),
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
