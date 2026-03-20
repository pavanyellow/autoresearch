"""
Shared loader and scorer for the bilingual STT forwarding replay benchmark.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from fast_langdetect import detect as lang_detect

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EVENTS_PATH = REPO_ROOT / "eval_bilingual_stt/bilingual_stt_events.json"
DEFAULT_ORACLE_DIR = REPO_ROOT / "eval_bilingual_stt/eval_results/oracle"

ARRIVAL_TIME_GROUPING_WINDOW = 0.8
CONFIDENCE_DECAY_FACTOR = 0.7
MIN_CONFIDENCE_SWITCH_DELTA = 0.05
PRIMARY_LANGUAGE_SWITCH_DELTA = 0.20
NON_PRIMARY_CONFIDENCE_DECAY = 0.5
ENGLISH_TRIGGER_BLOCKLIST = {
    "no",
    "hello",
    "yes",
    "hi",
    "hey",
    "okay",
    "ok",
    "yeah",
    "yep",
    "nope",
    "sure",
    "right",
    "well",
    "oh",
    "uh",
    "um",
}


@dataclass(frozen=True)
class STTEvent:
    ts: float
    audio_ts: float | None
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
class AgentSessionEvent:
    ts: float
    type: str
    reason: str | None = None
    text: str | None = None
    lang: str | None = None
    confidence: float | None = None
    committed_transcript: str | None = None
    current_interim_transcript: str | None = None
    audio_transcript: str | None = None
    committed_final_transcript: str | None = None


@dataclass(frozen=True)
class OracleUtterance:
    start: float
    end: float
    text: str
    lang: str
    word_langs: dict


@dataclass(frozen=True)
class ForwardedEvent:
    ts: float
    text: str
    lang: str
    is_final: bool
    text_lang: str


@dataclass(frozen=True)
class CallInput:
    call_id: str
    result: str
    duration: int
    language: str
    stt_events: tuple[STTEvent, ...]
    routing_events: tuple[LoggedEvent, ...]
    tool_events: tuple[LoggedEvent, ...]
    agent_session_events: tuple[AgentSessionEvent, ...]
    oracle_utterances: tuple[OracleUtterance, ...]


@dataclass
class CallScore:
    call_id: str
    call_language: str
    first_oracle_es_time: float | None = None
    first_oracle_es_text: str | None = None
    oracle_utterance_count: int = 0
    mixed_utterances: int = 0
    stream_switch_delay: int | None = None
    stream_switch_delay_events: list[dict] = field(default_factory=list)
    stream_false_es_events: int = 0
    stream_false_es_detail: list[dict] = field(default_factory=list)
    text_switch_delay: int | None = None
    text_switch_delay_events: list[dict] = field(default_factory=list)
    text_false_es_events: int = 0
    text_false_es_detail: list[dict] = field(default_factory=list)
    en_region_total_events: int = 0
    error: str | None = None


def resolve_events_path(events_path: str | None = None) -> Path:
    path = Path(events_path) if events_path else DEFAULT_EVENTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Event log not found at {path}")
    return path


def resolve_oracle_dir(oracle_dir: str | None = None) -> Path:
    path = Path(oracle_dir) if oracle_dir else DEFAULT_ORACLE_DIR
    if not path.exists():
        raise FileNotFoundError(f"Oracle directory not found at {path}")
    return path


def detect_text_language(text: str) -> str:
    if not text.strip():
        return "?"
    detected = lang_detect(text)[0]
    if detected["lang"] in ("en", "es") and detected["score"] >= 0.5:
        return detected["lang"]
    return "?"


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def make_forwarded_event(ts: float, text: str, lang: str, is_final: bool) -> ForwardedEvent:
    return ForwardedEvent(
        ts=ts,
        text=text,
        lang=lang,
        is_final=is_final,
        text_lang=detect_text_language(text),
    )


def _load_stt_events(raw_events: list[dict]) -> tuple[STTEvent, ...]:
    parsed = [
        STTEvent(
            ts=float(event["ts"]),
            audio_ts=float(event["audio_ts"]) if event.get("audio_ts") is not None else None,
            type=str(event["type"]).upper(),
            stream=int(event["stream"]) if event.get("stream") is not None else None,
            lang=str(event.get("lang", "?")),
            conf=float(event.get("conf", 0.0)),
            text=str(event.get("text", "")),
        )
        for event in raw_events
    ]
    return tuple(
        sorted(
            parsed,
            key=lambda event: (
                event.ts,
                -1 if event.audio_ts is None else event.audio_ts,
                -1 if event.stream is None else event.stream,
                event.type,
            ),
        )
    )


def _load_logged_events(raw_events: list[dict]) -> tuple[LoggedEvent, ...]:
    return tuple(
        sorted(
            (LoggedEvent(ts=float(event["ts"]), event=str(event["event"])) for event in raw_events),
            key=lambda event: event.ts,
        )
    )


def _load_agent_session_events(raw_events: list[dict]) -> tuple[AgentSessionEvent, ...]:
    return tuple(
        sorted(
            (
                AgentSessionEvent(
                    ts=float(event["ts"]),
                    type=str(event["type"]),
                    reason=event.get("reason"),
                    text=event.get("text"),
                    lang=event.get("lang"),
                    confidence=float(event["confidence"]) if event.get("confidence") is not None else None,
                    committed_transcript=event.get("committed_transcript"),
                    current_interim_transcript=event.get("current_interim_transcript"),
                    audio_transcript=event.get("audio_transcript"),
                    committed_final_transcript=event.get("committed_final_transcript"),
                )
                for event in raw_events
            ),
            key=lambda event: event.ts,
        )
    )


def _load_oracle(oracle_path: Path) -> tuple[OracleUtterance, ...]:
    raw = json.loads(oracle_path.read_text())
    return tuple(OracleUtterance(**item) for item in raw)


def load_calls(events_path: str | None = None, oracle_dir: str | None = None) -> list[CallInput]:
    events_path = resolve_events_path(events_path)
    oracle_dir = resolve_oracle_dir(oracle_dir)
    raw_calls = json.loads(events_path.read_text())

    calls: list[CallInput] = []
    for call in raw_calls:
        call_id = call["call_id"]
        oracle_path = oracle_dir / f"{call_id}.json"
        if not oracle_path.exists():
            continue

        calls.append(
            CallInput(
                call_id=call_id,
                result=str(call["result"]),
                duration=int(call["duration"]),
                language=str(call.get("language", "en")),
                stt_events=_load_stt_events(call.get("stt_events", [])),
                routing_events=_load_logged_events(call.get("routing_events", [])),
                tool_events=_load_logged_events(call.get("tool_events", [])),
                agent_session_events=_load_agent_session_events(call.get("agent_session_events", [])),
                oracle_utterances=_load_oracle(oracle_path),
            )
        )

    return calls


def _infer_lang_from_context(routing_events: tuple[LoggedEvent, ...], ts: float) -> str:
    for event in routing_events:
        if abs(event.ts - ts) > 0.5:
            continue
        if "FINAL EN" in event.event or "INTERIM EN" in event.event:
            return "en"
        if "FINAL ES" in event.event or "INTERIM ES" in event.event:
            return "es"
    return "?"


def extract_logged_forwarded_events(call: CallInput) -> list[ForwardedEvent]:
    forwarded: list[ForwardedEvent] = []
    for event in call.routing_events:
        if "LLM_RECEIVED FINAL:" in event.event:
            text = event.event.split('FINAL: "', 1)[-1].rstrip('"')
            forwarded.append(
                make_forwarded_event(event.ts, text, _infer_lang_from_context(call.routing_events, event.ts), True)
            )
        elif "LLM_RECEIVED interim:" in event.event:
            text = event.event.split('interim: "', 1)[-1].rstrip('"')
            forwarded.append(
                make_forwarded_event(event.ts, text, _infer_lang_from_context(call.routing_events, event.ts), False)
            )
    return forwarded


def extract_multi_stt_forwarded_events(call: CallInput) -> list[ForwardedEvent]:
    primary_language = "en"
    current_language: str | None = None
    current_language_confidence = 0.0
    current_text = ""
    last_event_arrival_time = 0.0
    forwarded_final_stream_index: int | None = None
    forwarded: list[ForwardedEvent] = []

    for event in call.stt_events:
        if event.stream is None or event.stream >= 2:
            continue

        text = event.text.strip()
        if not text:
            continue

        language = "en" if event.stream == 0 else "es"
        confidence = event.conf
        is_final = event.type == "FINAL"
        arrival_time = event.ts
        time_since_last = arrival_time - last_event_arrival_time if last_event_arrival_time > 0 else 0.0
        is_same_group = last_event_arrival_time > 0 and time_since_last < ARRIVAL_TIME_GROUPING_WINDOW
        last_event_arrival_time = arrival_time

        if is_final and is_same_group and forwarded_final_stream_index is not None:
            if event.stream != forwarded_final_stream_index:
                forwarded_language = "en" if forwarded_final_stream_index == 0 else "es"
                if forwarded_language == current_language:
                    continue

        is_primary_language = language == primary_language
        if confidence < 0.5 and not is_primary_language:
            continue

        if not is_same_group:
            forwarded_final_stream_index = None
            if current_language == primary_language:
                current_language_confidence *= CONFIDENCE_DECAY_FACTOR
            else:
                current_language_confidence *= NON_PRIMARY_CONFIDENCE_DECAY
            current_text = ""

        if current_language is None:
            current_language = primary_language

        if language == current_language:
            if is_same_group:
                if confidence > current_language_confidence or len(text) > len(current_text):
                    current_language_confidence = max(confidence, current_language_confidence)
                    if len(text) > len(current_text):
                        current_text = text
            else:
                current_text = text
                current_language_confidence = confidence
        else:
            if not is_final:
                continue

            if language == "es" and current_language == "en":
                if len(text) < 2 or text.lower() in ENGLISH_TRIGGER_BLOCKLIST:
                    continue

            leaving_primary = current_language == primary_language
            required_delta = PRIMARY_LANGUAGE_SWITCH_DELTA if leaving_primary else MIN_CONFIDENCE_SWITCH_DELTA

            if is_same_group:
                conf_delta = confidence - current_language_confidence
                if conf_delta > required_delta:
                    current_language = language
                    current_language_confidence = confidence
                    current_text = text
                else:
                    continue
            else:
                if confidence > current_language_confidence + (required_delta if leaving_primary else 0):
                    current_language = language
                    current_language_confidence = confidence
                    current_text = text

        if is_same_group:
            should_forward = language == current_language
        else:
            should_forward = not is_final or language == current_language

        if not should_forward:
            continue

        forwarded.append(make_forwarded_event(event.ts, text, language, is_final))
        if is_final:
            forwarded_final_stream_index = event.stream

    return forwarded


def _infer_lang_from_events(
    events: list[ForwardedEvent] | tuple[ForwardedEvent, ...],
    ts: float,
    text: str,
    is_final: bool,
    fallback_lang: str,
) -> str:
    normalized_text = _normalize_text(text)
    best_lang = "?"
    best_score = float("-inf")

    for event in events:
        dt = abs(event.ts - ts)
        if dt > 3.0:
            continue
        event_text = _normalize_text(event.text)
        if not event_text:
            continue

        score = SequenceMatcher(None, normalized_text, event_text).ratio() * 100 - dt * 5
        if is_final and event.is_final:
            score += 10
        if not is_final and not event.is_final:
            score += 5
        if normalized_text and event_text and (normalized_text in event_text or event_text in normalized_text):
            score += 15

        if score > best_score:
            best_score = score
            best_lang = event.lang

    if best_score > 35:
        return best_lang

    text_lang = detect_text_language(text)
    if text_lang in ("en", "es"):
        return text_lang
    return fallback_lang


def _es_stream_active_near(stt_events: tuple[STTEvent, ...], ts: float, window: float = 7.0) -> bool:
    for event in stt_events:
        if event.stream != 1:
            continue
        if event.ts > ts:
            break
        if ts - event.ts <= window:
            return True
    return False


def _es_stream_has_started(stt_events: tuple[STTEvent, ...], ts: float) -> bool:
    for event in stt_events:
        if event.stream == 1 and event.ts <= ts:
            return True
        if event.ts > ts:
            break
    return False


def _should_suppress_event(
    text_lang: str, believed_lang: str, call_language: str, text: str,
    stt_events: tuple[STTEvent, ...] | None = None, ts: float = 0.0,
    routed_lang: str | None = None,
) -> bool:
    if text_lang not in ("en", "es"):
        return False
    if text_lang == believed_lang:
        return False
    normalized = text.strip().lower()
    if believed_lang == "en" and text_lang == "es":
        if len(normalized) <= 3 or normalized in ENGLISH_TRIGGER_BLOCKLIST:
            return True
        if routed_lang == "en" and len(normalized) <= 10:
            return True
    if believed_lang == "es" and text_lang == "en":
        if len(normalized) <= 4 or normalized in ENGLISH_TRIGGER_BLOCKLIST:
            return True
        if stt_events is not None:
            if _es_stream_active_near(stt_events, ts):
                return True
            if not _es_stream_has_started(stt_events, ts):
                first_stt_ts = next((e.ts for e in stt_events if e.stream is not None and e.stream < 2), None)
                if first_stt_ts and ts - first_stt_ts < 5.0:
                    return True
    return False


def _update_believed_language(
    believed_lang: str, primary_lang: str, forwarded: list[ForwardedEvent],
) -> str:
    switch_to_window = 3
    switch_away_window = 6
    window = switch_away_window if believed_lang == primary_lang else switch_to_window
    if len(forwarded) < window:
        return believed_lang
    recent = forwarded[-window:]
    recent_finals = [e for e in recent if e.is_final and e.text_lang in ("en", "es")]
    recent_langs = [e.text_lang for e in recent if e.text_lang in ("en", "es")]
    if believed_lang == primary_lang:
        if len(recent_finals) >= 3 and all(e.text_lang != primary_lang for e in recent_finals[-3:]):
            other = "es" if primary_lang == "en" else "en"
            return other
    else:
        if len(recent_langs) >= switch_to_window and all(lang == primary_lang for lang in recent_langs[-switch_to_window:]):
            return primary_lang
    return believed_lang


def extract_agent_session_forwarded_events(call: CallInput) -> list[ForwardedEvent]:
    routed_events = extract_multi_stt_forwarded_events(call)
    forwarded: list[ForwardedEvent] = []
    last_interim_text = ""
    pending_prepend: AgentSessionEvent | None = None
    pending_skips: list[AgentSessionEvent] = []
    primary_lang = "en"
    believed_lang = primary_lang

    for event in call.agent_session_events:
        if event.type == "prepend_buffered_speech":
            pending_prepend = event
            continue

        if event.type == "skip_llm_call":
            pending_skips.append(event)
            continue

        if event.type == "confident_interim_accepted":
            text = (event.current_interim_transcript or event.audio_transcript or "").strip()
            if not text or text == last_interim_text:
                continue
            text_lang = detect_text_language(text)
            routed_lang = _infer_lang_from_events(routed_events, event.ts, text, False, call.language)
            if _should_suppress_event(text_lang, believed_lang, call.language, text, call.stt_events, event.ts, routed_lang):
                continue
            effective_lang = believed_lang if routed_lang != believed_lang and text_lang != routed_lang else routed_lang
            forwarded.append(make_forwarded_event(event.ts, text, effective_lang, False))
            last_interim_text = text
            believed_lang = _update_believed_language(believed_lang, primary_lang, forwarded)
            continue

        if event.type != "eou_detection":
            continue

        text = (event.audio_transcript or event.current_interim_transcript or event.committed_final_transcript or "").strip()
        final_ts = event.ts

        if pending_prepend and abs(pending_prepend.ts - event.ts) <= 0.75:
            text = (pending_prepend.text or text or "").strip()
            final_ts = max(final_ts, pending_prepend.ts)
            pending_prepend = None

        if not text:
            last_interim_text = ""
            continue

        should_skip = False
        for skip_event in pending_skips:
            skip_text = (skip_event.text or "").strip()
            if abs(skip_event.ts - event.ts) <= 1.0 and skip_text == text:
                should_skip = True
                break

        if not should_skip:
            text_lang = detect_text_language(text)
            routed_lang = _infer_lang_from_events(routed_events, final_ts, text, True, call.language)
            if _should_suppress_event(text_lang, believed_lang, call.language, text, call.stt_events, final_ts, routed_lang):
                last_interim_text = ""
                continue
            effective_lang = believed_lang if routed_lang != believed_lang and text_lang != routed_lang else routed_lang
            forwarded.append(
                make_forwarded_event(
                    final_ts,
                    text,
                    effective_lang,
                    True,
                )
            )
            believed_lang = _update_believed_language(believed_lang, primary_lang, forwarded)

        last_interim_text = ""

    return forwarded


def estimate_wallclock_offset(stt_events: tuple[STTEvent, ...]) -> float | None:
    for event in stt_events:
        if event.audio_ts is not None and event.ts:
            return event.ts - event.audio_ts
    return None


def score_forwarded_events(call: CallInput, forwarded_events: list[ForwardedEvent]) -> CallScore:
    score = CallScore(
        call_id=call.call_id,
        call_language=call.language,
        oracle_utterance_count=len(call.oracle_utterances),
        mixed_utterances=sum(1 for utterance in call.oracle_utterances if utterance.lang == "mixed"),
    )

    offset = estimate_wallclock_offset(call.stt_events)
    if offset is None:
        score.error = "no_timestamp_mapping"
        return score

    first_oracle_es = next((utterance for utterance in call.oracle_utterances if utterance.lang == "es"), None)
    en_region_end = first_oracle_es.start + offset if first_oracle_es else float("inf")

    for event in forwarded_events:
        if event.ts >= en_region_end:
            break
        score.en_region_total_events += 1
        if event.lang == "es":
            score.stream_false_es_events += 1
            if len(score.stream_false_es_detail) < 10:
                score.stream_false_es_detail.append(
                    {"ts": event.ts, "text": event.text, "is_final": event.is_final}
                )
        if event.text_lang == "es":
            score.text_false_es_events += 1
            if len(score.text_false_es_detail) < 10:
                score.text_false_es_detail.append(
                    {"ts": event.ts, "text": event.text, "is_final": event.is_final}
                )

    if first_oracle_es is None:
        return score

    score.first_oracle_es_time = first_oracle_es.start
    score.first_oracle_es_text = first_oracle_es.text
    oracle_es_wallclock = first_oracle_es.start + offset

    stream_delay_events: list[dict] = []
    text_delay_events: list[dict] = []

    for event in forwarded_events:
        if event.ts < oracle_es_wallclock:
            continue

        if score.stream_switch_delay is None:
            if event.lang == "es":
                score.stream_switch_delay = len(stream_delay_events)
            elif event.lang == "en":
                stream_delay_events.append(
                    {"ts": event.ts, "text": event.text, "is_final": event.is_final}
                )

        if score.text_switch_delay is None:
            if event.text_lang == "es":
                score.text_switch_delay = len(text_delay_events)
            elif event.text_lang == "en":
                text_delay_events.append(
                    {
                        "ts": event.ts,
                        "text": event.text,
                        "text_lang": event.text_lang,
                        "is_final": event.is_final,
                    }
                )

        if score.stream_switch_delay is not None and score.text_switch_delay is not None:
            break

    if score.stream_switch_delay is None:
        score.stream_switch_delay = len(stream_delay_events)
    if score.text_switch_delay is None:
        score.text_switch_delay = len(text_delay_events)

    score.stream_switch_delay_events = stream_delay_events
    score.text_switch_delay_events = text_delay_events
    return score


def evaluate_replay(
    replay_fn: Callable[[CallInput], list[ForwardedEvent]],
    calls: list[CallInput],
) -> tuple[dict, list[CallScore]]:
    scores = [score_forwarded_events(call, replay_fn(call)) for call in calls]
    processed = [score for score in scores if score.error is None]
    switched = [score for score in processed if score.first_oracle_es_time is not None]
    total_en_events = sum(score.en_region_total_events for score in processed)

    stream_delays = [score.stream_switch_delay for score in switched if score.stream_switch_delay is not None]
    text_delays = [score.text_switch_delay for score in switched if score.text_switch_delay is not None]
    stream_false = sum(score.stream_false_es_events for score in processed)
    text_false = sum(score.text_false_es_events for score in processed)

    metrics = {
        "total_calls": len(calls),
        "processed": len(processed),
        "errors": len(scores) - len(processed),
        "switched_calls": len(switched),
        "stream_based": {
            "avg_switch_delay": round(sum(stream_delays) / len(stream_delays), 3) if stream_delays else None,
            "zero_delay_calls": sum(1 for delay in stream_delays if delay == 0),
            "false_es_events": stream_false,
            "false_es_denominator": total_en_events,
            "false_es_rate": round(stream_false / total_en_events * 100, 4) if total_en_events else 0.0,
        },
        "text_based": {
            "avg_switch_delay": round(sum(text_delays) / len(text_delays), 3) if text_delays else None,
            "zero_delay_calls": sum(1 for delay in text_delays if delay == 0),
            "false_es_events": text_false,
            "false_es_denominator": total_en_events,
            "false_es_rate": round(text_false / total_en_events * 100, 4) if total_en_events else 0.0,
        },
    }
    return metrics, scores


def build_report(metrics: dict, scores: list[CallScore], events_path: str, oracle_dir: str) -> dict:
    return {
        "events_path": events_path,
        "oracle_dir": oracle_dir,
        "metrics": metrics,
        "per_call": [asdict(score) for score in scores],
    }
