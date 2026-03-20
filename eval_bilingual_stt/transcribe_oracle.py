#!/usr/bin/env python3
"""
Transcribe call recordings with Deepgram Nova-3 multi as the oracle.

Downloads user-channel audio from S3, transcribes with nova-3 multi,
labels each utterance with language (en/es/mixed), and caches results.

Usage:
    python scripts/eval_bilingual_stt/transcribe_oracle.py \
        --events scripts/eval_bilingual_stt/bilingual_stt_events.json \
        --output-dir scripts/eval_bilingual_stt/eval_results/oracle/ \
        --max-concurrent 5

    # Specific calls
    python scripts/eval_bilingual_stt/transcribe_oracle.py \
        --events scripts/eval_bilingual_stt/bilingual_stt_events.json \
        --output-dir scripts/eval_bilingual_stt/eval_results/oracle/ \
        --call-ids abc123,def456

    # Skip already-transcribed
    python scripts/eval_bilingual_stt/transcribe_oracle.py \
        --events scripts/eval_bilingual_stt/bilingual_stt_events.json \
        --output-dir scripts/eval_bilingual_stt/eval_results/oracle/ \
        --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import wave
from collections import Counter
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class OracleUtterance:
    start: float
    end: float
    text: str
    lang: str  # "en", "es", or "mixed"
    word_langs: dict


def _extract_user_channel(stereo_path: str) -> str:
    """Extract user channel (ch0) from stereo WAV. Returns path to mono WAV."""
    mono_path = stereo_path.replace(".wav", ".ch0.wav")
    if os.path.exists(mono_path):
        return mono_path

    with wave.open(stereo_path, "rb") as src:
        n_channels = src.getnchannels()
        if n_channels == 1:
            return stereo_path

        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        frames = src.readframes(src.getnframes())

    fmt = f"<{len(frames) // sampwidth}{'h' if sampwidth == 2 else 'b'}"
    samples = struct.unpack(fmt, frames)
    user_samples = samples[0::n_channels]

    with wave.open(mono_path, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(struct.pack(f"<{len(user_samples)}{'h' if sampwidth == 2 else 'b'}", *user_samples))

    return mono_path


def download_audio(call_id: str) -> str | None:
    """Download user-channel WAV from S3. Returns path to mono user-channel WAV."""
    from utils.s3_call_recordings import get_recording_from_s3

    for rec_type in ["local", "telnyx", "twilio", "egress"]:
        files = get_recording_from_s3(call_id, recording_type=rec_type)
        if files:
            return _extract_user_channel(files[0])
    return None


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio with Deepgram nova-3 multi. Returns raw response dict."""
    from deepgram import DeepgramClient, PrerecordedOptions

    from secret_manager import access_secret

    deepgram_api_key = access_secret("deepgram-api-key")
    deepgram = DeepgramClient(deepgram_api_key)

    with open(audio_path, "rb") as f:
        payload = {"buffer": f.read()}

    options = PrerecordedOptions(
        model="nova-3",
        language="multi",
        utterances=True,
        smart_format=False,
        filler_words=True,
        punctuate=False,
    )

    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    return response.to_dict()


def parse_oracle_response(response: dict) -> list[OracleUtterance]:
    """Extract utterances. Deepgram per-word language as primary, fast-langdetect to sanity-check Spanish."""
    from fast_langdetect import detect as lang_detect

    utterances = []
    raw_utterances = response.get("results", {}).get("utterances", [])
    words = response.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("words", [])

    for utt in raw_utterances:
        start = utt["start"]
        end = utt["end"]
        text = utt["transcript"]

        utt_words = [w for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
        lang_counts = Counter()
        for w in utt_words:
            lang = w.get("language", "en").split("-")[0].lower()[:2]
            lang_counts[lang] += 1

        total_words = sum(lang_counts.values())
        dg_lang, dg_count = lang_counts.most_common(1)[0] if lang_counts else ("en", 1)
        dg_ratio = dg_count / total_words if total_words > 0 else 0

        if dg_ratio >= 0.7:
            if dg_lang == "es":
                fl = lang_detect(text)[0]
                if fl["lang"] == "en" and fl["score"] >= 0.5:
                    label = "en"
                else:
                    label = "es"
            else:
                label = dg_lang
        else:
            label = "mixed"

        utterances.append(
            OracleUtterance(
                start=start,
                end=end,
                text=text,
                lang=label,
                word_langs={"deepgram": dict(lang_counts), "fasttext": lang_detect(text)[0]["lang"]},
            )
        )

    return utterances


async def process_call(call_id: str, output_dir: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    """Download, transcribe, and cache oracle result for one call."""
    async with sem:
        try:
            audio_path = download_audio(call_id)
            if not audio_path:
                return call_id, "no_audio"

            response = transcribe_audio(audio_path)
            utterances = parse_oracle_response(response)

            oracle_path = os.path.join(output_dir, f"{call_id}.json")
            with open(oracle_path, "w") as f:
                json.dump([asdict(u) for u in utterances], f, indent=2)

            return call_id, f"{len(utterances)} utterances"
        except Exception as e:
            return call_id, f"error: {e}"


async def main():
    parser = argparse.ArgumentParser(description="Transcribe call recordings with Deepgram Nova-3 multi")
    parser.add_argument("--events", required=True, help="Path to bilingual_stt_events.json")
    parser.add_argument("--output-dir", required=True, help="Directory for oracle JSON files")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Deepgram API concurrency limit")
    parser.add_argument("--call-ids", help="Comma-separated call IDs (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip calls with existing oracle JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.events) as f:
        all_calls = json.load(f)

    call_ids = [c["call_id"] for c in all_calls]
    if args.call_ids:
        target = set(args.call_ids.split(","))
        call_ids = [cid for cid in call_ids if cid in target]

    if args.skip_existing:
        existing = {f.replace(".json", "") for f in os.listdir(args.output_dir) if f.endswith(".json")}
        call_ids = [cid for cid in call_ids if cid not in existing]

    print(f"Transcribing {len(call_ids)} calls (max_concurrent={args.max_concurrent})")

    sem = asyncio.Semaphore(args.max_concurrent)
    tasks = [process_call(cid, args.output_dir, sem) for cid in call_ids]

    done = 0
    transcribed = 0
    no_audio = 0
    errors = 0
    for coro in asyncio.as_completed(tasks):
        call_id, status = await coro
        done += 1
        if "utterances" in status:
            transcribed += 1
        elif "no_audio" in status:
            no_audio += 1
        else:
            errors += 1
            print(f"  [{call_id[:8]}] {status}")
        if done % 10 == 0 or done == len(call_ids):
            print(f"  {done}/{len(call_ids)}  transcribed={transcribed}  no_audio={no_audio}  errors={errors}")

    print(f"\nDone: {transcribed} transcribed, {no_audio} no audio, {errors} errors")
    print(f"Oracle files: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
