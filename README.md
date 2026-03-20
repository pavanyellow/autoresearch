# autoresearch-language-switch

This repo is a self-contained replay harness for bilingual STT forwarding research.

The root loop is:

- [train.py](/tmp/auto-research-language-switch-0319-v2/train.py): baseline replay runner
- [prepare.py](/tmp/auto-research-language-switch-0319-v2/prepare.py): shared loader and scorer
- [eval_bilingual_stt](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt): bundled data, oracle cache, black-box evaluator, and spotcheck tools

## Goal

Reduce downstream wrong-language output, prioritizing what the LLM actually receives.

Primary target:

- `text_avg_switch_delay < 0.5`

Primary guardrail:

- do not materially worsen `text_false_es_rate`

Current baseline on the bundled `0319_v2` dataset:

- `text_avg_switch_delay = 4.1390`
- `text_false_es_rate = 0.72% (88/12216)`
- `stream_avg_switch_delay = 2.5560`
- `stream_false_es_rate = 1.42% (174/12216)`
- `processed_calls = 508`
- `switched_calls = 72`

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -e .
```

## Canonical Run

Run the root replay with the bundled data:

```bash
.venv/bin/python train.py
```

Write a report:

```bash
.venv/bin/python train.py --report-path reports/latest.json
```

Run the bundled black-box evaluator directly:

```bash
.venv/bin/python eval_bilingual_stt/eval_routing_accuracy.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --output-dir eval_bilingual_stt/eval_results
```

Spotcheck suspicious calls:

```bash
.venv/bin/python eval_bilingual_stt/spotcheck.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --call-ids 5c7c8787,de68583a \
  -o /tmp/spotcheck.txt
```

## How It Works

- The bundled dataset is [eval_bilingual_stt/bilingual_stt_events.json](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/bilingual_stt_events.json).
- The cached oracle transcripts are in [eval_bilingual_stt/eval_results/oracle](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/eval_results/oracle).
- The root baseline replay reconstructs the logged forwarded `LLM_RECEIVED` outputs, so the root numbers match the black-box evaluator.
- The next iteration step is to replace that replay with an editable simulator that uses the bundled `stt_events`, `routing_events`, and `agent_session_events` to beat the baseline.

## Notes

- This repo is self-contained for replay and scoring.
- [eval_bilingual_stt/extract_stt_events.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/extract_stt_events.py) and [eval_bilingual_stt/transcribe_oracle.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/transcribe_oracle.py) are provenance utilities and still depend on Taylor production setup.
- Mixed oracle utterances are excluded from scoring.
