# Bilingual STT Eval Bundle

This directory contains the bundled dataset and the reference black-box evaluator.

## Files

- [bilingual_stt_events.json](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/bilingual_stt_events.json): active `0319_v2` event bundle
- [eval_results/oracle](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/eval_results/oracle): cached oracle transcripts for `508` calls
- [eval_routing_accuracy.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/eval_routing_accuracy.py): black-box evaluator
- [spotcheck.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/spotcheck.py): side-by-side debugging view

Copied for provenance only:

- [extract_stt_events.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/extract_stt_events.py)
- [transcribe_oracle.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/transcribe_oracle.py)

## Current Reference Numbers

On the bundled `0319_v2` data:

- `text_based.avg_switch_delay = 4.1`
- `text_based.false_es_rate = 0.72% (88/12216)`
- `stream_based.avg_switch_delay = 2.6`
- `stream_based.false_es_rate = 1.42% (174/12216)`
- `508 / 508` calls processed
- `72` switched calls

## Run

```bash
.venv/bin/python eval_bilingual_stt/eval_routing_accuracy.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --output-dir eval_bilingual_stt/eval_results
```

```bash
.venv/bin/python eval_bilingual_stt/spotcheck.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --call-ids 5c7c8787,de68583a \
  -o /tmp/spotcheck.txt
```
