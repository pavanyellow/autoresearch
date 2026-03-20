# Bilingual STT Routing Eval

This directory contains the current black-box benchmark for `BilingualSTT` forwarding behavior.

It compares production forwarded output against Deepgram Nova-3 multi oracle utterances.

## Files

- [bilingual_stt_events.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/bilingual_stt_events.json): extracted STT/routing/tool events
- [eval_results/oracle](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/oracle): cached oracle utterances
- [eval_routing_accuracy.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_routing_accuracy.py): portable evaluator
- [spotcheck.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/spotcheck.py): side-by-side debugging view

Copied but not standalone in this repo:

- [extract_stt_events.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/extract_stt_events.py)
- [transcribe_oracle.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/transcribe_oracle.py)

Those two still require Taylor repo production utilities and secrets.

## Metrics

Positive path:

- `switch_delay`: after the first clear oracle Spanish utterance, count bilingual forwarded events before the first bilingual Spanish forward

Negative path:

- `false_es_events`: bilingual Spanish forwards during the oracle-English region
- `false_es_finals`: same, finals only

Mixed oracle utterances are excluded from scoring.

## Run

```bash
python eval_bilingual_stt/eval_routing_accuracy.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --output-dir eval_bilingual_stt/eval_results
```

Spotcheck specific calls:

```bash
python eval_bilingual_stt/spotcheck.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --call-ids 5c7c8787,de68583a \
  -o /tmp/spotcheck.txt
```

## Cached Result Set

The copied bundle currently contains:

- `122` calls with oracle JSONs
- `117` processed successfully
- `22` calls with oracle Spanish regions
- `avg_switch_delay = 3.1`
- `false_es_events = 24`

See [eval_results/aggregate.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/aggregate.json) and [eval_results/per_call.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/per_call.json).
