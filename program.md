# autoresearch-language-switch

This repo currently hosts the oracle benchmark bundle for bilingual STT routing.

## Active Benchmark

Use the files under [eval_bilingual_stt](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt).

Treat these as the important files:

- [eval_bilingual_stt/eval_routing_accuracy.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_routing_accuracy.py)
- [eval_bilingual_stt/bilingual_stt_events.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/bilingual_stt_events.json)
- [eval_bilingual_stt/eval_results/oracle](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/oracle)
- [eval_bilingual_stt/spotcheck.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/spotcheck.py)

The copied [prepare.py](/Users/pavan/code/auto-research-language-switch/prepare.py) and [train.py](/Users/pavan/code/auto-research-language-switch/train.py) are legacy and should not be treated as the current benchmark.

## Goal

Measure black-box `BilingualSTT` forwarding quality against Nova-3 multi.

The two benchmark questions are:

1. Positive path: once the oracle reaches clear Spanish, how many wrong forwarded bilingual events happen before Spanish forwarding begins?
2. Negative path: while the oracle is still English, how many Spanish forwards happen spuriously?

Mixed oracle utterances are excluded.

## Canonical Run

Run:

```bash
python eval_bilingual_stt/eval_routing_accuracy.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --output-dir eval_bilingual_stt/eval_results
```

Inspect specific calls with:

```bash
python eval_bilingual_stt/spotcheck.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --call-ids 5c7c8787,de68583a \
  -o /tmp/spotcheck.txt
```

## Metrics

Primary metrics to track separately:

- `avg_switch_delay`
- `zero_delay_calls`
- `false_es_events`
- `false_es_finals`
- `calls_with_false_es`

Do not collapse these into a single score unless explicitly asked. The positive-path and negative-path tradeoff is the point.

## Scope

- This bundle evaluates production black-box forwarded behavior.
- It does not yet replay a new candidate policy.
- `extract_stt_events.py` and `transcribe_oracle.py` were copied for provenance, but they still require Taylor repo production utilities and secrets.
- The portable pieces inside this repo are the cached data, evaluator, and spotcheck script.

## Logging

If you run experiments or modify the evaluator, keep results in untracked artifacts such as:

- `eval_bilingual_stt/eval_results/aggregate.json`
- `eval_bilingual_stt/eval_results/per_call.json`
- `/tmp/spotcheck.txt`

If you need a tabular experiment log, keep using untracked `results.tsv`, but do not pretend the old `dev_bal_acc` workflow is still the active benchmark.
