# autoresearch-language-switch

This repo now carries the bilingual STT oracle benchmark workspace used to audit `BilingualSTT` forwarding behavior against Deepgram Nova-3 multi.

The active benchmark lives under [eval_bilingual_stt](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt). It is a black-box eval of what production actually forwarded, not the older call-level `language_switched` backtest.

## Current Workspace

- [eval_bilingual_stt/bilingual_stt_events.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/bilingual_stt_events.json): extracted production routing/STT events
- [eval_bilingual_stt/eval_results/oracle](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/oracle): cached Nova-3 multi oracle transcripts
- [eval_bilingual_stt/eval_routing_accuracy.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_routing_accuracy.py): portable evaluator
- [eval_bilingual_stt/spotcheck.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/spotcheck.py): side-by-side call inspection

Copied for provenance but not portable by themselves in this repo:

- [eval_bilingual_stt/extract_stt_events.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/extract_stt_events.py)
- [eval_bilingual_stt/transcribe_oracle.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/transcribe_oracle.py)

Those two still depend on Taylor production utilities and secrets for S3 / Deepgram access.

## Benchmark

The current oracle eval measures two routing failure modes:

- Positive-path delay: after the oracle sees the first clear Spanish utterance, how many bilingual forwarded events happen before the first forwarded Spanish event
- Negative-path false positives: how many Spanish forwards happen while the oracle is still in English

Mixed oracle utterances are excluded.

This is the current bundled result set:

- `122` calls with oracle transcripts cached
- `117` processed successfully
- `22` calls with oracle Spanish regions
- `avg_switch_delay = 3.1` bilingual events
- `false_es_events = 24`

See [eval_bilingual_stt/eval_results/aggregate.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/aggregate.json) and [eval_bilingual_stt/eval_results/per_call.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/per_call.json).

## Run

Run the portable evaluator with the bundled data:

```bash
python eval_bilingual_stt/eval_routing_accuracy.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --output-dir eval_bilingual_stt/eval_results
```

Generate a spotcheck for specific calls:

```bash
python eval_bilingual_stt/spotcheck.py \
  --events eval_bilingual_stt/bilingual_stt_events.json \
  --oracle-dir eval_bilingual_stt/eval_results/oracle \
  --call-ids 5c7c8787,de68583a \
  -o /tmp/spotcheck.txt
```

## Status

The old root-level [prepare.py](/Users/pavan/code/auto-research-language-switch/prepare.py) and [train.py](/Users/pavan/code/auto-research-language-switch/train.py) are the earlier call-level `language_switched` harness. They are still present for reference, but they are not the benchmark to use for the current bilingual routing work.
