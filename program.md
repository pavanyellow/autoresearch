# Program

This repo is for offline bilingual STT forwarding research.

## Objective

Start from the prod-matching baseline in [train.py](/tmp/auto-research-language-switch-0319-v2/train.py) and beat it.

Primary target:

- reduce `text_avg_switch_delay` to `< 0.5`

Primary guardrail:

- keep `text_false_es_rate` at or below the current baseline unless there is a clear reason not to

Current baseline:

- `text_avg_switch_delay = 4.1390`
- `text_false_es_rate = 0.72% (88/12216)`
- `stream_avg_switch_delay = 2.5560`
- `stream_false_es_rate = 1.42% (174/12216)`

## Rules

- Treat [prepare.py](/tmp/auto-research-language-switch-0319-v2/prepare.py) as the fixed scorer/loader unless the harness is broken.
- Edit [train.py](/tmp/auto-research-language-switch-0319-v2/train.py) to change replay behavior.
- The active dataset is bundled in [eval_bilingual_stt/bilingual_stt_events.json](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/bilingual_stt_events.json).
- Use the bundled oracle cache in [eval_bilingual_stt/eval_results/oracle](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/eval_results/oracle).

## Workflow

1. Run `.venv/bin/python train.py`.
2. Inspect the baseline metrics.
3. Change the replay logic in [train.py](/tmp/auto-research-language-switch-0319-v2/train.py).
4. Run `.venv/bin/python train.py --report-path reports/latest.json`.
5. Use [eval_bilingual_stt/spotcheck.py](/tmp/auto-research-language-switch-0319-v2/eval_bilingual_stt/spotcheck.py) on suspicious calls.
6. Keep a change only if it improves `text_avg_switch_delay` without unacceptable regressions in `text_false_es_rate`.

## Interpretation

- `text_*` metrics are primary because they reflect what the LLM effectively received.
- `stream_*` metrics are secondary guardrails because they reflect raw routing purity.
- Mixed oracle utterances are excluded from scoring.
