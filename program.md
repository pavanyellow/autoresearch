# Program

This repo is for offline bilingual STT forwarding research.

## Objective

Start from the prod-matching baseline in [train.py](/Users/pavan/code/auto-research-language-switch/train.py) and beat it.

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

- Treat [prepare.py](/Users/pavan/code/auto-research-language-switch/prepare.py) as the fixed scorer/loader unless the harness is broken.
- Edit [train.py](/Users/pavan/code/auto-research-language-switch/train.py) to change replay behavior.
- The active dataset is bundled in [eval_bilingual_stt/bilingual_stt_events.json](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/bilingual_stt_events.json).
- Use the bundled oracle cache in [eval_bilingual_stt/eval_results/oracle](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/eval_results/oracle).
- Do not cheat. New candidate policies may only use information that would exist online at decision time.

Allowed policy inputs:

- `call.stt_events`
- initial call metadata known at start, such as `call.language`
- `call.agent_session_events` only insofar as each event is consumed causally in timestamp order

Forbidden policy inputs:

- `call.routing_events`
- logged `LLM_RECEIVED` outputs
- oracle transcripts
- aggregate or per-call eval outputs

## Workflow

1. Run `.venv/bin/python train.py`.
2. Inspect the baseline metrics.
3. Change the replay logic in [train.py](/Users/pavan/code/auto-research-language-switch/train.py).
4. Run `.venv/bin/python train.py --report-path reports/latest.json`.
5. Use [eval_bilingual_stt/spotcheck.py](/Users/pavan/code/auto-research-language-switch/eval_bilingual_stt/spotcheck.py) on suspicious calls.
6. Keep a change only if it improves `text_avg_switch_delay` without unacceptable regressions in `text_false_es_rate`.

## Interpretation

- `text_*` metrics are primary because they reflect what the LLM effectively received.
- `stream_*` metrics are secondary guardrails because they reflect raw routing purity.
- Mixed oracle utterances are excluded from scoring.
