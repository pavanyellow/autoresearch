# autoresearch-language-switch

This repo repurposes `autoresearch` into an offline backtesting loop for bilingual STT language-switch logic.

The goal is to improve the EN->ES switch decision when English and Spanish Nemotron streams run in parallel on real production call logs.

## What matters

This repo now has two important files:

- `prepare.py` - fixed offline benchmark harness. Do not modify during the experiment loop.
- `train.py` - the only file the agent edits. It contains the current switch policy.

`program.md` describes the autonomous experiment workflow.

## Dataset

The benchmark uses a local production event log and keeps it out of git.

- default path: `/Users/pavan/Downloads/bilingual_stt_events.json`
- override: `AUTORESEARCH_EVENTS_PATH=/path/to/bilingual_stt_events.json`

Expected dataset shape:

- one list entry per call
- call-level labels including `language_switched`
- per-call `stt_events`, `routing_events`, and `tool_events`

The current dataset has 136 calls:

- 20 switched calls
- 116 non-switched calls

The fixed split is stratified with seed `42`:

- train: 82 calls = 12 switched, 70 non-switched
- dev: 27 calls = 4 switched, 23 non-switched
- test: 27 calls = 4 switched, 23 non-switched

## Metric

Primary metric:

- `dev_bal_acc` = balanced accuracy on the dev split

Supporting metrics:

- `dev_precision_switched`
- `dev_recall_switched`
- `dev_fp`
- `dev_fn`
- `dev_median_latency_s`

Higher `dev_bal_acc` is better.

## Quick start

Validate the local dataset:

```bash
python prepare.py
```

Run the current baseline policy on the dev split:

```bash
python train.py --split dev
```

Or with uv:

```bash
uv run train.py --split dev
```

Write a per-call report:

```bash
python train.py --split dev --report-path reports/dev_report.json
```

## Policy shape

`train.py` replays the STT events in timestamp order and lets the policy emit at most one decision:

- switch to Spanish
- or no decision

The baseline is a simplified port of the current arrival-time bilingual routing logic:

- English is the starting language for every call
- only final Spanish events can trigger a switch
- same-group switching requires a confidence delta threshold
- new-group switching requires a higher confidence than the current English state

## Experiment loop

The intended autoresearch loop is:

1. edit only `train.py`
2. run `python train.py --split dev > run.log 2>&1`
3. read `dev_bal_acc` and error counts from `run.log`
4. keep or discard the change
5. occasionally validate on `--split test`

Artifacts such as `results.tsv`, `run.log`, and `reports/` are intentionally gitignored.
