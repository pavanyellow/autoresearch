# autoresearch-language-switch

This is an experiment to have the LLM improve bilingual STT switch logic by backtesting on real production calls.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date.
2. Create a fresh branch `autoresearch/<tag>`.
3. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. Verify the local dataset exists:
   - default path: `/Users/pavan/Downloads/bilingual_stt_events.json`
   - override: `AUTORESEARCH_EVENTS_PATH`
5. Initialize `results.tsv` with:

```tsv
commit	dev_bal_acc	dev_fp	dev_fn	status	description
```

6. Confirm setup and start.

## Experimentation

This repo is no longer a model-training repo.

Each experiment is an offline replay of the fixed production-call dataset.

Launch experiments with:

```bash
uv run train.py --split dev
```

You may also use:

```bash
python train.py --split dev
```

## What you CAN do

- Modify `train.py`
- Change the policy logic, thresholds, grouping rules, and decision heuristics
- Add helper functions and policy state inside `train.py`

## What you CANNOT do

- Do not modify `prepare.py`
- Do not change the dataset split logic
- Do not change the evaluation metric definitions
- Do not add dependencies
- Do not commit the production event log into git

## Goal

The primary goal is:

- maximize `dev_bal_acc`

Secondary goals:

- reduce `dev_fp`
- reduce `dev_fn`
- keep latency reasonable
- keep the policy simple

All else equal, simpler is better.

## Output format

`train.py` prints a summary like:

```text
---
dev_bal_acc: 0.923913
dev_precision_switched: 0.800000
dev_recall_switched: 1.000000
dev_fp: 1
dev_fn: 0
dev_median_latency_s: -0.125000
num_dev_calls: 27
```

The key metric is:

```bash
grep "^dev_bal_acc:" run.log
```

## Logging results

Log each experiment to `results.tsv` as tab-separated values:

```tsv
commit	dev_bal_acc	dev_fp	dev_fn	status	description
```

Rules:

1. `commit` = short git hash
2. `dev_bal_acc` = numeric score, use `0.000000` for crashes
3. `dev_fp` = false positives on dev, use `0` for crashes
4. `dev_fn` = false negatives on dev, use `0` for crashes
5. `status` = `keep`, `discard`, or `crash`
6. `description` = short summary of the idea

Example:

```tsv
commit	dev_bal_acc	dev_fp	dev_fn	status	description
a1b2c3d	0.500000	0	4	keep	baseline never-switch sanity check
b2c3d4e	0.891304	2	0	keep	port arrival-time switch baseline
c3d4e5f	0.847826	4	0	discard	lower switch threshold too aggressively
```

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Change `train.py`.
3. Commit the experiment.
4. Run:

```bash
uv run train.py --split dev > run.log 2>&1
```

5. Extract:

```bash
grep "^dev_bal_acc:\|^dev_fp:\|^dev_fn:" run.log
```

6. If the output is empty, treat the run as a crash and inspect:

```bash
tail -n 50 run.log
```

7. Record the result in `results.tsv` and leave that file untracked.
8. If `dev_bal_acc` improved, keep the commit.
9. If `dev_bal_acc` is equal or worse, reset back.

## Test split policy

Do not optimize on the test split during normal iteration.

Use `--split test` only for milestone validation after a meaningful dev improvement.

## Scope assumptions

- v1 is EN->ES only
- every call starts from English as the base language
- the gold switch label is `language_switched`
- the gold switch timestamp comes from `change_language` tool events, with routing-event fallback
