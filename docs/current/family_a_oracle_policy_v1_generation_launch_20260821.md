# Family-A Oracle Policy Dataset V1 Generation Launch

Date: 2026-08-21

## Status

Early health classification: **`DATASET_GENERATION_RUNNING_HEALTHY`**

The scaled dataset generation was launched in detached tmux and monitored for
approximately three minutes only. The job was left running.

## Prelaunch

- branch: `contextual-compositional-heuristics-20260731`
- HEAD: `8e1223beb58fd4d296061b6b48e3ba493714108f`
- upstream: `origin/contextual-compositional-heuristics-20260731`
- ahead/behind: `0 / 0`
- git locks: none found
- tmux sessions before launch: none
- active scientific jobs before launch: none found
- CPU count: `20`
- launch-time load: `1.00 / 0.75 / 0.65`
- free RAM: `22Gi free / 58Gi available`
- free disk: `638G`

Prelaunch tests:

```
python3 -m pytest -q \
  tests/test_family_a_oracle_policy_v1_generation.py \
  tests/test_family_a_oracle_policy_pilot_v1.py \
  tests/test_family_a_observability_continuation_v1.py \
  tests/test_family_a_receding_horizon_oracle_v1.py
```

Result: `51 passed in 445.91s`.

Dry run:

```
python3 scripts/generate_family_a_oracle_policy_v1.py dry-run \
  --output-dir datasets/family_a_oracle_policy_v1 \
  --workers 4 \
  --target-scenarios 704 \
  --dry-run-offset 88 \
  --dry-run-scenarios 1 \
  --max-events-per-scenario 3 \
  --min-event-step-gap 100 \
  --max-extra-steps 1500
```

Result: `dry_run_ok`, `n_rows=1`, deterministic checksum
`bb03840ac5e311da7a3acdb3cc995661f6e3cbc1bc900b1025f3c978a7b9fae0`.

## Launch

tmux session: `family_a_oracle_dataset_v1_1k`

Master PID: `156831`

Worker PIDs:

- shard 0: `156878`
- shard 1: `156879`
- shard 2: `156880`
- shard 3: `156881`

Command:

```
cd <repo-root> &&
PYTHONUNBUFFERED=1 python3 -u scripts/generate_family_a_oracle_policy_v1.py run-all \
  --output-dir datasets/family_a_oracle_policy_v1 \
  --log-dir logs \
  --workers 4 \
  --target-scenarios 704 \
  --max-events-per-scenario 3 \
  --min-event-step-gap 100 \
  --max-extra-steps 1500 \
  > logs/family_a_oracle_dataset_v1_1k.log 2>&1
```

Output path: `datasets/family_a_oracle_policy_v1/`

Master log: `logs/family_a_oracle_dataset_v1_1k.log`

Shard logs:

- `logs/family_a_oracle_dataset_v1_1k.shard_000.log`
- `logs/family_a_oracle_dataset_v1_1k.shard_001.log`
- `logs/family_a_oracle_dataset_v1_1k.shard_002.log`
- `logs/family_a_oracle_dataset_v1_1k.shard_003.log`

Start time: `2026-08-21T20:51:30-04:00`

## Three-Minute Health Check

Final sampled time: `2026-08-21T20:54:51-04:00`

- tmux session: alive
- master process: alive
- worker processes: all four alive
- CPU: each worker about `109-110%`, master about `1.7%`
- memory: each worker about `282-292MiB RSS`, master about `196MiB RSS`
- master log: grew from `293B` to `1.7K`
- shard logs: grew from `138B` each to `1.4-1.6K`
- progress JSON files: present and updating
- shard row CSVs: present; header-only at final check because initial skew-1
  control scenarios yielded zero eligible rows
- error scan count: `0` across master and shard logs for traceback/error/OOM/
  duplicate/corrupt/TEST/integrity/disk-error patterns

Progress at final check:

- labels generated: `0`
- scenarios completed: about `35`
- shards completed: `0 / 4`
- current scenario positions: shard 0 at `10/176`, shard 1 at `10/176`,
  shard 2 at `10/176`, shard 3 at `9/176`

The zero-label early progress is expected for the initial low-skew control
block and is not a health concern; the row-producing dry run validated that
eligible high-skew rows are retained.

Preliminary ETA: unstable. Based on the pilot estimate of about `37.6` hours
single-process for 1,000 labels and `4` workers, an order-of-magnitude wall
estimate is roughly `9-12` hours, but early zero-label control scenarios are
not representative of later high-skew branch cost.

## Confirmation

- detached tmux job left running
- no wait for completion
- only approximately three minutes of post-launch monitoring
- no TEST
- no model training
- no controller deployment
- no commit/push/stage/stash/reset/clean
