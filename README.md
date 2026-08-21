# autoresearch

This repository is a small single-GPU LLM training setup plus a structured
automated-science control layer. The main workflow is:

1. prepare the data and tokenizer,
2. run a baseline training job,
3. let an agent ingest scoped knowledge and register the available toolkit,
4. grow a structured idea archive and literature-check candidate novelty,
5. preregister a falsifiable mechanism and hypothesis,
6. pass offline, pilot, matched, multi-seed, validation, and locked gates,
7. preserve run facts and update evidence and beliefs without rewriting history.

The `rsi/baiyu_v3` branch contains an active append-only research ledger. The
generated `research/knowledge/RESEARCH_STATE.md` view is the quickest summary;
the JSONL registries remain authoritative.

## Repository Files

- `prepare.py`: downloads/prepares data, trains the tokenizer, and provides dataloader/evaluation utilities.
- `train.py`: model, optimizer, and training loop. This is the main file changed during experiments.
- `program.md`: instructions for the research agent.
- `vibeautoresearch/`: typed schemas, cross-reference validation, gates, audits,
  and generated research-state support.
- `research/`: populated structured registries for Knowledge, Toolkit, Ideas,
  Experiments, and Refinement.
- `docs/AUTOMATED_SCIENCE_SYSTEM.md`: full design and responsibility of each layer.
- `docs/AGENT_PROTOCOL.md`: exact research-agent workflow.
- `docs/SCHEMA.md`: IDs, versioning, fingerprints, and source-of-truth rules.
- `docs/COST_AND_GATES.md`: offline-first compute and promotion discipline.
- `tools/run_stage.py`: the only remote experiment scheduler; launches bounded
  paired tranches with advisory locks, physical-GPU sampling, and automatic
  RunRecords.
- `tools/run_gated.py`: binds and verifies each arm when invoked by the staged scheduler.
- `run_h200_training.sh`: retired compatibility entrypoint that fails closed.
- `training_dashboard/`: local dashboard for watching logs and run status.

## Requirements

- Python 3.10+
- `uv`
- A single NVIDIA GPU for training, or access to a configured remote GPU machine

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Research-system initialization

From the repo root:

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m vibeautoresearch validate
uv run python -m vibeautoresearch audit --strict
uv run python -m vibeautoresearch list-challenges
uv run python -m vibeautoresearch check-setup
```

The research schemas, CLI, and tests use only the Python standard library and
work on macOS and Linux.

## GPU training initialization

Install the optional training stack, then prepare data and train:

```bash
uv sync --extra training
uv run prepare.py
uv run train.py
```

On Linux, uv obtains PyTorch from the CUDA 12.8 index. On other platforms it
falls back to PyPI so the environment can still resolve, but `train.py` itself
requires an NVIDIA CUDA GPU.

## Remote H200 experiments

The ordered challenge catalog is data-driven:

1. `walltime_5min_h200` — five minutes, currently active and passed;
2. `fixed_steps_2000` — 2,000 steps, currently pending re-measurement.

The latest append-only challenge-selection event is sticky across commands and
process restarts. The repository starts with challenge 1 active. Omitted scope
arguments use that selection and never fall back to challenge 2. Change or stop
it explicitly:

```bash
python -m vibeautoresearch select-challenge walltime_5min_h200 \
  --reason "Work on the five-minute challenge"
python -m vibeautoresearch stop-challenge --reason "Stop challenge work"
python -m vibeautoresearch select-challenge fixed_steps_2000 \
  --reason "Switch to the 2,000-step challenge"
python -m vibeautoresearch hourly-report-status
```

GPU experiments must reference a selected record in
`research/ideas/archive.jsonl` and declare a `search_policy`: the decision
challenge ID, decision frame, research direction and subsystem, expected changes in steps,
quality-per-step, and endpoint quality, the current `1 -> 3 -> 6 -> 10`
paired-seed stage, stopping thresholds, and the immediate parent stage.
Archived ideas preserve title, hypothesis summary, experimental plan, mutation
parents, self-assessed interestingness/novelty/feasibility scores, and up to ten
literature-query rounds. A completed novelty check must cite structured
literature evidence and pass its frozen semantic-similarity threshold.

Launch one bounded paired tranche through the scheduler:

```bash
python tools/run_stage.py exp_name \
  --gpus 0,1,2,3,4,5 \
  --ssh "ssh -p 50002 -o BatchMode=yes user@HOST" \
  --remote-wd /path/to/frozen/remote/repo
```

The scheduler pre-authorizes every exact arm tuple over a live inherited
capability channel and launches at most three seed pairs. Cooperative processes
are serialized by advisory locks. Each run pins
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, resolves the selected GPU to its UUID, binds
CUDA to that UUID, and requires the training PID to appear on the same physical
GPU while sampling for foreign PIDs every two seconds. It then verifies the
exact code/environment and appends lock-protected RunRecords. The sampling
detects observed co-tenancy; it cannot prevent non-cooperating jobs and can miss
a co-tenant whose entire lifetime falls between samples.

Every launched stage-1 experiment is also a direction round. After five
consecutive rounds in one direction, the next stage-1 proposal must select a
different direction. The exhausted direction then remains on cooldown until
two stage-1 rounds have explored other directions, so a one-round token pivot
cannot reset the attachment cap. These limits live in the challenge catalog's
`campaign_policy`, not in the launch command. Funnel continuations do not reset
or consume this diversity counter. One archived idea may enter only one funnel
chain, so it can receive sustained development only by earning the 3-, 6-, and
10-pair stages. The four stages are the idea-level effort cap; a restart or
variant must be a new child idea with its own novelty record. After each
tranche:

```bash
python -m vibeautoresearch evaluate-stage exp_name
```

Do not launch the next tranche unless the verdict is `continue`. A `promote`
verdict requires a newly frozen experiment for the next funnel stage. Direct
remote launchers are retired; `tools/run_gated.py` permits remote execution only
when invoked by the paired scheduler.

The active campaign also requires an immutable summary paper every configured
hour (currently 60 minutes). Check the clock and publish a structured input
file with:

```bash
python -m vibeautoresearch hourly-report-status
python -m vibeautoresearch publish-hourly-report path/to/hourly_report.json
```

Start from `research/reports/hourly/REPORT_INPUT_TEMPLATE.json`.
The command binds the paper to the sticky challenge selection, validates every
idea/experiment/run/evidence reference, appends its record, and renders Markdown
under `research/reports/hourly/`. When a paper is overdue, new gate checks and
GPU authorizations pause; a run already in flight is never killed. A paper must
separate supported findings from preliminary observations, include negative
results and limitations, account for direction coverage, and preregister the
next hour's questions and stop conditions.

## Dashboard

Start the local training dashboard:

```bash
./start_training_dashboard.sh
```

The default dashboard URL is:

```text
http://127.0.0.1:8787/training_dashboard/
```

Use another port if needed:

```bash
DASHBOARD_PORT=8788 ./start_training_dashboard.sh
```

## Agent Workflow

Before proposing work, regenerate and verify the state:

```bash
python -m vibeautoresearch render-state
python -m vibeautoresearch render-literature
python -m vibeautoresearch validate
python -m vibeautoresearch audit --strict
python -m vibeautoresearch check-setup
```

Then ask the agent to read `research/knowledge/RESEARCH_STATE.md`,
`research/knowledge/LITERATURE_SYNTHESIS.md`, and `docs/AGENT_PROTOCOL.md`. A
new training experiment is allowed only after its spec is fingerprinted in
`research/experiments/gated/experiments.jsonl`.

## Project Notes

- Training assumes NVIDIA GPU support.
- Validation metric is `val_bpb`; lower is better.

## License

Unless a source file says otherwise, the project is MIT licensed; see
`LICENSE`. Inherited training sources with Apache-2.0 headers retain those
notices; see `THIRD_PARTY_NOTICES.md`.
