# OPHIS generated site

Standalone, self-contained HTML. Open either file directly in a browser
(`open docs/site/ophis-wiki.html`) — no server, no network, no build step.

| File | What it is |
|---|---|
| `ophis-wiki.html` | Full reference for the OPHIS research ledger: the frozen challenge, the five-stage loop, an interactive explorer for all 21 registries (purpose + enforced invariants + fields + a real example record for each), record schemas, the engine and CLI, gate discipline, campaign results, the seven integrity findings, and known drift. |
| `ophis-val-bpb-curve.html` | `val_bpb` across all 42 in-scope gated experiments in campaign order, with the lever changed at each point, the running-best staircase, and the ±2σ noise floor. Hover any point for its paired delta and verdict. |

Both are generated artifacts, not source. The authoritative data is the JSONL
registries under `research/`; regenerate the live view with
`python -m vibeautoresearch render-state`.

Counts in the wiki were read from the working tree, not from the generated
`RESEARCH_STATE.md` (which is currently stale — see the "Known drift" section).
