# Research State

This directory implements the five-stage scientific workflow:

```text
Knowledge
    ↓
Ideas ← Toolkit
    ↓
Experiments
    ↓
Evidence
    ↓
Refinement ──→ Knowledge / Toolkit / Ideas
```

The source-of-truth rules are:

- paper metadata and run facts are factual records;
- observations are exploratory and do not count as evidence by themselves;
- evidence keeps measurements separate from the agent's assessment;
- beliefs are versioned interpretations that cite evidence;
- tool definitions, hypotheses, outcomes, controls, and experiments are fingerprinted;
- approved experiments and runs are append-only;
- tags are only search aids and never replace a spec;
- `knowledge/RESEARCH_STATE.md` is generated and must not be edited manually;
- `knowledge/LITERATURE_SYNTHESIS.md` is editorial, but its registry/scope
  snapshot is generated and drift-checked.

This branch contains an append-only historical ledger spanning an obsolete
three-shard scope and the corrected ten-shard scope. Terminal does not mean
current: the generated state verifies the setup scope and demotes unscoped,
mismatched, unsupported, and underpowered beliefs without deleting history.

Validate and generate the current state with:

```bash
python -m vibeautoresearch validate
python -m vibeautoresearch render-state
python -m vibeautoresearch render-literature
python -m vibeautoresearch audit --strict
python -m vibeautoresearch check-setup
```
