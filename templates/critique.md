# Fable critique — <UTC ISO>

<!--
  Four INDEPENDENT Fable subagents (model claude-fable-5), dispatched in parallel, each
  reading PRIMARY state: runs/sweep/results/*.json, the generated variant sources,
  lit/claims.jsonl, the host over read-only ssh. Never the owning agent's narration.

  Fable has NO execution authority: it may not stage, kill, or edit train.py. It may
  propose claims, lessons, mechanisms and protocol changes; the owning agent verifies
  and registers them through the normal gates.

  Validated on content by tools/council.py: four sections, distinct agent ids, an
  independent synthesizer, word floors.
-->

Inspected: <exact files, result records, and host state, with timestamps>

## fable_evidence (<agent-id>)

Recompute every headline number yourself from `runs/sweep/results/*.json` — do not
accept one. Which claim is strongest, and which is most likely to be withdrawn? What has
been rated too highly? Is any effect being quoted that is smaller than the measured
band, or any number cited that `tools/coe.py registry` does not contain?

## fable_method (<agent-id>)

Where do specification, code, activation diagnostic and interpretation disagree? Read
the generated variant **source**, not the config, and verify each variant contains the
edit it claims. Which assumptions are unmeasured, ranked by cheapest decisive check?

## fable_process (<agent-id>)

Is the portfolio over-concentrated? Check `tools/direction.py` for axes still UNEXPLORED
after N runs and say why they were skipped. Check `tools/agenda.py`: is the active
direction being exploited past the point of return, or abandoned while it was still
paying? Were the previous critique's recommendations adopted or ignored — name the
ignores explicitly, with their cost.

## fable_synthesis (<agent-id> — independent of the three above)

What should be refined, pivoted, blocked, or tested next, each with a falsifier and an
activation diagnostic. What in the research **system** itself must change, and what
evidence justifies it? Close with an explicit fact / inference / disagreement ledger.
