# Council round — <UTC ISO>

<!--
  Validated on CONTENT by tools/council.py, not on mtime. It checks: all four sections
  present, each above its word floor, each naming a DISTINCT agent id, the synthesizer
  not also a reviewer, and a parseable ```queue block. `touch` does not satisfy it.

  Before dispatching: run `python3 tools/agenda.py` for the active direction, then
  `python3 tools/lit.py read <direction>` and `python3 tools/claims.py show <direction>`,
  and READ the unread full texts. A round whose direction still has unread papers is a
  round arguing from ignorance.

  Replace every <agent-id> with the real subagent identity. Distinct role labels inside
  one context are not independence.
-->

Active direction: `<direction>` — <why agenda.py chose it>
Registered evidence consulted: <belief_keys and mechanism names>
Unread full texts remaining for this direction: <n>

## explorer (<agent-id>)

3–5 interventions, concentrated on the active direction, each citing the `belief_key`
of the claims behind it. For each: the causal mediator, why the effect could be large
*here*, and the observable that proves it engaged. Any axis `tools/direction.py` marks
UNEXPLORED is a first obligation — a knob never tried once is not evidence about it.

## pragmatist (<agent-id>)

Can each be expressed in `train.py` alone? `prepare.py` is frozen. What does it cost in
throughput, and does that cost exceed the gain at a fixed 300s budget? Rank by expected
val_bpb gain per unit of implementation risk. Kill what cannot be built.

## critic (<agent-id>)

Destroy each surviving proposal *before* it runs. Name the confound, the cheaper
explanation, and the evidence that would already refute it — including any registered
`opposes` claim the explorer walked past, and any claim being cited beyond what its
`transfer` score supports. Any proposal whose predicted effect is under the measured
noise band is not runnable at n=1: say so and demand a yoked pair or a bigger lever.

## synthesis (<agent-id> — must be none of the above)

Say which objection changed which proposal. Preserve dissent rather than averaging it;
if the critic and explorer still disagree, record the disagreement and what would settle
it. Then emit the queue.

```queue
[
  {"name": "R1_example",
   "cfg": {"dbs": 128, "tbs": 19, "depth": 8, "dim": 512, "mlp": 4, "ve": 2, "win": "SSSL", "swdiv": 2},
   "hypothesis_id": "hyp_optional_but_preferred",
   "rationale": "why this run, in one sentence, grounded in a cited claim",
   "falsifier": "what result would refute it",
   "expected": "predicted direction and size vs the measured band"}
]
```

Then run `python3 tools/queue_from_round.py` — it builds every variant immediately, so a
broken edit fails now rather than at 3am on a GPU.
