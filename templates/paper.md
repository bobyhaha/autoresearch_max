# <one-sentence finding, not a topic>

<!--
  The paper is the campaign's public claim about itself, so it is the artifact most
  likely to drift from its evidence -- and prose drifts in the flattering direction.
  Two rules make that hard:

  E5 NUMERIC (AutoResearchClaw §3.4): every number here must appear in
  `python3 tools/coe.py registry`, which is built ONLY from runs/sweep/results. You may
  read the registry; you may not add to it by asserting a value. A derived number
  (a difference, a ratio) must be shown as a derivation from cited registry values,
  not stated as if it were measured.

  Run `python3 tools/coe.py` before publishing. Unmatched numbers are a rejection.
-->

- **Scope:** val_bpb, 300 charged training seconds, single H200, frozen `prepare.py`.
- **Covers:** <result names, hypothesis ids, and claim keys new since the last paper>
- **Chain of evidence:** `python3 tools/coe.py` — <INTACT | N breaks, listed in §6>

## 1. What this block establishes

The finding in three sentences. If no experiment landed, say **"no new experimental
evidence"** here and synthesise only literature, mechanism, implementation or process
progress. Never recycle the previous paper with a new timestamp.

## 2. Evidence

| result | cfg | val_bpb | steps | activation | verdict |
|---|---|---|---|---|---|

Every row copied from a result record. For each hypothesis tested, state the activation
outcome **before** the outcome: a run whose diagnostic was absent or failed its rule is
**inconclusive** about the mechanism and must not be reported as a negative.

State the comparison design (yoked pair or unpaired) and the measured noise band from
`tools/direction.py`, including how it was measured. An effect below the band from a
single unpaired run is "below the noise floor", not a result.

## 3. What each experiment taught about its mechanism

Not whether it won — what it revealed about the mediator. A valid negative that
localises *why* is worth more than a win nobody can attribute.

## 4. Beliefs changed

Which registered claims and mechanisms were strengthened, weakened, contested or left
alone. Include claims that now look **over-rated**, with the reason.

## 5. Strongest claim, and the one most likely to be withdrawn

State both. A paper that names no vulnerable claim has not been read adversarially.

## 6. Failures, typed

`runtime | integrity | non_activation | valid_negative | degenerate | resource | scope |
overclaim` — with the lesson registered for each and its action
(`proceed | refine | pivot | block`). Chain-of-evidence breaks from `tools/coe.py` go
here with their remediation.

## 7. Next hypotheses

Each with a mechanism, an activation diagnostic, a falsifier, and a predicted effect
size compared against the measured band. Say which direction they belong to and whether
`tools/agenda.py` currently has that direction open.

## 8. Fact / inference / disagreement

Separate them explicitly. **Fact** = in a result record or a cited full text.
**Inference** = reasoned, with its uncertainty. **Disagreement** = where this paper
contradicts an earlier one, stated plainly rather than quietly dropped.
