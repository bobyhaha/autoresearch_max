# Templates — the shape of every artifact this campaign produces

Four record types and three documents. The record types form a **chain of evidence**;
`tools/coe.py` walks it and reports every break.

```
   paper (2506.xxxxx)                      lit/sources/arxiv_<id>_fulltext.txt
        |                                        ^  E1 SOURCE: snapshot on disk,
        |  read in full, not the abstract        |  digest matches the corpus index
        v                                        |
   claim.json ---------------------------------->+   one attributable assertion,
        |   belief_key, locator, stance,             worded no stronger than the source
        |   internal_validity 0-4, transfer 0-4      (scored SEPARATELY, never averaged)
        v
   mechanism.json      E2 LINK: cites registered claim keys; declares the MEDIATOR,
        |              the activation observable, the falsifier, and the competing
        |              explanation that must be ruled out
        v
   hypothesis.json     E2 LINK: cites registered mechanisms AND claims.
        |              Declares an ACTIVATION PREDICATE with an emitted diagnostic.
        |              Failure to activate => INCONCLUSIVE, never a negative result.
        v
   queue entry  -----> variant -----> run -----> result record
        |              E4 METHOD-CODE: the variant must differ from the control and
        |              its cfg must match the hypothesis it claims to test
        v
   paper.md            E5 NUMERIC: every number must appear in `coe.py registry`,
                       which is built ONLY from result records. You may read the
                       registry; nothing may write a number into it by asserting it.
        |
        v
   lesson.json         typed outcome: proceed | refine | pivot | block
```

## Why each rule exists

- **Abstracts are triage, never evidence.** A claim whose source has no full-text
  snapshot is rejected by `claims.py` at registration and by E1 at audit.
- **Validity and transfer never average.** A rigorous 7B/100B-token study can be
  internally excellent and nearly irrelevant at this operating point. One number hides
  that; two numbers show it.
- **Activation is separate from outcome.** A worse `val_bpb` from an intervention that
  never engaged says nothing about the mechanism. Recording it as a negative retires
  good ideas for free.
- **Numbers come from the registry.** Prose is the part of a campaign most likely to
  drift from its evidence, and it drifts in the flattering direction.
- **Failure is typed, not fatal.** `proceed` / `refine` / `pivot` / `block` are
  different consequences; collapsing them into "it didn't work" throws away the
  information the run actually bought.

## Files

| file | registered with | validated by |
|---|---|---|
| `claim.json` | `claims.py add` | `validate_claim` + E1 |
| `mechanism.json` | `claims.py mech` | `validate_mech` + E2 |
| `hypothesis.json` | `claims.py hyp` | `validate_hyp` + E2/E3/E4 |
| `lesson.json` | `claims.py lesson` | `validate_lesson` |
| `paper.md` | written to `papers/` | E5 |
| `round.md` | written to `rounds/` | `council.py` content validation |
| `critique.md` | written to `critiques/` | `council.py` content validation |
