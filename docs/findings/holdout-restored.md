# Restoring a holdout: ADP-anchored boards for 2019–2021

**Verdict: KEPT. Three clean seasons recovered and sealed in code. ADP is as good
an anchor as ECR — not worse, as a confounded first pass suggested.**

Run 2026-08-16. Script: `experiments/worker_holdout.py`.

## Why

The project had no clean test left. 2024/2025 were the designated gate with a
budget of three touches; five replay artifacts were found on disk. ECR history
begins in 2021, which is what capped the replayable universe at 2022–2025 in the
first place.

FFC ADP availability, verified against the API rather than assumed:

```
half-ppr   2018-2026
standard   2014-2026
ppr        2014-2026
```

Our league is half-PPR, so **2018 is the floor**. An ADP-anchored board needs no
ECR, and `preseason_snapshot`, `join_actuals` and `fit_baseline` all already
accept an injected `history` frame — so `as_ecr_history` reshapes FFC into that
schema and every existing code path works unchanged. One construction with two
anchors, rather than two pipelines that could drift apart and confound the era
with the construction.

## A. The two anchors agree

Within-position Spearman, on the seasons where both exist:

```
season   n both   overall     QB     RB     WR     TE
2022        115     0.945  0.982  0.932  0.955  0.972
2023        172     0.955  0.972  0.965  0.980  0.955
2024        158     0.935  0.917  0.937  0.957  0.955
2025        144     0.969  0.922  0.978  0.983  0.961
```

## B. And they predict equally well — after fixing my own error

The first version of this comparison reported ECR at rho ≈ −0.78 against ADP at
≈ −0.50 and concluded ECR was the better anchor. **That was a range-restriction
artifact.** ECR lists ~500 players and FFC ~150; a rank-versus-outcome correlation
computed over a wider range is mechanically stronger, because it includes
replacement-level players who score near zero. It measured coverage, not accuracy
— the same artifact caught in B1.

Restricted to the players *both* anchors rank:

```
season  anchor    n      QB      RB      WR      TE
2022    ecr     115  -0.486  -0.663  -0.604  -0.538
2022    adp     115  -0.482  -0.576  -0.692  -0.566
2023    ecr     172  -0.305  -0.550  -0.805  -0.553
2023    adp     172  -0.364  -0.541  -0.773  -0.537
2024    ecr     157  -0.577  -0.654  -0.571  -0.278
2024    adp     157  -0.586  -0.646  -0.524  -0.375
2025    ecr     140  -0.253  -0.724  -0.738  -0.239
2025    adp     140  -0.229  -0.693  -0.687  -0.414
```

Sixteen cells, split roughly evenly, differences mostly under 0.1. **The market
and the experts are the same object, functionally** — which is a third
independent confirmation of the no-edge result, alongside V3 and the ECR
circularity in §4 of `WHAT_ARE_WE_OPTIMIZING.md`.

## C. The seasons build, with full coverage

```
season   train seasons        rows  skill  top-180 coverage
2019     [2018]                196    169            100.0%
2020     [2018, 2019]          208    178            100.0%
2021     [2018, 2019, 2020]    222    191            100.0%
```

Name-join match rates against nflverse run **97.5–99.2%**, and `fit_baseline`
passes with `enforce_match_gate=True` — so the safety check stays on rather than
being disabled to make this work.

2018 itself is not buildable: the curve needs at least one prior ADP season and
2018 is the first.

**Their outcomes were not read.** Only structure and coverage were checked, which
is the entire point — measuring board accuracy on them would have spent them.

## A bug this turned up

`build_board` accepted no `history` at all, so it called both `preseason_snapshot`
and `fit_baseline` with the default ECR panel. The first run of this experiment
therefore reported 2021 as a success (it silently built an **ECR**-anchored board,
since ECR covers 2021) and 2019/2020 as data limitations — when all three were one
plumbing bug. Fixed, and pinned by
`test_build_board_actually_uses_the_history_it_is_given`, which builds 2019
specifically because ECR cannot.

The parameter also had to reach `fit_baseline`, not just the snapshot. Feeding the
snapshot an ADP panel while the curve trains on ECR would anchor the two halves of
a board on different rank systems.

## Sealed

`SEALED_SEASONS = (2019, 2020, 2021)` with `MAX_SEALED_TOUCHES = 3`, refused under
`protocol="tune"` and requiring pre-registration under `protocol="gate"`.

The budget is **separate** from the 2024/2025 one. That matters: the old budget is
already exhausted, so counting the seal against the same total would have made the
new seasons unusable the moment they arrived — silently wasting the whole exercise.
Pinned by `test_the_sealed_budget_is_separate_from_the_spent_gate_budget`.

## What it unblocks

The replayable universe goes from four seasons to seven, three of them genuinely
untouched. That is enough to settle the one open question left: V6 found usage at
t = 1.78 as a draft signal on four seasons, short of the bar and explicitly
labelled *not established* rather than null. It is the only candidate that could
still overturn the no-edge result, and it now has somewhere clean to be tested.

Spend the three touches on that, and on nothing else, until it is answered.
