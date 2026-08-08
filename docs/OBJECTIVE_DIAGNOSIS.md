# Why the draft agent loses to consensus

Measured 2026-08-07/08 on the tuning seasons (2022, 2023). **Nothing here has
touched 2024/2025** — those are the gate, ≤3 touches, still unspent.

## The finding in one line

The agent's objective cannot rank *picks*. Optimizing it harder is therefore
strictly harmful, and every planned RL arm optimizes it.

## The symptom

Mean real finishing rank (lower is better), turning up how hard `U` is optimized:

| policy | trust in `U` | mean rank |
|---|---|---|
| `need_adp` | none | **4.21** |
| `season_sim` | 2 rollouts, shrinks toward consensus | 4.62 |
| `oracle` | 50 rollouts, hard argmax | **5.88** |

Monotone in one knob. A single number could be luck; a clean gradient is a
mechanism. It also means the failure is *not* "the search is too weak to find
`U`'s maximum" — the better it gets at that, the worse it does.

## E2 — the objective is coin-flip at the pick level

Regret in real ranks against the best available shortlist candidate:

```
U's argmax           1.582
need_adp's pick      0.891
a random candidate   1.548

spearman(U, actual rank) within a decision: -0.192 (se 0.115, n=16 states)
candidate spread (best vs worst):            2.93 ranks
```

`U`'s argmax is indistinguishable from random. The 2.93-rank spread proves the
decision matters, so this is not "the candidates are interchangeable". `U` does
correlate with outcomes across whole rosters (rho = -0.32, p<0.0001) — it knows a
good team from a bad one, and cannot rank two players at a pick.

## E5 — decomposition: objective vs continuation

A candidate's score is confounded: a bad argmax can come from `U` mis-valuing the
roster, or from the rollout assuming we play `need_adp` for the remaining rounds.
Hand each channel perfect information separately.

|                         | continuation: `need_adp` | continuation: hindsight |
|-------------------------|--------------------------|-------------------------|
| objective: projections  | arm1 (the ceiling)       | arm3                    |
| objective: hindsight    | arm2                     | arm4                    |

plus arm5, `need_adp` drafting off realized value — the pure valuation ceiling.

| arm | gain vs `need_adp` | se | t |
|---|---|---|---|
| arm1 projections + `need_adp` | −3.632 | 0.983 | −3.69 |
| arm2 **hindsight objective** | **+2.950** | 0.564 | +5.23 |
| arm3 hindsight continuation | −2.368 | 1.122 | −2.11 |
| arm4 both | **+3.211** | 0.516 | +6.22 |
| arm5 greedy on realized value | +2.714 | 0.586 | +4.63 |

1. **The objective is the problem.** Fixing only it swings ~+6.6 ranks.
2. **The continuation is not.** Fixing only it buys ~+1.3 and stays deeply
   negative — policy iteration alone would not have rescued this.
3. **The search machinery is sound.** arm4 > arm5: given correct valuations,
   searching *beats* greedily taking the best available player.

arm2/arm4 are cheats by construction, so their level is a bound, not a score.

## Why: the optimizer's curse

The agent takes `argmax_a Q^πc(s,a)` — exactly one step of policy improvement
over `need_adp`. The policy improvement theorem guarantees `V^π' >= V^π`, so
losing to `need_adp` means a precondition broke. Not the exactness of `Q` (E0:
candidates are resolved, not tied) and not the action set (a legality mask).
**The evaluation MDP is not the grading MDP** — the guarantee holds inside the
simulator, and the agent is graded on real seasons.

The mechanism: a shortlist is 8 candidates adjacent in ADP, so their true values
differ by less than projection error. With `mu_hat_i = mu_i + eps_i`,
`build_samples` applies a *mean-preserving* lognormal around `mu_hat_i` — the
model represents outcome uncertainty but assumes its centre is right. So
`E[U | take i]` increases in `mu_hat_i`, and among near-equivalent candidates
argmax selects the largest `eps_i`:

    E[mu_i | i = argmax_j mu_hat_j] < E[mu_i]

This dominates whenever between-candidate spread < projection error, which is
exactly the regime a shortlist creates.

It predicts every signature: the monotone degradation (a sharper argmax selects
more perfectly on `eps`, so estimation noise was *protecting* us — measured
optimal shrinkage is ~100%, i.e. `need_adp`); E2's argmax landing at or below
random (random is unbiased, argmax is selected); and arm2 flipping −3.6 to +3.0
once `eps` is zero.

**`need_adp` is immune** because it ranks by market ADP rather than by its own
projections — nothing is selected on its own error. This is also why the board's
+22.4 pts/pick edge does not contradict any of the above: that is an *estimation*
result averaged over all players, while the draft asks a *selection* question,
where the same error that averages out becomes the thing being maximized over.

## Hypotheses tested and rejected

| # | hypothesis | result |
|---|---|---|
| 1 | miscalibrated waiver floors drive it | RB round 6.0 → 6.1 with measured floors |
| 2 | `U` is risk-seeking (2x title term) | `p_playoffs` alone was **worse** (t=−2.91) |
| 3 | the season model favours RB points | −0.0004 (t=−0.43) at equal projection |
| 4 | VORP distortion feeds the objective | `board.vorp` never reaches `season_sim` |
| 5 | `U` is quantized below the pick gap | separates 7.6/8 candidates already (E0) |

Rejected as *evidence*, not as a hypothesis: **positional timing**. "The agent
drafts RB three rounds early" was used as the behavioural signature of the bug.
arm5 — perfect foresight — drafts RB at mean round 4.9 vs consensus 7.4, so RB
early was *correct* here; and the two best arms have near-opposite profiles
(arm4 RB 8.0, arm5 RB 4.9). The statistic is also crude: mean round over all 15
picks conflates when you first take a position with how many you take.

## Consequences

**Stage 1 and all four RL arms stay on hold.** A value net, a tree, CMA-ES and a
sequence model all optimize this same `U`; a better optimizer of a cursed
estimate is strictly worse, which is what the oracle already demonstrated. Any
value network trained on simulator rollouts inherits the model bias exactly.

**The fixes are standard decision theory:**

1. Rank by a shrunk posterior `lambda*mu_hat + (1-lambda)*mu_market`, not by the
   sample max. `confidence_margin` is a crude one-parameter version that already
   demonstrably helps.
2. Pessimism under model uncertainty: `Q(s,a) - beta*sigma(s,a)`. This is the
   offline-RL fix (CQL/MOPO) for the same disease — an optimizer exploiting a
   model where the model is least trustworthy.
3. Make the model represent its own centring error. `SIGMA_PROJ` currently says
   "outcomes vary around my projection"; it must also say "my projection's centre
   is uncertain, by this much, and by different amounts per position."

## Next: calibrate sigma(eps) by cross-entropy

The season model is a real generative distribution, so the NLL of realized
outcomes under it *is* cross-entropy up to the entropy of the truth — a strictly
proper scoring rule, and much stronger than the current coverage check (80%
intervals contain 79%), which mis-shaped distributions can also satisfy.

- **Descriptive drift:** `KL(p_s || p_t)` between seasons per position x ADP tier.
- **Predictive transfer:** fit `(SIGMA_PROJ, WEEKLY_CV, availability)` on season
  `s`, evaluate NLL on season `t`; the full matrix exposes the correlation
  structure rather than assuming it, and its decay gives *fitted* weights for
  pooling history.

Maximizing log-likelihood fits `sigma(eps)` while penalizing both bias and
mis-scaled uncertainty, which yields `lambda` and `beta` directly. Cheap: pure
density evaluation, no drafts, no GPU.

Traps: align players by **within-season ADP rank bucket, not identity** (identity
does not transfer across seasons, rank does); handle the zero point-mass for
players who never played; cluster/block-bootstrap by player and by week.

## Statistical caveats that apply to everything above

Seasons are **not iid**. Player persistence (the same players recur, and their
projection errors correlate across years — cluster by player), regime drift
(2022 resembles 2023 far more than 2015, so pooling a decade equally calibrates
to a league that no longer exists), and common season factors in `eps` (a
league-wide passing spike misses every WR low together; this partly cancels for
selection *between* candidates but not for calibrating sigma).

Every SE quoted here is a **within-season paired** SE. That is the right tool for
"did this beat consensus in 2022" and badly understates "will this work next
year." Report season-clustered SEs alongside, and treat n=4 as the real sample
size for any generalization claim. Prefer parameters that are *stable across
seasons* over ones that fit best in-sample.
