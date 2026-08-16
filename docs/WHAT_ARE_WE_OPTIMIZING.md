# What are we optimizing, and is it wrong?

Written 2026-08-10, after V1–V4 and the P(title) regrade. This is the walkthrough
document — it assumes nothing and defines its terms. `OBJECTIVE_DIAGNOSIS.md` is
the terse version; `WHERE_WE_ARE.md` is the project status.

---

## 0. Glossary

Every term used below, defined once.

| term | meaning |
|---|---|
| **ADP** | Average Draft Position. Where a player actually gets drafted, averaged over many real drafts. The *market price*. |
| **ECR** | Expert Consensus Ranking. FantasyPros' aggregate of many analysts' rankings. A ranking, not a valuation — it says who is better, never by how much. |
| **ppg** | Points per game. |
| **the curve** | A function fitted on past seasons: given a position and a rank within that position, return expected ppg. `curve.ppg_at("RB", 12)` = "what does the 12th-best RB score?" |
| **the board** | Our draft board: one row per player with `proj_ppg`, `proj_games`, `VORP`. |
| **VORP** | Value Over Replacement Player. A player's projected points minus the points of the worst player you'd still have to start at that position. Converts raw points into "how much does this pick actually help me." |
| **`need_adp`** | The baseline policy. Take the best-ranked available player that fills a roster need. No valuation, no simulation, no search. This is roughly what a competent human does. |
| **`season_sim`** | Our agent. Simulates seasons to score candidates, then picks the best. |
| **`oracle`** | Our agent with the search turned up to maximum (50 rollouts, no shrinkage). Not a cheat — it has no hindsight. It is simply the best available optimizer of `U`. |
| **`U`** | The objective. `U = P(make playoffs) + 2 × P(win the title)`. |
| **rollout** | One simulated continuation: finish the draft, then simulate the season. |
| **argmax** | "Take the option with the highest score." |
| **residual** | Realized minus predicted. What the model got wrong. |
| **systematic vs idiosyncratic** | Systematic error follows a pattern (e.g. "we always underrate TEs"). Idiosyncratic error is player-specific and patternless. Systematic error is fixable; idiosyncratic is not, without new information. |
| **t-statistic** | Effect size divided by its standard error. Rule of thumb: \|t\| > 2 means "probably not chance." \|t\| < 1 means "indistinguishable from nothing." |
| **paired comparison** | Run two policies through the *same* simulated world and compare within-pair. Cancels the shared luck and detects much smaller effects than comparing separate runs. |
| **holdout / gate** | Seasons deliberately never looked at, reserved to test a final claim once. |

---

## 1. What the agent actually does

At each of our 15 picks:

1. **Shortlist.** Take the top ~8 available players by ADP that fit a roster need.
2. **Score.** For each candidate: pretend we drafted them, simulate the rest of the
   draft (assuming we play `need_adp` for the remaining rounds), then simulate the
   season many times.
3. **Objective.** Compute `U = P(playoffs) + 2·P(title)` for each candidate.
4. **Choose.** Take the candidate with the highest `U`.

Step 4 is one step of what reinforcement learning calls **policy improvement** over
`need_adp`. There is a theorem — the policy improvement theorem — that says this
should never be *worse* than the baseline you improved over.

It is worse. So a precondition of the theorem broke, and finding which one is the
whole diagnosis.

---

## 2. The symptom: it gets worse the harder it tries

Mean finishing rank, lower is better, on the tuning seasons (2022 + 2023):

| policy | how much it trusts `U` | mean rank |
|---|---|---|
| `need_adp` | not at all | **4.18** |
| `season_sim` | 2 rollouts, shrinks toward consensus | 4.82 |
| `oracle` | 50 rollouts, pure argmax | **5.61** |

Monotone in a single knob. One bad number could be luck; a clean gradient is a
mechanism.

**This immediately rules out one whole class of explanation.** It is not "the
search is too weak to find `U`'s maximum." The better it gets at finding that
maximum, the worse it performs. The search is working. What it is searching for is
the problem.

---

## 3. Estimation is not selection

This is the conceptual crux, and it is the part most people's intuition gets wrong.

**The board looks good on its own terms.** Tested walk-forward (fit on past
seasons, tested on a future one, never peeking), it beats its baseline by **+22.4
points per pick, standard error 8.6, t = 2.60, better in 4 of 4 seasons**.
Coverage 100% vs 81%.

That number is real but it is **not** evidence that we beat expert consensus —
the baseline it beats is naive persistence, not ECR. See §4, which should be read
before putting any weight on it.

So how does a good projection model produce a bad draft agent?

Because a draft does not ask an **estimation** question ("how many points will
this player score?"). It asks a **selection** question ("of these 8 similar
players, which is best?"). Those have completely different error properties.

Write our estimate of player *i* as:

```
estimate_i = truth_i + error_i
```

Across all 500 players, `error_i` averages out — that is why the board looks good.
But when you deliberately take the **maximum**, you are not averaging. You are
selecting. And among candidates whose true values are close together, the one with
the highest estimate is mostly just the one with the largest positive error.

The formal statement, the **optimizer's curse** (also called the winner's curse):

```
E[ truth_i | i was chosen as the argmax ] < E[ truth_i ]
```

Read it carefully: the expected *true* value of the thing you selected is **below**
average — not above. Selecting on a noisy estimate is anti-selection whenever the
noise is large relative to the real differences.

And notice what a shortlist is: **8 players adjacent in ADP**. It is engineered to
make the true differences small. Meanwhile the projection error is unchanged. The
shortlist deliberately constructs the exact regime where the curse dominates.

**Why `need_adp` is immune.** It never computes a value at all. It takes the
best-ranked player that fills a need and stops. *You cannot be wrong about a
quantity you never compute.* It inherits whatever error the experts have, but it
does not add selection error on top of it.

**Why our uncertainty model didn't save us.** `build_samples` wraps each projection
in a lognormal spread, so the model does represent that outcomes vary. But that
spread is **mean-preserving** (`distributions.py:238`, comment: *"keep E[mu_s] =
mu"*). It says "outcomes vary around my projection." It never says "my projection's
centre might be in the wrong place." So `E[U | take i]` is still increasing in our
point estimate, and argmax still selects on the error.

---

## 4. How would we even know if we beat ECR?

Our board is built *from* ECR. So "does it beat ECR?" is a circularity question,
and it needs care.

### The +22.4 is not that measurement

`validate.py:84-85` names the two things being compared:

```python
("ECR baseline curve",                 "pred_ecr")
(f"clean.py: {holdout-1} ppg_played",  "pred_prior_year")
```

It compares *ECR plus our curve* against **last season's points per game** —
naive persistence. So the result says the experts know more than a stale box
score. That is true, unsurprising, and it is **the experts' result, not ours.**
It has been quoted in this project as general evidence of board quality; it is
not.

### Within a position, "beating ECR" is definitionally a tie

`proj_ppg = curve.ppg_at(position, pos_rank)`, and the curve is monotone in rank.
So our within-position ordering is **identical** to ECR's — Spearman exactly 1.0,
always, by construction. No experiment can show us beating ECR at ranking running
backs, because we do not rank running backs. We copy their ranking and attach
numbers to it.

The only things we add that ECR does not supply:

1. **Cardinal magnitude.** ECR says RB5 > RB6, never by how much.
2. **The cross-position exchange rate.** RB12 vs WR15. V1 measured it: right to
   within 0.86 ppg against a 2.77 ppg gap. Not adding error — and not adding
   edge.
3. **Availability** (`proj_games`). V4 found our one real defect lives here
   (`missed_prior`, t = −2.58).

### The real test is at the policy level, and it already runs

The right comparison is not at the projection level at all:

> **`need_adp` *is* the ECR baseline.** "Take the best ECR-ranked player who
> fills a need," and nothing else.

So **`season_sim` vs `need_adp` *is* the "do we beat ECR?" experiment.** That is
what the whole ceiling study has been. Answer: −0.636 ranks pooled, sign-flipping
across seasons, season-clustered t well under 1 (§13). At best a tie.

Everything built on top of ECR — the curve, the season simulator, the search — is
measured against ECR-alone and buys nothing detectable.

### ECR is not the market

`board.py:93` sets `adp_rank = ecr_rank`, quietly treating **expert consensus**
and **market price** as the same object. They are not. Real ADP is on the board
as `ffc_adp` and is consumed **only by the opponent model** — it never touches
our valuation.

Two consequences:

- **The framing has been slightly wrong.** We have been trying to beat expert
  consensus, but what actually governs a draft is market price: ADP decides who
  is still on the board when our pick comes. Different bars.
- **ECR-vs-ADP disagreement is a free, unused signal.** Where the market drafts a
  player earlier than the experts rank him, one side knows something. Prior that
  this yields edge is low — V3 found deviations from market underperform
  (t = −2.07) — but V3's exact contrast should be re-read before treating that as
  closing the question.

---

## 5. Direct confirmation: E2

Measured at the level of individual picks, against real outcomes. "Regret" = how
many ranks worse than the best available candidate.

```
U's argmax                                    1.582
a randomly chosen candidate                   1.548
need_adp's pick                               0.891

Spearman(U, actual outcome) within a pick:   -0.192  (se 0.115, n=16 states)
spread between best and worst candidate:      2.93 ranks
```

Three readings:

1. **`U`'s pick is indistinguishable from random** (1.582 vs 1.548), and the
   correlation is *negative*. As a ranker of picks, `U` carries no information.
2. **The decision genuinely matters.** The 2.93-rank spread rules out "these
   candidates are all the same anyway." There is real value on the table; we
   just aren't capturing it.
3. **`U` is not broken in general.** Across whole rosters it correlates with real
   outcomes at rho = −0.32, p < 0.0001. It reliably tells a good team from a bad
   team. It cannot tell two players apart at a single pick.

That last point is the precise failure. `U` works at the resolution it was
designed for and fails at the resolution we use it at.

---

## 6. Which channel is at fault: E5

A candidate's score mixes two things. When we score "draft player X," we assume we
then play `need_adp` for rounds 2–15. So a bad score could come from:

- **(a) the objective** — `U` mis-valuing the resulting roster, or
- **(b) the continuation** — the assumption about how we finish the draft being wrong

To separate them, hand each channel **perfect information** (the actual realized
season points) one at a time. n = 32 paired drafts per arm:

| arm | what gets the truth | mean rank | gain vs `need_adp` | se | t |
|---|---|---|---|---|---|
| `need_adp` | — | 4.69 | — | — | — |
| arm1 | nothing (the real agent) | 7.28 | **−2.594** | 0.883 | −2.94 |
| arm2 | **the objective** | 1.38 | **+3.312** | 0.560 | **+5.92** |
| arm3 | the continuation | 5.62 | −0.938 | 0.957 | −0.98 |
| arm4 | both | 1.22 | +3.469 | 0.537 | +6.46 |
| arm5 | `need_adp` on realized value | 1.62 | +3.062 | 0.564 | +5.43 |

arms 2/4/5 are cheats by construction — their *level* is a ceiling, not a score.
What matters is the comparison between them.

1. **Valuation is the whole story.** Fixing only the objective swings **+5.91
   ranks** (−2.594 → +3.312): from significantly worse than consensus to
   significantly better.
2. **The continuation is worth nothing.** Fixing only it lands at −0.938,
   t = −0.98 — statistically indistinguishable from consensus. It repairs damage
   and generates **zero edge**. Every planned RL arm was aimed at this channel.
3. **The search machinery is fine.** arm4 (+3.469) at least matches arm5 (+3.062)
   — searching with correct valuations does as well as simply taking the best
   available player with correct valuations. The search is not the bottleneck.
   (Treat arm4 ≈ arm5 as the supported claim; their SEs overlap.)

**Conclusion: 100% of the headroom is in valuation.** This is the most important
result in the project.

---

## 7. So we hunted for valuation edge. Four experiments, four negatives.

Residual is defined as `log(realized ppg / predicted ppg)`.

### V1 — is the curve systematically biased?

| component | share of residual variance |
|---|---|
| systematic (position × rank tier) | **3.0%** |
| idiosyncratic (player-specific) | **97.0%** |

Cross-position spread: **0.86 ppg**, against a **2.77 ppg** gap between adjacent
draftable players.

My standing hypothesis had been "our cross-position exchange rate is wrong — we
misjudge how RB #12 compares to WR #15." **Refuted.** The exchange rate is correct
to well within the resolution that matters. And 97% of the error is player-specific
noise, invisible to any model whose only input is a rank.

### V2 — is any position's bias stable enough to correct?

Only **RB** (−0.101, sd 0.052, never flips sign across seasons). QB, WR and TE flip
sign year to year — which means they are noise, not bias, and "correcting" them
would be fitting last season.

### V3 — when we disagree with the market, are we right?

Pooled correlation between our deviation-from-market and subsequent outperformance:
**−0.086, t = −2.07.**

**Negative.** Where our model says a player is better than the market does, that
player tends to *underperform*. Our disagreements with consensus are not merely
uninformative — they are slightly harmful.

### V4 — where does the mispricing live? (pre-registered)

| feature | n | corr | t | by position |
|---|---|---|---|---|
| `is_rookie` | 614 | +0.016 | +0.40 | QB +0.03, RB −0.06, WR +0.04, TE +0.20 |
| `draft_capital` | 614 | −0.023 | −0.56 | QB +0.15, RB −0.13, WR +0.07, TE +0.05 |
| `age` | 614 | +0.001 | +0.01 | QB +0.04, RB −0.01, WR −0.05, TE +0.11 |
| `missed_prior` | 554 | **−0.109** | **−2.58** | QB −0.15, RB +0.02, WR −0.15, TE −0.31 |

Multivariate **R² = 0.043** (n = 306, tuning seasons only).

- **Rookies are not mispriced** (t = 0.40). Neither is draft capital, nor age.
- **Injury history is the one signal that survives** — and it implicates *our*
  model, not the market: `proj_games = curve.games_at(position, rank)`, so a
  player with three injured seasons and a durable player at the same rank get
  identical availability. Fixable, and small.
- **Team environment failed to load** (wrong column name in `team_weekly`), so the
  "weak-division schedule inflates production" hypothesis is **untested**.
- Together these explain **4.3%** of a residual that is 97% of the total error.

Caveat worth stating: V4 measured **bias**, not **variance**. "Rookies are
higher-variance" is a different claim and remains untested — and given a title
objective, it is the more relevant one.

---

## 8. The rescue attempt, and why it failed

One escape remained. Every number above is **mean finishing rank**. But the goal is
to **win the league**, not to place well. And `U` carries a 2× on the title term,
which makes it deliberately **variance-seeking**. So perhaps we had been penalizing
the agent for correctly accepting variance — diagnosing a feature as a bug.

Regraded the same ceiling data on the right metric (2022 + 2023, 44 paired drafts):

| policy | mean rank | **P(title)** | P(playoffs) | sd of rank |
|---|---|---|---|---|
| `need_adp` | 4.18 | **0.136** | 0.795 | 2.73 |
| `season_sim` | 4.82 | **0.136** | 0.727 | 2.96 |
| `oracle` | 5.61 | **0.068** | 0.614 | 3.14 |

Paired against `need_adp`: oracle title gain **−0.068** (se 0.060, t = −1.13);
`season_sim` **+0.000** (se 0.065).

**The rescue fails.** The mechanism was real — the oracle *did* take more variance,
sd 3.14 vs 2.73 — and it won *fewer* titles anyway. Not decisive on its own (44
drafts is thin for a 13% event, so this is directional), but it points the wrong
way, so nothing is saved.

**Why, intuitively:** variance only buys title odds when it is **free** — same
expected value, wider spread. The oracle bought variance by *paying* mean. A
distribution that is both worse-centred and wider does not put more mass on 1st
place; it slides the whole thing toward the back and fattens both tails. You want
variance at equal expected value. The oracle got variance at worse expected value.

---

## 9. The answer: which "what we're optimizing" is wrong?

"What we're optimizing" is four separate things, and the answer differs for each.

### 1. The objective's *form* — not the problem

`U = P(playoffs) + 2·P(title)`. Tested directly: `p_playoffs` alone was **worse**
(t = −2.91), and `−mean_rank` did not fix positional timing either. `U` is also
well-aligned with the stated goal of winning. **Not the bug.**

### 2. The grading metric — was wrong, and it didn't matter

Every number was mean finishing rank, which rewards consistently finishing 3rd.
That is a genuine methodological error, now corrected (§8). Correcting it **changed
nothing**. Real mistake, no consequences.

### 3. The inputs — not *wrong*, **empty**

This is the one people miss. `proj_ppg = curve.ppg_at(position, pos_rank)`
(`board.py:63-67`). Verified in code: **no player identity enters the projection at
all**. And `board.py:93` sets `adp_rank = ecr_rank`.

Consequences:

- **Within a position, our ordering is identical to the experts' by construction.**
  We are structurally incapable of disagreeing about which RB is better.
- The only thing the curve adds is the **cross-position exchange rate** — and V1
  says that is already right.
- Therefore we hold **no private information whatsoever**. We are a repackaging of
  ECR with a scoring function bolted on.
- And V3 says the small amount we *do* add is **negatively** correlated with
  outcomes.

The right word is not "mispriced." It is **we do not have a price.**

### 4. The operation — actively wrong

Given (3), taking `argmax` over a shortlist does not merely fail to add value. Via
the optimizer's curse it **converts zero information into negative information**.

### The one-line answer

> **The objective is fine. The inputs carry no information. And the argmax turns
> no information into negative information.**

That is exactly why the failure is monotone in optimization pressure, and exactly
why every planned RL arm — value network, MCTS, CMA-ES, sequence model — is on
hold. All four optimize this same `U` harder. A better optimizer of an empty
estimate is *strictly worse*, which is what the oracle already demonstrated.

---

## 10. What still stands

Not everything is negative. These results survive intact:

- **The board's estimation edge over naive persistence**: +22.4 pts/pick,
  t = 2.60, 4/4 seasons. Note the baseline — this is not an edge over ECR, and
  cannot be (§4).
- **The search machinery**: arm4 ≈ arm5 — search with correct values works.
- **The simulator**, post-FLEX-fix, with true CRN and deterministic sorting.
- **The gate discipline**, now enforced in code (`src/evaluation/protocol.py`).
- **The diagnosis itself.** A clean, replicated demonstration of the optimizer's
  curse in a real sequential decision problem, with a monotone dose-response
  curve and a hindsight decomposition isolating the channel. That is a genuine
  result, and the more useful half of this project.

---

## 11. Where the leverage could still be

### A. Play consensus

`need_adp` is the optimum on **both** metrics in our own data. This is what the
evidence supports, and it should be the default until something beats it.

### B. Fix the argmax, not the price

Even with no edge, stop the bleeding:

- **Shrinkage** — rank by `λ · (our estimate) + (1 − λ) · (market)` instead of the
  raw maximum. `confidence_margin` in `search.py:437-462` is a crude
  one-parameter version that already demonstrably helps.
- **Pessimism** — score `Q(s,a) − β · σ(s,a)`, penalizing candidates we are least
  certain about. This is the standard offline-RL fix (CQL, MOPO) for precisely
  this disease: an optimizer exploiting a model exactly where the model is least
  trustworthy.

**Predicted outcome:** measured optimal λ is near zero, i.e. it converges to
`need_adp`. So this recovers the baseline rather than beating it. Worth doing
anyway — it turns "we have no edge" from a rhetorical claim into a measured
number.

### C. Change the source of edge entirely

**C1 — genuinely new information.** Usage and opportunity metrics (targets,
carries, snap share, route participation, air yards), Vegas totals, offensive line
quality. Honest prior: **low**. Professional projections already use these, and V3
says our deviations have run negative. V4 tested four cheap features, not the
space — so this is not closed, just unpromising.

**C2 — correlation and stacking. The one real gap.**

Verified in code: there is **not a single** correlation, covariance or rho term
anywhere in `season.py` or `distributions.py`. Every player's weekly score is drawn
independently.

Reality is not independent. A quarterback and his top receiver score together — the
same touchdown pays both. Deliberately rostering both ("stacking") raises the
**variance of your weekly team score without changing its mean.**

That is *exactly* the free variance §8 showed the oracle failed to obtain. And it
is the right lever for a title objective, because with no valuation edge the only
way to raise P(title) above the 1-in-12 base rate is variance at equal expected
value.

Three things make this the most attractive remaining option:

1. **It does not require beating the market at anything.** Correlation is a
   property of the roster you construct, not a forecast you have to be right
   about. It is immune to the optimizer's curse — there is no argmax over noisy
   estimates involved.
2. **It has never been tried** — not rejected, but *unrepresentable*. The
   simulator cannot express it, so it has never appeared in any experiment as
   either a benefit or a cost.
3. **It is measurable from data we already have.** Week-level scoring for
   teammates is in the nflverse `weekly` table; the correlation can be estimated
   directly and dropped into `build_samples` as a covariance structure.

**Recommendation: B, then C2.** B is cheap and makes the negative result rigorous.
C2 is the only untried thing with an actual mechanism for raising P(title).

---

## 12. Constraint on any future claim: the holdout is spent

- ECR history begins in 2021, so `_train_seasons` can only build boards for
  **2022–2025**. That is the entire replayable universe: **four seasons.**
- 2024 and 2025 were the designated gate, budget **3 touches**. Five replay
  artifacts were found on disk — **the budget is spent.** Enforced in code and
  pinned by `tests/test_protocol.py`.
- Consequence: **there is currently no clean holdout.** Any new claim tested on
  2022–2025 is tuning, not evidence.
- **Fix available:** Fantasy Football Calculator publishes real ADP back to 2012
  (half-PPR from 2018). An ADP-anchored board does not need ECR, so **2018–2020
  is a buildable, genuinely untouched holdout.** Build it before making any
  claim that needs to survive.

---

## 13. How confident are we actually? Season-clustered standard errors

Every standard error quoted above is a **within-season paired** SE. That is the
right tool for "did this beat consensus in 2022." It badly understates "will this
work next year," because it treats replicates within one season as independent
evidence about seasons. They are not: within a season the board, the projections,
and the realized player outcomes are all **fixed**. The replicates vary only in
who falls to us and in the simulator's draws.

Decomposed on the ceiling data:

```
oracle, paired gain vs need_adp
  2022:  -3.000   (within-season se 0.895, n=22)
  2023:  +0.136   (within-season se 0.700, n=22)
  between-season spread:  3.136 ranks
  mean within-season se:  0.798

season_sim
  2022:  -1.682
  2023:  +0.409
  between-season spread:  2.091 ranks
```

**The season-to-season swing is roughly 4× the within-season noise, and for the
oracle the sign flips.** Clustering by season:

```
mean gain            -1.432
season-clustered se   1.568
t = -0.91
```

**This corrects a number quoted throughout this project.** The oracle's deficit
has been cited at t = −1.95; clustered by season it is **t = −0.91**. The claim
"the oracle loses to consensus" is **not statistically established** by that
experiment on its own.

### Consequence for simulation budget

Two independent error sources, and more simulations only touch one. Using the
components above (within-season sd per replicate ≈ 3.74, between-season sd ≈ 2.22):

| plan | compute | pooled SE |
|---|---|---|
| 22 drafts × 2 seasons (today) | 1× | 1.67 |
| **1000 drafts** × 2 seasons | **45×** | 1.57 |
| 22 drafts × **8 seasons** | **4×** | **0.83** |

**45× the compute buys 6%. 4× the seasons buys 50%.** More simulations per season
is near-worthless; more seasons is the only axis that moves the answer. This is
the argument for §12's 2018–2021 extension.

(Caveat: this σ_B is estimated from two seasons — essentially one degree of
freedom. Treat 3.14 as an order of magnitude. It does not change the conclusion,
because within-season noise is already small relative to the observed gap.)

### Re-ranking the findings by how much weight they bear

**Solid.** The code facts — `proj_ppg` a function of rank alone with no player
identity, and zero correlation modelling in the simulator. These are not
statistical claims. V1's 97% / 3% variance decomposition. V4's R² = 0.043
(n = 306–614). V3's t = −2.07, pooled across players.

**Solid enough.** E5's hindsight decomposition. The effects are large (+3.31,
t = +5.92) and structural rather than marginal — "hand the objective the truth and
it swings 5.9 ranks" does not hinge on which two seasons were run.

**Weak.** The ceiling's monotone ordering and the P(title) regrade. Two seasons,
sign flips, season-clustered t below 1. The direction is consistent across three
policies, and a dose-response across a single knob is stronger evidence than one
comparison — but it is suggestive, not demonstrated.

The overall diagnosis survives, because it rests mainly on the first two tiers.
But the number leaned on most in conversation is the flimsiest one.
