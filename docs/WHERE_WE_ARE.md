# Where this project actually stands

Written 2026-08-09. Read this first if you have lost the thread. It assumes no
memory of previous sessions and explains every term it uses.

---

## 1. What we are trying to build

A program that drafts a fantasy football team better than you would by following
consensus rankings. In 2025 the previous version finished 11th of 12, which is
what started the rebuild.

The plan was a research build: port the simulator to run on a GPU, then try four
different machine-learning approaches against the same test, and see which wins.

**That plan is currently on hold, for a reason that is now measured rather than
suspected.** Sections 5 and 6 explain why.

---

## 2. Glossary

Everything below uses these. No other jargon is introduced without definition.

### Fantasy football

| term | meaning |
|---|---|
| **ADP** | Average Draft Position. Where a player typically gets taken across thousands of real drafts. The market's opinion. |
| **ECR** | Expert Consensus Ranking. Where *experts* rank a player. Similar to ADP but NOT the same, and confusing the two caused a real bug (section 4). |
| **FFC** | Fantasy Football Calculator, a site that publishes real ADP measured from actual drafts. |
| **VORP** | Value Over Replacement Player. How much better someone is than the freely available alternative. |
| **FLEX** | A roster slot that accepts any running back, receiver, or tight end. |
| **half-PPR** | Half Point Per Reception. A scoring rule; assumed to be your league's format (see section 9 — this is unverified). |

### Statistics

| term | meaning |
|---|---|
| **SE** | Standard Error. How uncertain a measured average is. Smaller means more confident. |
| **t** | The measured effect divided by its standard error. Bigger than about 2 in size means "probably real, not chance." t = 6.46 is very strong; t = 0.98 is nothing. |
| **rho** | A rank correlation, between -1 and +1. Zero means no relationship. |
| **n** | How many observations a number is based on. |
| **regret** | How much worse your choice was than the best choice available, judged afterwards. |
| **iid** | Independent and identically distributed: the assumption that every observation is a fresh, unrelated draw. Seasons are NOT this (section 8). |
| **paired** | Comparing two methods on the *same* draft with the *same* luck, so the difference reflects the method rather than chance. |

### Machine learning

| term | meaning |
|---|---|
| **RL** | Reinforcement Learning. Learning by trial and error toward a goal. |
| **MCTS** | Monte Carlo Tree Search, the algorithm behind AlphaGo. Explores a branching tree of futures. |
| **CMA-ES** | An evolutionary optimizer that tunes a small set of readable parameters by trial and error. |
| **argmax** | "Whichever option scored highest." For scores [3, 9, 4] the argmax is the second one. |
| **rollout** | One simulated play-through of the rest of the draft plus a season, to see how a choice turns out. |
| **shortlist** | The ~8 plausible players the program actually evaluates, instead of all 500. |
| **objective** | The scoring function the program is trying to maximize. Ours is called **U**. |
| **continuation policy** | The program's assumption about *how it will draft in later rounds* while simulating. |

### Our experiment labels

These were internal shorthand and are only meaningful here.

| label | what it asked |
|---|---|
| 0a–0g | The repair jobs: fix bugs in the simulator before trusting any measurement |
| **0e** | How much could better search possibly be worth, at best? |
| **E0** | Is the objective too coarse to tell candidates apart? |
| **E2** | Can the objective rank individual *picks*, or only whole teams? |
| **E5** | Is the fault in the objective, or in the continuation assumption? |
| E1, E3, E4 | E1/E4 cancelled once E0 made them pointless. E3 never ran (the cloud machine was reclaimed). |

---

## 3. How the draft agent works

You are on the clock. Here is the machine, step by step.

1. **Shortlist.** Of ~500 available players, keep the 8 most plausible by ADP.
   No point evaluating a kicker in round 2.
2. **Roll out each one.** Pretend you take him, then simulate the remaining 12
   rounds. The other 11 teams pick using a model of drafter behaviour; *you*
   pick using a simple fallback rule. Now you have a full 15-man roster.
3. **Score that roster.** Simulate a whole 14-week season with it — injuries,
   byes, waiver pickups, head-to-head matchups. Did you make the playoffs? Win
   the title? Repeat 300–2000 times with different luck and count:

   ```
   U = (share of simulated seasons you made the playoffs)
     + 2 x (share of simulated seasons you won the title)
   ```

4. **Take the argmax.** Draft whichever of the 8 scored highest.

Two pieces of that deserve names, because they turned out to be the entire story:

- **the objective** — U, the scoring function in step 3
- **the continuation policy** — the fallback rule in step 2, i.e. the program's
  assumption about its own future drafting. It currently assumes it will follow
  consensus.

The baseline we compare everything against is called **`need_adp`**: follow
consensus rankings, but respect roster limits so you do not end up with eight
quarterbacks. That is what a competent human does, and it is the bar.

---

## 4. What we fixed before trusting any number

A broken measuring instrument makes every downstream result meaningless, so this
came first.

| fix | what was wrong | effect |
|---|---|---|
| **FLEX hindsight** | the flex starter was chosen *knowing how the week turned out* | worth **+6.20 points per week** of free points. The real damage was the **2.17 points/week spread between rosters** — a hidden bonus for hoarding bench players |
| **re-ran the test** | every previous number came through that bug | the agent's measured edge went from **+0.207 (t=1.58)** to **+0.485 (t=2.69)** |
| **determinism** | results depended on internal dictionary ordering | now bit-for-bit identical across runs |
| **common random numbers** | comparisons were not actually using matched luck; the old approach made paired comparisons *noisier* than unpaired | variance ratio **1.10 → 0.74** |
| **opponent realism** | simulated drafters were built from ECR (expert disagreement), measured about **2x too erratic** vs real drafts | replaced with real ADP spread from FFC. Simulated field strength went from 4.47 to **7.03** |
| **checkpointing** | long jobs held everything in memory and lost it if killed | proved itself immediately, preserving 780 rows when jobs died |

Test suite: **110 tests before, 141 now, all passing.**

Two of these deserve emphasis. The FLEX bug meant *every number the project had
ever produced* was measured through a scoring rule that granted the agent
knowledge of the future. And the ECR/ADP mix-up meant we were training against
opponents who behaved nothing like real drafters.

---

## 5. The result that stopped the plan

Take one dial — how hard the program optimizes U — and turn it up. The number is
average finishing position out of 12, so **lower is better**:

```
follow consensus, ignore U entirely     4.21
trust U a little (2 rollouts, hedged)   4.62
trust U completely (50 rollouts)        5.88
```

The harder it optimizes its own objective, the worse it finishes. Not randomly
worse — worse in order, at every step.

That trend rules out the obvious explanation. It is **not** "the search is too
weak to find U's best answer," because the better it gets at finding that
answer, the worse it does. The best answer is in the wrong place.

Against that, we also measured the **valuation ceiling**: drafting by what
players *actually* scored that season beats consensus by **+2.84 places**
(SE 0.32). That is the total prize available from better player evaluation.

---

## 6. Finding out why: five dead ends, then two answers

### Five hypotheses, all rejected

| # | idea | test | outcome |
|---|---|---|---|
| 1 | the waiver-wire assumptions distort position scarcity | measured the true values, swapped them in | changed behaviour by 0.1 rounds. No. |
| 2 | the "2x title" term makes it reckless | tried three different objectives | the alternatives were **worse** (one at t = -2.91). No. |
| 3 | the season simulator favours running backs | swapped RB and WR at equal projection | t = -0.43, i.e. nothing. No. |
| 4 | a known distortion in VORP feeds the objective | traced the code | VORP never reaches this code path at all. No. |
| 5 | U is too coarse to tell candidates apart | E0 | it already separates 7.6 of 8 candidates; more precision changes nothing. No. |

Hypothesis 2 is the informative one. Changing the *shape* of the objective made
things worse, which rules out "we wrote down the wrong goal" and points instead
at the **numbers being fed into it**.

### E2 — the objective cannot rank picks

For many real draft situations, we asked: which of the 8 did U choose, and how
good was that choice really?

```
average places lost versus the best available candidate
  U's choice           1.582
  consensus's choice   0.891
  a random candidate   1.548

how much the choice mattered (best vs worst): 2.93 places
```

**U's choice is no better than choosing at random.** Consensus loses about half
as much. And 2.93 confirms the choice genuinely matters — the 8 candidates were
not interchangeable.

Yet U *does* correlate with real outcomes when comparing whole teams
(rho = -0.32). So U can tell a good roster from a bad one. It cannot tell which
of two players to take right now. Only the second skill wins drafts.

### E5 — which half is broken?

A candidate's score depends on two things that could each be wrong: the
objective (step 3) and the continuation assumption (step 2). So we handed each
one perfect information — literally letting it see what actually happened that
season — separately, and watched what changed. 32 paired drafts per arm.

```
arm                                      finish   gain vs consensus     t
consensus (the bar)                       4.69          --
arm1: normal objective, normal rollout    7.28        -2.594          -2.94
arm2: PERFECT objective                   1.38        +3.312          +5.92
arm3: PERFECT continuation assumption     5.62        -0.938          -0.98
arm4: both perfect                        1.22        +3.469          +6.46
arm5: just take the best players          1.62        +3.062          +5.43
      (using hindsight)
```

- **arm2** — fixing only the objective takes the agent from 2.59 places *worse*
  than consensus to 3.31 places *better*. A swing of **5.91 places**, t = 5.92.
- **arm3** — fixing only the continuation assumption improves things by 1.66
  places but lands at t = -0.98, which is **indistinguishable from just
  following consensus**. It stops the bleeding; it produces no edge.
- **arm4 vs arm5** — searching with correct valuations (1.22) does about as well
  as simply taking the best players with hindsight (1.62). **The search
  machinery works.** It was being fed meaningless numbers.

**Conclusion: the objective's player valuations are the problem. Not the search,
not the tree depth, not the opponent modelling.**

---

## 7. Why it fails: the noisy tape measure

Suppose you want the tallest of 8 people, and your tape measure is **unbiased
but noisy** — sometimes 2 inches high, sometimes 2 inches low, no systematic
error. Suppose those 8 people are all within an inch of each other.

Measure all 8, take the highest reading. **Who did you pick?** Almost certainly
not the tallest — you picked whoever the tape happened to over-read most. When
real differences (1 inch) are smaller than measurement error (2 inches), ranking
your measurements mostly ranks your errors.

That is exactly our situation:

- the 8 shortlisted players are adjacent in ADP, so genuinely close in value
  (the "same height" part)
- projection error is comparable to or larger than those gaps (the "noisy tape
  measure" part)
- the program takes the highest projected value (the "highest reading" part)

So it systematically selects whichever player its projections are most
*over*-optimistic about. In decision theory this is called the **optimizer's
curse**: taking the maximum of noisy estimates gives a biased answer even when
each individual estimate is unbiased.

This explains everything observed:

- **why more optimization hurts** — a sharper maximum selects the error more
  perfectly. The noise in a weak search was randomly knocking the program off
  the cursed pick, accidentally protecting it. Hence hedged (4.62) beats
  confident (5.88), and full hedging — plain consensus at 4.21 — beats both.
- **why E2 saw random-level performance** — a random pick is unbiased, an argmax
  pick is *selected*, so argmax lands at or just below random. Measured: 1.582
  versus 1.548.
- **why arm2 fixed it** — with true values there is no error to select on.
- **why consensus is immune** — ADP comes from thousands of real drafters, not
  from our projections, so the program is not selecting on *its own* errors.

### The tension worth understanding

Our player projections are **better than consensus** — measured at +22.4 fantasy
points per pick across 4 seasons, t = 2.60. That result holds up.

And yet using them to *select* is worse than consensus.

These are not contradictory. Being better on average (estimation) is a different
skill from being better at picking the maximum (selection). Averaging cancels
errors out; maximizing hunts them down. The practical consequence: "just improve
the projections" may not be enough. You need better projections **and** a
decision rule that knows they are uncertain.

---

## 8. Two corrections from Matt that changed the analysis

**Running backs early is normal strategy, not a symptom.** I had been citing
"the agent drafts RB three rounds early" as evidence of the bug. The data
disagrees: arm5, which has perfect foresight, drafts RB at mean round 5.5 versus
consensus 7.5 — so RB-early was *correct* here. And the two best arms have
opposite profiles while finishing within 0.4 places of each other. Withdrawn as
evidence.

**Seasons are not independent.** Replaying 10,000 rosters against 2022 gives
10,000 *correlated* readings of one season, not 10,000 observations. Four
channels of correlation matter: the same players recur across years and their
projection errors persist; the league drifts (2022 resembles 2023 far more than
2015, so pooling a decade of history equally would calibrate to a league that no
longer exists); league-wide shocks miss every receiver low together; and every
standard error quoted in this document is a *within-season* one, which is right
for "did this beat consensus in 2022" and badly understates "will this work next
year." **Treat n = 4 as the real sample size for any claim about the future.**

---

## 9. Exactly where we are

### Stage status

| item | state |
|---|---|
| 0a FLEX hindsight fix | done |
| 0b determinism | done |
| 0c common random numbers | done |
| 0d realistic opponents | done |
| 0e ceiling experiment | done — **failed its own bar** |
| 0f gate protocol enforcement | **NOT BUILT** |
| 0g checkpointing | done |
| Stage 1 (GPU port) and all four ML arms | **on hold** |

Stage 0 carried a pre-registered stopping rule: *if the best-case search gains
less than +0.35 places, stop and write it up.* It came in at **-1.667**. The rule
fired. We stopped, and `docs/OBJECTIVE_DIAGNOSIS.md` is the write-up.

So the project is **past the rebuild stage but not into Stage 1**. The work in
front of us is a repair stage the original plan did not have a name for.

### Two things are open and neither is optional

**The league settings are still guesses.** There is no Yahoo credential file, so
everything runs on a placeholder config (`league_id: "PROVISIONAL"`), assuming
12 teams, half-PPR, and a particular roster shape. Roster shape *is* positional
scarcity — if your league runs 3 receivers, or two flex slots, or full PPR, the
valuations move. Every number in this document was computed on a guess. **This
requires you; it needs a browser login.**

**The final test seasons are protected only by discipline.** 2024 and 2025 are
the held-out gate and remain untouched, which is good. But 0f — the code that
was supposed to *refuse* to run against them without a pre-registered
experiment name — was never built. Worth building before anything is tested
there, because the value of a held-out test is exactly the discipline around it.

---

## 10. What happens next

| # | task | why | blocked by |
|---|---|---|---|
| 9 | Measure projection error by position and ADP tier | the central quantity, never yet measured. The whole optimizer's-curse story rests on how wide it is. Cheap: pure data analysis, no simulation | — |
| 10 | Cross-entropy season-transfer matrix | scores how surprised the model is by what really happened. Fits the error width properly, and its decay across seasons gives *measured* weights for how far back history is useful | — |
| 11 | Add shrinkage and pessimism to the decision rule, re-run the ceiling | the actual fix (below) | 9, 10 |
| 8 | Run the robustness harness | built with 9 passing tests, never actually executed | — |

Task 9 is also a genuine test of the diagnosis: if projection error turns out to
be *small* relative to the gaps between candidates, the optimizer's-curse
explanation is wrong and we would want to know that before building task 11.

### The fix, in plain terms

1. **Lean on the market when unsure.** Rank candidates by a blend of our
   projection and ADP, weighted by how much error each carries, instead of by
   our projection alone.
2. **Penalize uncertainty.** Score each candidate as its value *minus* a
   multiple of how uncertain that value is. Standard practice in offline RL —
   CQL and MOPO are published algorithms doing exactly this, for exactly this
   failure.
3. **Let the model know its projections may be off-centre.** Today the simulator
   says "outcomes vary around my projection." It must also say "and my
   projection's centre may itself be wrong, by about this much, and by more for
   some positions than others."

Tasks 9 and 10 measure the numbers that steps 1 and 2 need. Without them, both
are guesses.

**Important caveat:** arm2's +3.312 is a *cheat* — it uses hindsight. It bounds
what perfect valuation would be worth. How much of that is achievable without
hindsight is completely unknown, and re-running the ceiling with the repaired
rule is the only way to find out. The four ML arms stay on hold until it clears.

---

## 11. What to do about the actual 2026 draft

Drafts are roughly three weeks out. The measured answer:

**Use the projections board with `need_adp`, and leave the season-simulation
search switched off.**

- the board beats consensus at projecting: **+22.4 points per pick**, 4 of 4
  seasons, t = 2.60
- the search is measured *worse* than not searching: **-2.594 places**

You take the part that works and skip the part that does not. This also matches
the rule already written in `RUNBOOK.md`: nothing learned replaces `need_adp` on
draft day until it passes the held-out test, and draft day runs locally and
offline.

Do get the real league settings pulled first (section 9).

---

## 12. Where things live

| path | what |
|---|---|
| `src/projections/board.py`, `ecr.py` | builds the player projections. **This is the part that works.** |
| `src/simulation/season.py` | simulates a season and computes U. Where the FLEX bug lived |
| `src/simulation/distributions.py` | player uncertainty model. **Where the fix in section 10 goes** |
| `src/draft/engine.py`, `opponents.py` | the draft itself and the simulated drafters |
| `src/draft/search.py` | the policies: `need_adp`, the season-sim agent, the shortlist |
| `src/evaluation/replay.py` | replays a draft against a real season. The scorekeeper |
| `src/evaluation/objective_check.py` | E2 lives here |
| `src/evaluation/robustness.py` | built, 9 tests, never run (task 8) |
| `src/data/league_config.py` | the placeholder config problem in section 9 |
| `src/data/ffc_adp.py` | real ADP data, added during 0d |
| `draft_day.py` | the offline draft-day tool |
| `docs/OBJECTIVE_DIAGNOSIS.md` | the formal technical write-up of sections 5–8 |
| `RUNBOOK.md` | the operating rules |
| `tests/` | 141 tests, ~5 seconds, run from the repo root |

Legacy from the previous version — `test_final/`, `test_colab_v2/`,
`model_weights/`, `deployment/colab_gpu/`, the assorted root-level
`*_draft_assistant.py` scripts — is not part of the rebuild and is scheduled for
deletion. Do not take anything in those as current.

---

## 13. The one-paragraph version

The projections are good and beat consensus. The season simulator and search
machinery both work. But the program picks by taking the highest-scoring option
among 8 near-identical candidates, and because its projection errors are larger
than the real gaps between those candidates, that maximum mostly selects its own
mistakes — so optimizing harder makes it monotonically worse, and it currently
loses to simply following consensus. The fix is not a better optimizer; it is a
decision rule that accounts for how uncertain its own numbers are. Measuring
that uncertainty is the next step, and the machine-learning work stays parked
until it is done.
