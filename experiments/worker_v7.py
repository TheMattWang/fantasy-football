"""V7 -- what does the HEAD-TO-HEAD structure actually reward?

In H2H you do not need the most points. You need to beat one specific roster
each week. Winning 150-100 counts exactly as much as winning 101-100, so points
above your opponent's score are thrown away. That is a different objective from
"maximize expected points", and nothing in this project has ever measured how
different.

Three questions, in increasing order of how much they matter.

  A. HOW MUCH DOES H2H DISTORT? Rank the same 12 teams by total points and by
     H2H record. If those orderings agree closely, H2H is a thin veneer over
     total points and can be ignored. If they diverge, schedule luck is a real
     component of finishing position and every rank-based measurement in this
     project carries that noise.

  B. DOES VARIANCE HELP OR HURT? I claimed, analytically and without measuring
     it, that added variance pulls a favorite toward a coin flip and therefore
     hurts a good team, while helping a bad one. That claim is load-bearing for
     the stacking idea and it has never been tested. Test it: hold a team's mean
     weekly score fixed, sweep its standard deviation, and read off P(playoffs),
     P(title) and mean rank -- at three different mean levels.

  C. WHAT DOES THE MISSING BRACKET COST? season.py does not simulate the
     playoffs at all; it draws a champion with probability proportional to
     1/seed. A real title is won in a two-to-three week single-elimination
     bracket, which is exactly the high-variance event where a boom roster is
     supposed to pay. So run B twice -- once under the 1/seed lottery the
     simulator uses, once under a real bracket over the same weekly scores --
     and the difference is what the simplification costs.

Scores are Gamma, matching the shape distributions.py already uses for weekly
points, parameterized so that mean is held EXACTLY constant while cv varies.
Gamma(shape=1/cv^2, scale=mu*cv^2) has mean mu for every cv, which is what makes
the variance sweep clean: nothing moves except the spread.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np

from src.data.league_config import provisional_config
from src.simulation.season import round_robin_schedule, simulate_league

CFG = provisional_config()
N_TEAMS = 12
REG_WEEKS = 14
PLAYOFF_WEEKS = 3
N_SAMPLES = 20000
BASE_MEAN = 107.0        # ~1500 points over 14 weeks, matching the replay data
BASE_CV = 0.22
N_PLAYOFF = 6


def gamma_scores(rng, mean, cv, shape):
    """Weekly scores with EXACTLY `mean` regardless of cv."""
    k = 1.0 / (cv ** 2)
    return rng.gamma(shape=k, scale=mean * (cv ** 2), size=shape).astype(np.float32)


def league(rng, my_mean, my_cv, weeks):
    scores = gamma_scores(rng, BASE_MEAN, BASE_CV, (N_TEAMS, N_SAMPLES, weeks))
    scores[0] = gamma_scores(rng, my_mean, my_cv, (N_SAMPLES, weeks))
    return scores


def standings(scores):
    """Regular-season wins, points, rank. Same rule as season.py: wins first,
    points-for as the tiebreak."""
    reg = scores[:, :, :REG_WEEKS]
    schedule = round_robin_schedule(N_TEAMS, REG_WEEKS)
    wins = np.zeros((N_TEAMS, N_SAMPLES), dtype=np.int32)
    for week in range(REG_WEEKS):
        opp = schedule[week]
        wins += (reg[:, :, week] > reg[opp, :, week]).astype(np.int32)
    points = reg.sum(axis=2)
    key = wins.astype(np.float64) * 1e6 + points
    order = np.argsort(-key, axis=0)              # order[0] = the 1 seed
    rank = np.empty_like(order)
    rows = np.arange(N_TEAMS)[:, None]
    np.put_along_axis(rank, order, np.broadcast_to(rows, order.shape), axis=0)
    return wins, points, rank + 1, order


def bracket_champion(scores, order):
    """Real single-elimination playoff over weeks 15-17.

    Seeds 1-2 get a bye. Week 15: 3v6 and 4v5. Week 16: the 1 seed plays the
    WORSE surviving seed, the 2 seed plays the other. Week 17: the final.
    """
    w15, w16, w17 = (scores[:, :, REG_WEEKS + i] for i in range(PLAYOFF_WEEKS))

    def team_score(week, teams):
        return np.take_along_axis(week, teams[None, :], axis=0)[0]

    s = [order[i] for i in range(N_PLAYOFF)]      # s[0] = 1 seed, ... s[5] = 6 seed

    # Week 15 -- seeds 3v6 and 4v5. Seed index is position in `s`.
    a_win = np.where(team_score(w15, s[2]) > team_score(w15, s[5]), 2, 5)
    b_win = np.where(team_score(w15, s[3]) > team_score(w15, s[4]), 3, 4)
    a_team = np.choose(a_win == 2, [s[5], s[2]])
    b_team = np.choose(b_win == 3, [s[3], s[4]])

    # Week 16 -- reseed: the 1 seed draws the worse survivor.
    a_is_worse = a_win > b_win
    low = np.where(a_is_worse, a_team, b_team)
    high = np.where(a_is_worse, b_team, a_team)

    f1 = np.where(team_score(w16, s[0]) > team_score(w16, low), s[0], low)
    f2 = np.where(team_score(w16, s[1]) > team_score(w16, high), s[1], high)

    return np.where(team_score(w17, f1) > team_score(w17, f2), f1, f2)


def lottery_champion(rng, rank):
    """What season.py actually does: sample proportional to 1/seed."""
    made = rank <= N_PLAYOFF
    weight = np.where(made, 1.0 / rank, 0.0)
    weight /= weight.sum(axis=0, keepdims=True)
    draws = rng.random(N_SAMPLES)
    return (np.cumsum(weight, axis=0) < draws[None, :]).sum(axis=0)


# --- A. how much does H2H distort the points ordering? --------------------

rng = np.random.default_rng(0)
scores = league(rng, BASE_MEAN, BASE_CV, REG_WEEKS + PLAYOFF_WEEKS)
wins, points, rank, order = standings(scores)

points_rank = np.argsort(np.argsort(-points, axis=0), axis=0) + 1
agree = np.mean([
    np.corrcoef(rank[:, s], points_rank[:, s])[0, 1] for s in range(2000)
])
top_points_wins_league = float(np.mean(rank[np.argmax(points, axis=0),
                                            np.arange(N_SAMPLES)] == 1))
top_points_makes_playoffs = float(np.mean(rank[np.argmax(points, axis=0),
                                               np.arange(N_SAMPLES)] <= N_PLAYOFF))

# All-play: how many of the other 11 you would have beaten each week. This is
# the schedule-luck-free record, so actual minus all-play IS the luck.
reg = scores[:, :, :REG_WEEKS]
allplay = np.zeros((N_TEAMS, N_SAMPLES))
for week in range(REG_WEEKS):
    col = reg[:, :, week]
    allplay += (col[:, None, :] > col[None, :, :]).sum(axis=1)
allplay_wins = allplay / (N_TEAMS - 1)
luck = wins - allplay_wins

print("V7  WHAT DOES THE HEAD-TO-HEAD STRUCTURE REWARD?")
print(f"  12 teams, {REG_WEEKS} weeks, {N_SAMPLES:,} simulated seasons, all teams identical\n")
print("A. HOW MUCH DOES H2H DISTORT THE POINTS ORDERING?")
print(f"   mean Spearman(H2H rank, total-points rank)   {agree:+.3f}")
print(f"   the highest-scoring team wins the league     {top_points_wins_league:.1%}")
print(f"   the highest-scoring team makes the playoffs  {top_points_makes_playoffs:.1%}")
print(f"   schedule luck, sd of (actual - all-play wins) {luck.std():.2f} wins")
print(f"   -- so roughly +/-{1.96*luck.std():.1f} wins of a {REG_WEEKS}-game season is who you drew\n")

# --- B/C. does variance help, and what does the bracket change? -----------

print("B/C. DOES VARIANCE HELP OR HURT, AT FIXED MEAN?")
print("   Mean weekly score held EXACTLY constant; only the spread moves.")
print("   'lottery' = P(title) as season.py computes it (1/seed draw).")
print("   'bracket' = P(title) under a real single-elimination playoff.\n")

for label, mult in (("underdog  (-8%)", 0.92),
                    ("average   ( 0%)", 1.00),
                    ("favourite (+8%)", 1.08)):
    print(f"   {label}")
    print(f"     {'cv':>6}{'sd':>7}{'P(playoff)':>12}{'mean rank':>11}"
          f"{'P(title) lot':>14}{'P(title) brk':>14}")
    for cv in (0.12, 0.18, 0.22, 0.30, 0.40):
        r = np.random.default_rng(1234)
        sc = league(r, BASE_MEAN * mult, cv, REG_WEEKS + PLAYOFF_WEEKS)
        _, _, rk, od = standings(sc)
        champ_b = bracket_champion(sc, od)
        champ_l = lottery_champion(r, rk)
        print(f"     {cv:>6.2f}{BASE_MEAN*mult*cv:>7.1f}"
              f"{np.mean(rk[0] <= N_PLAYOFF):>12.3f}{np.mean(rk[0]):>11.2f}"
              f"{np.mean(champ_l == 0):>14.3f}{np.mean(champ_b == 0):>14.3f}")
    print()

print("   Read: if P(title) rises with cv for the underdog and falls for the")
print("   favourite, the analytic claim holds. If the bracket column responds")
print("   to variance while the lottery column does not, the simulator is")
print("   structurally blind to the payoff that stacking is supposed to buy.")
