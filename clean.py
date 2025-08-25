# ============================== ONE-CELL DRAFT BOARD PIPELINE ==============================
# ▶️ Set this to your Yahoo ADP CSV path (uploaded to Colab or mounted from Drive)
ADP_PATH = "FantasyPros_2025_Overall_ADP_Rankings.csv"   # <-- EDIT THIS

# 0) Install deps

# 1) Imports & config
import pandas as pd, numpy as np, re, os
from nfl_data_py import import_weekly_data, import_schedules
from rapidfuzz import fuzz, process

# --- League & flex rules (edit roster/scoring if your league differs) ---
LEAGUE = {
    "teams": 12,
    "roster": {"QB":1, "RB":2, "WR":2, "TE":1, "WRT":1, "DEF":1, "K":1, "BENCH":6},
    "scoring": {
        "pass_yd":0.04,"pass_td":4,"int":-1,
        "rush_yd":0.1,"rush_td":6,
        "rec_yd":0.1,"rec_td":6,"rec":0.5,
        # K
        "fgm_0_39":3,"fgm_40_49":4,"fgm_50_plus":5,"fg_miss":-1,"pat_made":1,"pat_miss":-1,
        # DEF/DST events (we’re using PA brackets baseline below)
        "def_td":6,"def_int":2,"def_fum_rec":2,"def_sack":1,"def_safety":2,"def_block":2,"def_ret_td":6,
        # DEF points-allowed brackets (Yahoo-style)
        "pa_0":10,"pa_1_6":7,"pa_7_13":4,"pa_14_20":1,"pa_21_27":0,"pa_28_34":-1,"pa_35_plus":-4
    }
}
FLEX_RULES = {"WRT": {"WR","RB","TE"}}
SEASONS = [2024]  # use last season to build 2025 projections quickly

# 2) Helpers
SUFFIXES = (" jr"," sr"," ii"," iii"," iv"," v")
def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(d/?st|dst|defense|def|special teams)\b", "", s)
    s = re.sub(r"[^a-z\s]", "", s)      # remove punctuation
    for suf in SUFFIXES: s = s.replace(suf, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

TEAM_ABBR_TO_NAME = {
    "ARI":"arizona cardinals","ATL":"atlanta falcons","BAL":"baltimore ravens","BUF":"buffalo bills",
    "CAR":"carolina panthers","CHI":"chicago bears","CIN":"cincinnati bengals","CLE":"cleveland browns",
    "DAL":"dallas cowboys","DEN":"denver broncos","DET":"detroit lions","GB":"green bay packers",
    "HOU":"houston texans","IND":"indianapolis colts","JAX":"jacksonville jaguars","KC":"kansas city chiefs",
    "LV":"las vegas raiders","LAC":"los angeles chargers","LAR":"los angeles rams","MIA":"miami dolphins",
    "MIN":"minnesota vikings","NE":"new england patriots","NO":"new orleans saints","NYG":"new york giants",
    "NYJ":"new york jets","PHI":"philadelphia eagles","PIT":"pittsburgh steelers","SEA":"seattle seahawks",
    "SF":"san francisco 49ers","TB":"tampa bay buccaneers","TEN":"tennessee titans","WAS":"washington commanders"
}

def starting_slots():
    r = LEAGUE["roster"].copy(); r.pop("BENCH", None); return r

def replacement_rank(pos):
    teams = LEAGUE["teams"]; r = starting_slots()
    starters = teams * r.get(pos, 0)
    for slot, count in r.items():
        if slot in FLEX_RULES and pos in FLEX_RULES[slot]:
            starters += int(round(teams * count / len(FLEX_RULES[slot])))
    return max(starters, 1)

def vorp_table(df, proj_col="proj_ppg_2025"):
    out = []
    for pos in ["QB","RB","WR","TE","K","DEF"]:
        sub = df[df["position"]==pos].dropna(subset=[proj_col]).copy()
        if sub.empty: continue
        sub = sub.sort_values(proj_col, ascending=False).reset_index(drop=True)
        rr = min(replacement_rank(pos), len(sub))
        repl = float(sub.loc[rr-1, proj_col])
        sub["VORP"] = sub[proj_col] - repl
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=list(df.columns)+["VORP"])

def def_pa_points(pa, s):
    pa = int(pa) if pd.notna(pa) else 0
    if pa == 0: return s["pa_0"]
    if 1 <= pa <= 6: return s["pa_1_6"]
    if 7 <= pa <= 13: return s["pa_7_13"]
    if 14 <= pa <= 20: return s["pa_14_20"]
    if 21 <= pa <= 27: return s["pa_21_27"]
    if 28 <= pa <= 34: return s["pa_28_34"]
    return s["pa_35_plus"]

def composite_score(q, cand):
    return max(fuzz.WRatio(q, cand), fuzz.token_sort_ratio(q, cand), fuzz.partial_ratio(q, cand))

# 3) Build 2025 projections from 2024 stats (fast baseline)
s = LEAGUE["scoring"]
weekly = import_weekly_data(SEASONS)

def fp_skill(r):
    return (r.get("passing_yards",0)*s["pass_yd"] + r.get("passing_tds",0)*s["pass_td"] + r.get("interceptions",0)*s["int"]
            + r.get("rushing_yards",0)*s["rush_yd"] + r.get("rushing_tds",0)*s["rush_td"]
            + r.get("receiving_yards",0)*s["rec_yd"] + r.get("receiving_tds",0)*s["rec_td"] + r.get("receptions",0)*s["rec"])

def fp_k(r):
    return (r.get("field_goals_made_0_19",0)*s["fgm_0_39"] + r.get("field_goals_made_20_29",0)*s["fgm_0_39"]
            + r.get("field_goals_made_30_39",0)*s["fgm_0_39"] + r.get("field_goals_made_40_49",0)*s["fgm_40_49"]
            + r.get("field_goals_made_50_plus",0)*s["fgm_50_plus"] + r.get("field_goals_missed",0)*s["fg_miss"]
            + r.get("extra_points_made",0)*s["pat_made"] + r.get("extra_points_missed",0)*s["pat_miss"])

wk = weekly.copy()
wk["full_name"] = wk["player_display_name"].fillna(wk["player_name"])
wk["fp_skill"] = wk.apply(fp_skill, axis=1)
wk["fp_k"]     = wk.apply(fp_k, axis=1)

agg_players = (wk.groupby(["season","full_name","position"], as_index=False)
                 .agg(games=("week","nunique"),
                      fp_skill=("fp_skill","sum"),
                      fp_k=("fp_k","sum")))
agg_players["ppg"] = agg_players.apply(lambda r: (r["fp_k"] if r["position"]=="K" else r["fp_skill"]) / max(r["games"],1), axis=1)

sched = import_schedules(SEASONS)
home = sched[["season","week","home_team","away_score"]].rename(columns={"home_team":"team","away_score":"pa"})
away = sched[["season","week","away_team","home_score"]].rename(columns={"away_team":"team","home_score":"pa"})
team_week_pa = pd.concat([home, away], ignore_index=True)
team_week_pa["def_pa_pts"] = team_week_pa["pa"].apply(lambda x: def_pa_points(x, s))

def_agg = (team_week_pa.groupby(["season","team"], as_index=False)
           .agg(games=("week","nunique"), def_pa_pts=("def_pa_pts","sum")))
def_agg["ppg"] = def_agg["def_pa_pts"] / def_agg["games"]
def_agg = def_agg.rename(columns={"team":"full_name"}); def_agg["position"] = "DEF"

last_season = pd.concat([agg_players[["season","full_name","position","ppg"]],
                         def_agg[["season","full_name","position","ppg"]]], ignore_index=True)

proj = last_season[last_season["season"]==SEASONS[-1]].rename(columns={"full_name":"player_name","ppg":"proj_ppg_2025"}).copy()
for pos, shrink in {"K":0.70, "DEF":0.75}.items():
    mu = proj.loc[proj["position"]==pos, "proj_ppg_2025"].mean()
    idx = proj["position"]==pos
    proj.loc[idx, "proj_ppg_2025"] = shrink*proj.loc[idx, "proj_ppg_2025"] + (1-shrink)*mu

projections = proj[["player_name","position","proj_ppg_2025"]].copy()

# 4) Load & clean Yahoo ADP (collapse D/ST→DEF, drop IDP)
IDP_POS = {"D","DL","DB","LB","DT","DE","CB","S"}
DEF_ALIASES = {"DST":"DEF","D/ST":"DEF","DEFENSE":"DEF","DEFENSE/SPECIAL TEAMS":"DEF","DEFENSE SPECIAL TEAMS":"DEF","ST":"DEF"}
adp = pd.read_csv(ADP_PATH)
adp_std = adp.rename(columns={"Player":"player_name","POS":"position","Yahoo":"adp_rank"})
adp_std = adp_std[["player_name","position","adp_rank"]].dropna(subset=["player_name","position"])
adp_std["position"] = adp_std["position"].astype(str).str.upper().str.extract(r"([A-Z/]+)")
adp_std["position"] = adp_std["position"].replace(DEF_ALIASES)
adp_std = adp_std[~adp_std["position"].isin(IDP_POS)]
adp_std["adp_rank"] = pd.to_numeric(adp_std["adp_rank"], errors="coerce")
adp_std["name_key"] = adp_std["player_name"].apply(norm_name)
# Dedup: keep best (lowest) rank per (name_key, position)
adp_uniq = adp_std.sort_values("adp_rank").drop_duplicates(subset=["name_key","position"], keep="first")

# 5) Normalize projection names for merge (map DEF abbr→full team names)
proj_keys = projections.copy()
proj_keys["position"] = proj_keys["position"].astype(str).str.upper()
is_def = proj_keys["position"].eq("DEF")
proj_keys.loc[is_def, "player_name"] = proj_keys.loc[is_def, "player_name"].map(TEAM_ABBR_TO_NAME).fillna(proj_keys.loc[is_def, "player_name"])
proj_keys["name_key"] = proj_keys["player_name"].apply(norm_name)

# 6) Exact merge then fuzzy backfill on leftovers (position-aware)
proj_with_adp = proj_keys.merge(adp_uniq[["name_key","position","adp_rank"]],
                                on=["name_key","position"], how="left")
adp_by_pos = {p: df.reset_index(drop=True) for p, df in adp_uniq.groupby("position", as_index=False)}
need_idx = proj_with_adp.index[proj_with_adp["adp_rank"].isna()].tolist()
for i in need_idx:
    pos = proj_with_adp.at[i, "position"]; cand_df = adp_by_pos.get(pos)
    if cand_df is None or cand_df.empty: continue
    q = norm_name(proj_with_adp.at[i, "player_name"])
    names = cand_df["player_name"].tolist()
    cands = process.extract(q, names, scorer=fuzz.WRatio, limit=min(50, len(names)))
    best_score, best_rank = 0, np.nan
    for cand_name, wr, idx in cands:
        score = composite_score(q, norm_name(cand_name))
        if score >= 88 or score >= 70:
            rank = cand_df.at[idx, "adp_rank"]
            if pd.isna(rank): continue
            if (score > best_score) or (score == best_score and rank < best_rank):
                best_score, best_rank = score, rank
    if pd.isna(best_rank) and cands and cands[0][1] >= 62:
        best_rank = cand_df.at[cands[0][2], "adp_rank"]
    proj_with_adp.at[i, "adp_rank"] = best_rank

# 7) Build initial board (VORP)
board_core = vorp_table(proj_with_adp[["player_name","position","proj_ppg_2025"]], "proj_ppg_2025")
right_df = proj_with_adp[["player_name","position","adp_rank"]].drop_duplicates(subset=["player_name","position"])
board = (board_core.merge(right_df, on=["player_name","position"], how="left")
                    .sort_values(["VORP","proj_ppg_2025","adp_rank"], ascending=[False, False, True])
                    .reset_index(drop=True))

# 8) Rookie projection fill (ADP-anchored), then recompute VORP
def fill_rookies_with_adp_anchor(board_df, positions=("QB","RB","WR","TE")):
    b = board_df.copy()
    for pos in positions:
        vets = b[(b["position"]==pos) & b["proj_ppg_2025"].notna()].sort_values("proj_ppg_2025", ascending=False)
        if vets.empty: continue
        rank_to_ppg = dict(zip(range(1, len(vets)+1), vets["proj_ppg_2025"]))
        pos_mean = vets["proj_ppg_2025"].mean()
        need = b[(b["position"]==pos) & (b["proj_ppg_2025"].isna()) & (b["adp_rank"].notna())].index
        for i in need:
            pos_rank = int(min(max(1, b.at[i,"adp_rank"]), len(vets)))  # clamp into range
            est = rank_to_ppg.get(pos_rank, pos_mean)
            b.at[i, "proj_ppg_2025"] = 0.7*est + 0.3*pos_mean
    return b

board = fill_rookies_with_adp_anchor(board)
board = vorp_table(board[["player_name","position","proj_ppg_2025"]], "proj_ppg_2025") \
            .merge(board[["player_name","position","adp_rank"]].drop_duplicates(), on=["player_name","position"], how="left") \
            .sort_values(["VORP","proj_ppg_2025","adp_rank"], ascending=[False, False, True]).reset_index(drop=True)

# 9) Show DEF rows for verification
board_defs = (board[board["position"]=="DEF"][["player_name","proj_ppg_2025","VORP","adp_rank"]]
              .drop_duplicates().sort_values("adp_rank", na_position="last").reset_index(drop=True))
print("Total defenses in final board:", len(board_defs))
print(board_defs.to_string(index=False))

# 10) Save CSV
board_out = board[["player_name","position","proj_ppg_2025","VORP","adp_rank"]].copy()
board_out.to_csv("draft_board.csv", index=False)
print("\nSaved to:", os.path.abspath("draft_board.csv"))

# 11) Optional: peek top 20 overall
print("\nTop 20 overall by VORP/PPG/ADP blend:")
print(board.head(20)[["player_name","position","proj_ppg_2025","VORP","adp_rank"]].to_string(index=False))
# ==========================================================================================
