"""
Dixon-Coles Poisson model for scoreline distribution prediction.

Reads:  ../Datasets/Cleaned Datasets/*.csv
Writes: ../Datasets/Predictions/scorelines_*.csv

Each output row = one upcoming fixture with:
  - Home Win %, Draw %, Away Win %
  - xG Home, xG Away
  - p_i_j columns: probability (%) for every scoreline 0-0 .. 5-5
"""

import os
import json
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

DECAY_RATE = 0.0020   # exponential time-weight — half-life ~347 days.
                      # Fitted by tune_decay.py against held-out log loss across
                      # England, Italy and Spain, 3,420 matches. The previous
                      # 0.0065 (~107 days) was reasoned, never measured, and
                      # ranked 6th of 8 candidates: a 107-day half-life discards
                      # most of a season, and team strength persists longer than
                      # that. Worth -0.0104 log loss, about a quarter of the gap
                      # to the bookmakers' closing price.
MAX_GOALS  = 8        # internal grid; outputs capped at 5 for CSV columns


# ── Calibration ───────────────────────────────────────────────────────────────
# Raw Dixon-Coles output is overconfident at the extremes: backtesting showed it
# saying 85% where the real frequency was 69%, and 6% where it was 13%. The fix
# is a single temperature applied to the scoreline grid, m ** (1/T) renormalised.
#
# Applying it to the grid rather than to the 1X2 marginals keeps every derived
# number consistent — win/draw/loss, BTTS, over/under and the score matrix all
# fall out of the same tempered distribution. T > 1 flattens.
#
# The value is fitted by calibrate.py on held-out matches and cached; 1.0 means
# no adjustment, so an absent file leaves behaviour unchanged.

_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "Datasets", "Models", "calibration.json")


def _load_temperature():
    try:
        with open(_CAL_PATH) as f:
            return float(json.load(f).get("temperature", 1.0))
    except Exception:
        return 1.0


TEMPERATURE = _load_temperature()


def temper(m, temperature=None):
    """Flatten a probability grid toward uniform. T=1 returns it untouched."""
    t = TEMPERATURE if temperature is None else temperature
    if t == 1.0:
        return m
    out = np.clip(m, 1e-300, None) ** (1.0 / t)
    return out / out.sum()


# ── Dixon-Coles correction ────────────────────────────────────────────────────

def _dc_tau(hg, ag, lam, mu, rho):
    """Vectorised low-score correction factor (Dixon & Coles, 1997)."""
    tau = np.ones(len(hg))
    tau[(hg == 0) & (ag == 0)] = 1 - lam[(hg == 0) & (ag == 0)] * mu[(hg == 0) & (ag == 0)] * rho
    tau[(hg == 1) & (ag == 0)] = 1 + mu[(hg == 1) & (ag == 0)] * rho
    tau[(hg == 0) & (ag == 1)] = 1 + lam[(hg == 0) & (ag == 1)] * rho
    tau[(hg == 1) & (ag == 1)] = 1 - rho
    return tau


def _neg_log_lik(params, n, home_idx, away_idx, hg, ag, weights):
    # params: attack[0..n-2] (team n-1 pinned to 0), defense[0..n-1], home_adv, rho
    attack   = np.append(params[: n - 1], 0.0)
    defense  = params[n - 1 : 2 * n - 1]
    home_adv = params[-2]
    rho      = params[-1]

    lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    mu  = np.exp(attack[away_idx] + defense[home_idx])

    tau = _dc_tau(hg, ag, lam, mu, rho)
    if np.any(tau <= 0):
        return 1e10

    ll = weights * (
        np.log(tau)
        + poisson.logpmf(hg, lam)
        + poisson.logpmf(ag, mu)
    )
    return -ll.sum()


# ── Model ─────────────────────────────────────────────────────────────────────

class DixonColesModel:

    def __init__(self, decay_rate=DECAY_RATE):
        self.decay_rate = decay_rate
        self.teams_    = None
        self._idx      = None   # team → integer index
        self.attack_   = None
        self.defense_  = None
        self.home_adv_ = None
        self.rho_      = None

    def fit(self, matches: pd.DataFrame):
        """
        matches columns required:
            date (datetime-like), home_team (str), away_team (str),
            home_goals (numeric), away_goals (numeric)
        Only rows with non-null, whole-number goals are used for training.
        """
        df = matches.copy()
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
        df["date"]       = pd.to_datetime(df["date"], errors="coerce")

        # Keep only rows where goals look like real results (whole numbers)
        df = df.dropna(subset=["home_goals", "away_goals", "date"])
        df = df[(df["home_goals"] % 1 == 0) & (df["away_goals"] % 1 == 0)]
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        # Limit to last 5 years — older data has near-zero time-weight anyway
        # and including stale teams inflates the parameter space unnecessarily
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
        df     = df[df["date"] >= cutoff]

        self.teams_ = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._idx   = {t: i for i, t in enumerate(self.teams_)}
        n           = len(self.teams_)

        home_idx = df["home_team"].map(self._idx).values
        away_idx = df["away_team"].map(self._idx).values
        hg       = df["home_goals"].values
        ag       = df["away_goals"].values

        days_ago = (pd.Timestamp.now() - df["date"]).dt.days.values.clip(min=0)
        weights  = np.exp(-self.decay_rate * days_ago)

        x0       = np.zeros(2 * n - 1 + 2)
        x0[-2]   = 0.1    # home_adv: exp(0.1) ≈ 1.1 — slight home edge
        x0[-1]   = -0.1   # rho: small negative correction for low scores

        # Bounds prevent numerical blow-up in rho and extreme attack/defense values
        bounds = (
            [(-4.0,  4.0)] * (n - 1)   # attack (reference team pinned to 0)
            + [(-4.0, 4.0)] * n         # defense
            + [(-1.5, 1.5)]             # home_adv
            + [(-0.5, 0.4)]             # rho — must keep tau > 0 for realistic λ/μ
        )

        result = minimize(
            _neg_log_lik,
            x0,
            args=(n, home_idx, away_idx, hg, ag, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-12},
        )

        p = result.x
        self.attack_   = np.append(p[: n - 1], 0.0)
        self.defense_  = p[n - 1 : 2 * n - 1]
        self.home_adv_ = p[-2]
        self.rho_      = p[-1]
        return self

    def _params(self, team, allow_unknown=False):
        """
        Attack and defence for a team, falling back to the league average.

        A promoted club has no history in this division, so it has no rating.
        Refusing to rate it is right for a fixture page — better to say nothing
        than to invent a number — but a season simulation cannot skip its
        matches without handing it a guaranteed last place, so it is rated as an
        average side and flagged as such.
        """
        i = self._idx.get(team)
        if i is None:
            if not allow_unknown:
                raise KeyError(team)
            return float(self.attack_.mean()), float(self.defense_.mean())
        return float(self.attack_[i]), float(self.defense_[i])

    def _lambda_mu(self, home_team, away_team, allow_unknown=False):
        ha, hd = self._params(home_team, allow_unknown)
        aa, ad = self._params(away_team, allow_unknown)
        lam = np.exp(ha + ad + self.home_adv_)
        mu  = np.exp(aa + hd)
        return float(lam), float(mu)

    def scoreline_matrix(self, home_team, away_team, allow_unknown=False):
        """(MAX_GOALS+1) × (MAX_GOALS+1) probability matrix. m[i,j] = P(home=i, away=j)."""
        lam, mu = self._lambda_mu(home_team, away_team, allow_unknown)
        g       = np.arange(MAX_GOALS + 1)
        m       = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))

        # Clip each tau to a small positive floor so a boundary rho never
        # produces a negative cell (which would corrupt the whole matrix).
        m[0, 0] *= max(1e-6, 1 - lam * mu * self.rho_)
        m[1, 0] *= max(1e-6, 1 + mu  * self.rho_)
        m[0, 1] *= max(1e-6, 1 + lam * self.rho_)
        m[1, 1] *= max(1e-6, 1 - self.rho_)

        m /= m.sum()   # renormalise — corrections shift the sum slightly
        return temper(m)

    def predict(self, home_team, away_team, allow_unknown=False):
        known = home_team in self._idx and away_team in self._idx
        if not known and not allow_unknown:
            return None

        m   = self.scoreline_matrix(home_team, away_team, allow_unknown)
        lam, mu = self._lambda_mu(home_team, away_team, allow_unknown)

        g           = np.arange(MAX_GOALS + 1)
        total_goals = g[:, None] + g[None, :]

        return {
            "rated":    known,
            "home_win": float(np.tril(m, -1).sum()),
            "draw":     float(np.trace(m)),
            "away_win": float(np.triu(m, 1).sum()),
            "xg_home":  lam,
            "xg_away":  mu,
            "btts":     float(m[1:, 1:].sum()),
            "over_1_5": float(m[total_goals > 1].sum()),
            "over_2_5": float(m[total_goals > 2].sum()),
            "over_3_5": float(m[total_goals > 3].sum()),
            "matrix":   m,
        }


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_matches(path: str, min_home_games: int = 10) -> pd.DataFrame:
    """
    Read a cleaned CSV and return one row per unique match (home-perspective).

    FBREF Scores & Fixtures tables include European/cup games alongside league
    games, so non-league opponents end up in the CSV.  We keep only teams that
    appear as the home side at least `min_home_games` times — a genuine league
    participant plays 15–19 home games per season, a one-off cup opponent plays 1.
    """
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["gf"]   = pd.to_numeric(df["gf"], errors="coerce")
    df["ga"]   = pd.to_numeric(df["ga"], errors="coerce")

    # League games are numbered "Matchweek N"; cup and continental games carry
    # named rounds ("Group stage", "Round of 16").  Matching on the round rather
    # than the competition name survives league renames — Norway's Tippeligaen
    # became Eliteserien in 2017 and both are genuine league seasons.
    if "round" in df.columns:
        is_league = df["round"].astype(str).str.strip().str.startswith("Matchweek")
        dropped   = int((~is_league).sum())
        if dropped:
            print(f"   dropped {dropped} cup/continental rows")
        df = df[is_league]

    home = df[df["venue"].str.strip().str.lower() == "home"].copy()
    home = home.rename(columns={
        "team": "home_team", "opponent": "away_team",
        "gf":   "home_goals", "ga":      "away_goals",
    })

    # Keep only teams that regularly appear as the home side (league members)
    counts       = home["home_team"].value_counts()
    league_teams = set(counts[counts >= min_home_games].index)
    home = home[
        home["home_team"].isin(league_teams)
        & home["away_team"].isin(league_teams)
    ]

    return home[["date", "home_team", "away_team", "home_goals", "away_goals"]]


def upcoming_fixtures(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Return fixtures that have no recorded result AND are in the future.

    Old matches can also have null goals (scraping gaps), so we restrict to the
    most recent 12-month window, and keep today's matches.

    football-data publishes dates without kickoff times, so a fixture today is
    stored at midnight. Requiring date > today therefore dropped the entire
    current matchday — Bundesliga 2 lost three of its four games.
    """
    today        = pd.Timestamp("today").normalize()
    max_date     = matches["date"].dropna().max()
    season_start = max_date - pd.DateOffset(months=12)
    recent       = matches[(matches["date"] >= season_start) & (matches["date"] >= today)]

    no_result = (
        recent["home_goals"].isna()
        | recent["away_goals"].isna()
        | (recent["home_goals"] % 1 != 0)
        | (recent["away_goals"] % 1 != 0)
    )
    return recent[no_result][["date", "home_team", "away_team"]].drop_duplicates()


def _fixture_dates(league_key):
    """Cached ESPN dates for one league, keyed 'home|away'. Empty if unknown."""
    if not league_key:
        return {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Datasets", "Models", "fixture_dates.json")
    try:
        with open(path) as f:
            return json.load(f).get("leagues", {}).get(league_key, {})
    except Exception:
        return {}


def _double_round_robin(df, cur, teams, dates, tol=0.02, max_gap=0.20):
    """
    True when this league demonstrably plays every ordered pair exactly once.

    Measured from the data, never assumed. Assuming it once fabricated a title
    race for Greece, so there are two tests and both must pass.

    The first is history: in each of the last few completed seasons every club
    played 2(n-1) games. That identifies the format but trusts the current
    member list, and the member list is exactly what goes wrong — China's
    newest season lists 28 clubs where every previous season had 16, and a
    28-club double round-robin would invent five hundred fixtures.

    So the second test is arithmetic against the calendar: how much of the
    season has elapsed, against how much of n(n-1) has been played. A league
    four-fifths through its year with two-fifths of the meetings played is not
    playing the format it appears to play, whatever the reason.
    """
    n = len(teams)
    if n < 4:
        return False

    home = df['venue'].astype(str).str.lower().str.strip() == 'home'
    seasons = sorted(df['season'].dropna().astype(str).unique())

    ratios = []
    for s in seasons[:-1][-5:]:
        past = df[(df['season'].astype(str) == s) & home]
        past = past[past['gf'].notna() & past['ga'].notna()]
        if len(past) < 50:
            continue
        members = set(past['team'].astype(str)) | set(past['opponent'].astype(str))
        if len(members) < 4:
            continue
        per_team = pd.concat([past['team'], past['opponent']]).value_counts()
        ratios.append(per_team.median() / (2 * (len(members) - 1)))

    # No completed season to learn from is not evidence of a round-robin.
    if not ratios or any(abs(r - 1.0) > tol for r in ratios):
        return False

    played = cur[cur['gf'].notna() & cur['ga'].notna()]
    start  = pd.to_datetime(cur['date'], errors='coerce').min()
    ends   = [pd.to_datetime(cur['date'], errors='coerce').max()]
    for when in dates.values():
        w = pd.to_datetime(when, errors='coerce')
        if pd.notna(w):
            ends.append(w)
    ends = [e for e in ends if pd.notna(e)]
    if pd.isna(start) or not ends:
        return False
    end = max(ends)
    if end <= start:
        return False

    today   = pd.Timestamp.now().normalize()
    elapsed = min(max((today - start) / (end - start), 0.0), 1.0)
    return elapsed - len(played) / (n * (n - 1)) <= max_gap


def remaining_pairings(cleaned_path, scheduled, league_key=None):
    """
    Every fixture still to be played this season, including the ones no
    calendar has been published for yet.

    football-data publishes only the next matchweek, so a league in September
    shows nine fixtures when three hundred are still to come. A double
    round-robin plays each ordered pair exactly once, so the rest of the season
    is simply the pairings that have not happened — known with certainty even
    though the dates are not.

    Returns them with date NaT, which is honest: the meeting is scheduled, the
    day is not yet. Leagues that do not play a straight round-robin are excluded
    upstream, because for them this arithmetic invents fixtures.
    """
    df = pd.read_csv(cleaned_path, index_col=0, low_memory=False)
    if 'season' not in df.columns:
        return pd.DataFrame(columns=['date', 'home_team', 'away_team'])

    latest = sorted(df['season'].dropna().astype(str).unique())[-1]
    cur = df[(df['season'].astype(str) == latest)
             & (df['venue'].astype(str).str.lower().str.strip() == 'home')]
    if cur.empty:
        return pd.DataFrame(columns=['date', 'home_team', 'away_team'])

    teams  = sorted(set(cur['team'].astype(str)) | set(cur['opponent'].astype(str)))
    played = cur[cur['gf'].notna() & cur['ga'].notna()]
    done   = {(str(r['team']), str(r['opponent'])) for _, r in played.iterrows()}

    # Anything already on the calendar keeps its date; don't duplicate it here
    booked = {(str(h), str(a)) for h, a in
              zip(scheduled.get('home_team', []), scheduled.get('away_team', []))}

    dates = _fixture_dates(league_key)

    # football-data lags the calendar by days: a match already staged may still
    # be missing a result here. Anything before today is genuinely history and
    # is dropped; today itself is kept in full.
    today = pd.Timestamp.now().normalize()
    rows = []
    listed = set()

    if dates:
        # ESPN publishes the actual calendar, which beats deriving one. It also
        # works for leagues that are not a straight round-robin — championship
        # splits, Apertura/Clausura, conference schedules — where the pairings
        # arithmetic would invent fixtures that never happen.
        #
        # A future kickoff is the whole test. Membership of the current season
        # is not: where football-data's newest season is last year's completed
        # one, its team list is the wrong squad and every pairing already counts
        # as played, which is how Scotland ended up with nothing to show.
        for key, when in dates.items():
            h, _, a = key.partition('|')
            if not h or not a or (h, a) in booked:
                continue
            w = pd.to_datetime(when, errors='coerce')
            # Keep a match for the whole of its day. Dropping it the moment
            # kick-off passes empties the current matchday as the afternoon
            # goes on, and the model's line for a game in progress is still
            # what the model said — the result simply has not landed yet.
            if pd.isna(w) or w.normalize() < today:
                continue
            rows.append({'date': w, 'home_team': h, 'away_team': a})
            listed.add((h, a))

        # A calendar is not always a whole season. ESPN and FBref publish what
        # has been announced, which in September is often only the autumn: 2.
        # Bundesliga came back with 268 of its 306 meetings, and the 38 that
        # were missing appeared nowhere at all — not played, not predicted.
        # Across the leagues that could be checked, 637 fixtures had silently
        # gone missing this way.
        #
        # Where the format is known to be a straight double round-robin, the
        # rest of the season is arithmetic, so fill the gap with dateless
        # pairings. Where it is not, leave the gap alone: inventing fixtures is
        # the worse failure of the two.
        if _double_round_robin(df, cur, teams, dates):
            for h, a in itertools.permutations(teams, 2):
                if (h, a) in done or (h, a) in booked or (h, a) in listed:
                    continue
                rows.append({'date': pd.NaT, 'home_team': h, 'away_team': a})
        return pd.DataFrame(rows, columns=['date', 'home_team', 'away_team'])

    # No published calendar: fall back to the round-robin pairings, dateless.
    for h, a in itertools.permutations(teams, 2):
        if (h, a) not in done and (h, a) not in booked:
            rows.append({'date': pd.NaT, 'home_team': h, 'away_team': a})
    return pd.DataFrame(rows, columns=['date', 'home_team', 'away_team'])


# ── Per-league runner ─────────────────────────────────────────────────────────

def run_league(cleaned_path: str, output_path: str, league_name: str,
               include_unscheduled: bool = False, league_key: str = None):
    print(f"\n-- {league_name}")

    matches   = load_matches(cleaned_path)
    completed = matches[
        matches["home_goals"].notna()
        & matches["away_goals"].notna()
        & (matches["home_goals"] % 1 == 0)
        & (matches["away_goals"] % 1 == 0)
    ]

    print(f"   {len(completed)} completed matches for training")

    if len(completed) < 50:
        print(f"   Too few matches — skipping")
        return

    model    = DixonColesModel()
    model.fit(completed)

    # Save model parameters so the web simulator can use them
    models_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(output_path)), '..', 'Models'))
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.basename(output_path).replace('predictions_', 'model_').replace('.csv', '.json')
    with open(os.path.join(models_dir, model_file), 'w') as f:
        json.dump({
            'teams':    sorted(model.teams_),
            'attack':   {t: float(model.attack_[i]) for t, i in model._idx.items()},
            'defense':  {t: float(model.defense_[i]) for t, i in model._idx.items()},
            'home_adv': float(model.home_adv_),
            'rho':      float(model.rho_),
        }, f)
    print(f"   Model saved  -> {model_file}")

    fixtures = upcoming_fixtures(matches)
    fixtures = fixtures.assign(_sched=True)
    n_sched  = len(fixtures)

    if include_unscheduled:
        extra = remaining_pairings(cleaned_path, fixtures, league_key)
        if not extra.empty:
            fixtures = pd.concat([fixtures, extra.assign(_sched=False)],
                                 ignore_index=True)
    print(f"   {n_sched} scheduled"
          + (f" + {len(fixtures) - n_sched} unscheduled" if len(fixtures) > n_sched else "")
          + " fixtures")

    if fixtures.empty:
        print(f"   No upcoming fixtures to predict")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"   Removed stale predictions file")
        return

    rows, skipped = [], 0
    for _, row in fixtures.iterrows():
        pred = model.predict(row["home_team"], row["away_team"],
                             allow_unknown=include_unscheduled)
        if pred is None:
            skipped += 1
            continue

        m = pred["matrix"]

        scoreline_cols = {
            f"p_{i}_{j}": round(float(m[i, j]) * 100, 2)
            for i in range(6) for j in range(6)
        }

        top3 = sorted(scoreline_cols.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(
            f'{c.replace("p_","").replace("_","-")} ({v:.1f}%)'
            for c, v in top3
        )

        rows.append({
            "Date":              row["date"].date() if pd.notna(row["date"]) else None,
            # A blank date means the pairing is certain but the calendar is not
            # out yet, so the page can separate confirmed fixtures from the rest
            # of the season instead of showing them as one undifferentiated list.
            "Scheduled":         bool(row.get("_sched", pd.notna(row["date"]))),
            # False when a side has no history in this division — the line is
            # then an average-team guess, not a rating, and says so.
            "Rated":             bool(pred.get("rated", True)),
            "Home Team":         row["home_team"],
            "Away Team":         row["away_team"],
            "Home Win %":        round(pred["home_win"] * 100, 1),
            "Draw %":            round(pred["draw"]      * 100, 1),
            "Away Win %":        round(pred["away_win"]  * 100, 1),
            "xG Home":           round(pred["xg_home"],  2),
            "xG Away":           round(pred["xg_away"],  2),
            "BTTS %":            round(pred["btts"]     * 100, 1),
            "Over 1.5 %":        round(pred["over_1_5"] * 100, 1),
            "Over 2.5 %":        round(pred["over_2_5"] * 100, 1),
            "Over 3.5 %":        round(pred["over_3_5"] * 100, 1),
            "Most Likely Score": top3[0][0].replace("p_","").replace("_","-"),
            "Top 3 Scores":      top3_str,
            **scoreline_cols,
        })

    if skipped:
        print(f"   Skipped {skipped} fixtures (teams not seen in training)")

    if not rows:
        print(f"   No predictions generated")
        return

    out = pd.DataFrame(rows).sort_values("Date")
    out.to_csv(output_path, index=False)
    print(f"   Saved {len(out)} predictions -> {os.path.basename(output_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

LEAGUES = [
    ("Brazil_Serie_A.csv",        "predictions_brazil_serie_a.csv",    "Brazil Serie A"),
    ("Norway_Eliteserien.csv",    "predictions_norway_eliteserien.csv","Eliteserien (Norway)"),
    ("Finland_Veikkausliiga.csv", "predictions_finland_veikkausliiga.csv", "Veikkausliiga (Finland)"),
    ("Greece_Super_League.csv",   "predictions_greece_super_league.csv","Super League Greece"),
    ("England_Premier_League.csv","predictions_england_premier_league.csv","Premier League"),
    ("Italy_Serie_A.csv",         "predictions_italy_serie_a.csv",     "Serie A Italy"),
    ("Spain_La_Liga.csv",         "predictions_spain_la_liga.csv",     "La Liga"),
    ("Germany_Bundesliga.csv",    "predictions_germany_bundesliga.csv","Bundesliga"),
    ("France_Ligue_1.csv",        "predictions_france_ligue_1.csv",    "Ligue 1"),
]

if __name__ == "__main__":
    cleaned_dir     = "../Datasets/Cleaned Datasets/"
    predictions_dir = "../Datasets/Predictions/"

    for cleaned_file, output_file, name in LEAGUES:
        cleaned_path = os.path.join(cleaned_dir, cleaned_file)
        output_path  = os.path.join(predictions_dir, output_file)

        if not os.path.exists(cleaned_path):
            print(f"\n  {cleaned_path} not found — skipping {name}")
            continue

        run_league(cleaned_path, output_path, name)

    print("\n Done.")