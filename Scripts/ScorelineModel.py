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
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

DECAY_RATE = 0.0065   # exponential time-weight — half-life ~107 days
MAX_GOALS  = 8        # internal grid; outputs capped at 5 for CSV columns


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

    def _lambda_mu(self, home_team, away_team):
        hi  = self._idx[home_team]
        ai  = self._idx[away_team]
        lam = np.exp(self.attack_[hi] + self.defense_[ai] + self.home_adv_)
        mu  = np.exp(self.attack_[ai] + self.defense_[hi])
        return float(lam), float(mu)

    def scoreline_matrix(self, home_team, away_team):
        """(MAX_GOALS+1) × (MAX_GOALS+1) probability matrix. m[i,j] = P(home=i, away=j)."""
        lam, mu = self._lambda_mu(home_team, away_team)
        g       = np.arange(MAX_GOALS + 1)
        m       = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))

        # Clip each tau to a small positive floor so a boundary rho never
        # produces a negative cell (which would corrupt the whole matrix).
        m[0, 0] *= max(1e-6, 1 - lam * mu * self.rho_)
        m[1, 0] *= max(1e-6, 1 + mu  * self.rho_)
        m[0, 1] *= max(1e-6, 1 + lam * self.rho_)
        m[1, 1] *= max(1e-6, 1 - self.rho_)

        m /= m.sum()   # renormalise — corrections shift the sum slightly
        return m

    def predict(self, home_team, away_team):
        if home_team not in self._idx or away_team not in self._idx:
            return None

        m   = self.scoreline_matrix(home_team, away_team)
        lam, mu = self._lambda_mu(home_team, away_team)

        g           = np.arange(MAX_GOALS + 1)
        total_goals = g[:, None] + g[None, :]

        return {
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
    most recent 12-month window AND require date > today.
    """
    today        = pd.Timestamp("today").normalize()
    max_date     = matches["date"].dropna().max()
    season_start = max_date - pd.DateOffset(months=12)
    recent       = matches[(matches["date"] >= season_start) & (matches["date"] > today)]

    no_result = (
        recent["home_goals"].isna()
        | recent["away_goals"].isna()
        | (recent["home_goals"] % 1 != 0)
        | (recent["away_goals"] % 1 != 0)
    )
    return recent[no_result][["date", "home_team", "away_team"]].drop_duplicates()


# ── Per-league runner ─────────────────────────────────────────────────────────

def run_league(cleaned_path: str, output_path: str, league_name: str):
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
    print(f"   {len(fixtures)} upcoming fixtures")

    if fixtures.empty:
        print(f"   No upcoming fixtures to predict")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"   Removed stale predictions file")
        return

    rows, skipped = [], 0
    for _, row in fixtures.iterrows():
        pred = model.predict(row["home_team"], row["away_team"])
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
    ("Brazil_Serie_B.csv",        "predictions_brazil_serie_b.csv",    "Brazil Serie B"),
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