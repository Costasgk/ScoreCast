"""
Opponent-adjusted ratings model for Euroleague / Eurocup.

Basketball scores are roughly normal (totals 158 +/- 18), not Poisson, so the
Dixon-Coles machinery used for football does not transfer.  Instead two ridge
regressions over team indicators:

    margin = home_adv + rating[home] - rating[away]
    total  = base     + pace[home]   + pace[away]

Solving for ratings jointly across all games is what makes them
opponent-adjusted — a team that scores 90 a night against weak defences does
not out-rate one scoring 82 against strong ones.  Ridge shrinks thin-sample
teams toward average, which also gives promoted clubs a sane starting point.

Games are weighted by exp(-decay * days_ago) so that current form dominates
without discarding history outright.

Win probability comes from the fitted residual spread:
    P(home win) = Phi(predicted_margin / residual_sd)
Basketball has no draws (5.5% of games go to overtime), so this needs no
draw term.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

DECAY   = 0.0030   # per day; half-life ~231 days. Tuned by backtest.py --grid
ALPHA   = 5.0      # ridge penalty. Tuned by backtest.py --grid
MIN_SD  = 8.0      # floor on residual spread, guards degenerate fits

# Sponsor and filler tokens that churn year to year and carry no identity
_STOP = {
    'BC', 'BK', 'KK', 'FC', 'CB', 'AS', 'SC', 'BASKET', 'BASKETBALL', 'BASKETS',
    'CLUB', 'TEAM', 'COLLEGE', 'SPORTS', 'SK',
}


def normalise(name):
    """Uppercase ASCII with punctuation stripped — the CSVs shout, the API does not."""
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    s = re.sub(r'[^A-Za-z0-9 ]', ' ', s).upper()
    return re.sub(r'\s+', ' ', s).strip()


def _tokens(name):
    return {t for t in normalise(name).split() if t not in _STOP and len(t) > 2}


class TeamResolver:
    """
    Maps a fixture's club name onto the name used in history.

    Clubs rename constantly ('Fenerbahce Beko' -> 'Fenerbahce Tarfin'), so exact
    matching drops real teams.  Token overlap survives sponsor churn; anything
    that still fails is treated as genuinely new and cold-started rather than
    forced onto a wrong match.
    """

    def __init__(self, known_names, last_played=None):
        """
        last_played: optional {team_name: Timestamp of most recent game}.

        Without it, a club that has run under several sponsor names resolves to
        whichever variant scores highest on token overlap — and the *shortest*
        name usually wins, because it shares fewer non-matching tokens.
        'Fenerbahce Tarfin Istanbul' scores 0.67 against the long-dormant
        'Fenerbahce Istanbul' but only 0.50 against the current 'Fenerbahce Beko
        Istanbul', so the club silently inherits a decayed, near-zero rating.
        Ranking candidates by recency first fixes that.
        """
        self.canon = {}
        self.tokens = {}
        self.last = {}
        for n in known_names:
            key = normalise(n)
            self.canon[key] = n
            self.tokens[key] = _tokens(n)
            self.last[key] = (last_played or {}).get(n)

    def resolve(self, name, threshold=0.5):
        key = normalise(name)
        if key in self.canon:
            return self.canon[key]

        want = _tokens(name)
        if not want:
            return None

        # Containment, not Jaccard: sponsor words pad one side only, so
        # 'Armani Olimpia Milan' vs 'EA7 Emporio Armani Milan' scores 0.40 on
        # Jaccard (below threshold, club dropped) but 0.67 on containment.
        # Requiring two shared tokens keeps that from matching unrelated clubs
        # that happen to share one city name — 'Roma Basketball' must not
        # collapse into 'Maxima Roma'.
        cands = []
        for k, toks in self.tokens.items():
            if not toks:
                continue
            shared = want & toks
            if len(shared) < 2:
                continue
            score = len(shared) / min(len(want), len(toks))
            if score >= threshold:
                cands.append((k, score, self.last.get(k)))
        if not cands:
            return None

        # Most recently active first, then strongest token overlap
        floor = pd.Timestamp.min
        cands.sort(key=lambda c: ((c[2] or floor), c[1]), reverse=True)
        return self.canon[cands[0][0]]


class RatingsModel:

    def __init__(self, decay=DECAY, alpha=ALPHA):
        self.decay   = decay
        self.alpha   = alpha
        self.teams_  = None
        self.rating_ = None    # net rating, points vs average opponent
        self.pace_   = None    # contribution to combined total
        self.home_adv_ = 0.0
        self.total_base_ = 0.0
        self.sd_ = None

    def _design(self, games, symmetric):
        """+1/-1 per team for margin; +1/+1 for total."""
        idx = {t: i for i, t in enumerate(self.teams_)}
        X = np.zeros((len(games), len(self.teams_)))
        for r, (h, a) in enumerate(zip(games['home'], games['away'])):
            X[r, idx[h]] = 1.0
            X[r, idx[a]] = 1.0 if symmetric else -1.0
        return X

    def fit(self, games, as_of=None):
        g = games.dropna(subset=['home_score', 'away_score']).copy()
        g['date'] = pd.to_datetime(g['date'])
        as_of = pd.Timestamp(as_of) if as_of is not None else g['date'].max()
        g = g[g['date'] <= as_of]

        self.teams_ = sorted(set(g['home']) | set(g['away']))
        w = np.exp(-self.decay * (as_of - g['date']).dt.days.clip(lower=0).values)

        margin = (g['home_score'] - g['away_score']).values
        total  = (g['home_score'] + g['away_score']).values

        Xm = self._design(g, symmetric=False)
        mm = Ridge(alpha=self.alpha, fit_intercept=True).fit(Xm, margin, sample_weight=w)
        self.rating_   = mm.coef_
        self.home_adv_ = float(mm.intercept_)

        Xt = self._design(g, symmetric=True)
        tm = Ridge(alpha=self.alpha, fit_intercept=True).fit(Xt, total, sample_weight=w)
        self.pace_       = tm.coef_
        self.total_base_ = float(tm.intercept_)

        resid = margin - mm.predict(Xm)
        # Weighted spread: recent games describe the current error scale
        self.sd_ = max(MIN_SD, float(np.sqrt(np.average(resid ** 2, weights=w))))

        self._idx = {t: i for i, t in enumerate(self.teams_)}
        self._mm, self._tm = mm, tm
        return self

    def _get(self, vec, team):
        """0.0 for an unknown club — ridge centres ratings, so that is 'average'."""
        i = self._idx.get(team)
        return 0.0 if i is None else float(vec[i])

    def predict_one(self, home, away):
        margin = self.home_adv_ + self._get(self.rating_, home) - self._get(self.rating_, away)
        total  = self.total_base_ + self._get(self.pace_, home) + self._get(self.pace_, away)
        p_home = float(norm.cdf(margin / self.sd_))
        return {
            'pred_margin': margin,
            'pred_total':  total,
            'p_home':      p_home,
            'p_away':      1.0 - p_home,
            'pred_home_score': (total + margin) / 2.0,
            'pred_away_score': (total - margin) / 2.0,
            'known_home': home in self._idx,
            'known_away': away in self._idx,
        }

    def predict(self, fixtures):
        rows = [self.predict_one(h, a) for h, a in zip(fixtures['home'], fixtures['away'])]
        out = fixtures.reset_index(drop=True).join(pd.DataFrame(rows))
        return out

    def ratings_table(self):
        return (pd.DataFrame({'team': self.teams_,
                              'net_rating': self.rating_,
                              'pace': self.pace_})
                .sort_values('net_rating', ascending=False)
                .reset_index(drop=True))
