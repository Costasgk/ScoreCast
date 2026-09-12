"""
Benchmark ScoreCast against the betting market.

Accuracy is close to useless for judging a 1X2 forecaster: home teams win about
46% of the time, so "always pick home" scores respectably while saying nothing.
What matters is whether the probabilities are calibrated and sharp, which is
what log loss and the Brier score measure.

The reference is the closing price. Bookmakers' closing odds are the strongest
publicly available forecast of a football match — they absorb team news,
injuries and the weight of money right up to kick-off. A model that gets near
them is genuinely good; one that beats them consistently would be remarkable.

Every prediction here is made walk-forward, fitting only on matches that
finished before the one being predicted, so nothing leaks backwards.

Usage:
  python benchmark.py                     # Premier League, last 2 seasons
  python benchmark.py --div I1 SP1 D1     # other divisions
  python benchmark.py --seasons 2425 2526
"""

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))

from ScorelineModel import DixonColesModel

BASE    = "https://www.football-data.co.uk/mmz4281"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/153.0.0.0 Safari/537.36"}

OUT_PATH  = Path(__file__).resolve().parent.parent / "Datasets" / "Models" / "benchmark.json"
CACHE_DIR = Path(__file__).resolve().parent.parent / "Datasets" / "cache"
CURRENT_TTL_H = 6          # hours before the in-progress season is refetched
SEASONS_KNOWN = set()      # populated by load(); the newest is the live one

DIVS = {'E0': 'England Premier League', 'I1': 'Italy Serie A', 'SP1': 'Spain La Liga',
        'D1': 'Germany Bundesliga', 'F1': 'France Ligue 1', 'G1': 'Greece Super League'}

REFIT_EVERY = 10   # matches; refitting per match moves ratings by a rounding error


def _fetch(div, season, retries=3):
    """
    One season file, cached on disk.

    Completed seasons never change, and a parameter sweep re-reads the same
    files for every candidate value. Re-downloading them turns one sweep into
    hundreds of requests against a free service, which is slow and rude — and
    it is what made the first tuning run die on a dropped connection.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{season}_{div}.csv"
    if path.exists() and path.stat().st_size > 500:
        # Finished seasons are immutable, so their cache never expires. The
        # newest season is still being played — caching it forever would freeze
        # the benchmark on whatever results existed the first time it ran.
        is_current = season == max(SEASONS_KNOWN) if SEASONS_KNOWN else False
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if not is_current or age_h < CURRENT_TTL_H:
            return path.read_text(encoding="latin-1")

    last = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}/{season}/{div}.csv", headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.text) >= 500:
                path.write_text(r.text, encoding="latin-1")
                return r.text
            return None
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
    print(f"  [!] {div} {season}: {last}")
    return None


def load(div, seasons):
    SEASONS_KNOWN.update(seasons)
    frames = []
    for s in seasons:
        text = _fetch(div, s)
        if text is None:
            continue
        d = pd.read_csv(io.StringIO(text), encoding="latin-1", on_bad_lines="skip")
        d.columns = [c.lstrip("﻿").lstrip("ï»¿").strip() for c in d.columns]
        frames.append(d)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in df.columns}

    # Closing odds where published, opening otherwise
    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return df[cols[n.lower()]]
        return None

    out = pd.DataFrame({
        'date':       pd.to_datetime(pick('Date'), dayfirst=True, errors='coerce'),
        'home_team':  pick('HomeTeam'),
        'away_team':  pick('AwayTeam'),
        'home_goals': pd.to_numeric(pick('FTHG'), errors='coerce'),
        'away_goals': pd.to_numeric(pick('FTAG'), errors='coerce'),
        'oh':         pd.to_numeric(pick('B365CH', 'B365H', 'AvgCH', 'AvgH'), errors='coerce'),
        'od':         pd.to_numeric(pick('B365CD', 'B365D', 'AvgCD', 'AvgD'), errors='coerce'),
        'oa':         pd.to_numeric(pick('B365CA', 'B365A', 'AvgCA', 'AvgA'), errors='coerce'),
    })
    out = out.dropna(subset=['date', 'home_goals', 'away_goals', 'home_team', 'away_team'])
    return out.sort_values('date').reset_index(drop=True)


def devig(oh, od, oa):
    """
    Closing odds -> probabilities.

    Raw 1/odds sums above 1: the excess is the bookmaker's margin. Dividing
    through by the sum removes it proportionally, which is the standard
    treatment and good enough at the margins these markets run.
    """
    raw = np.array([1 / oh, 1 / od, 1 / oa])
    return raw / raw.sum()


def outcome_probs(model, home, away):
    """(home, draw, away) from the scoreline grid."""
    p = model.predict(home, away)
    if p is None:
        return None
    return np.array([p['home_win'], p['draw'], p['away_win']])


def _scores(P, Y):
    """P: (n,3) forecasts. Y: (n,) index of the outcome that happened."""
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(Y)
    onehot = np.zeros_like(P)
    onehot[np.arange(n), Y] = 1
    return {
        'n':        n,
        'logloss':  float(-np.log(P[np.arange(n), Y]).mean()),
        # Averaged over the three outcomes, matching how 1X2 Brier is usually
        # reported: a uniform 1/3 forecast scores 0.2222.
        'brier':    float(((P - onehot) ** 2).sum(axis=1).mean() / 3),
        'accuracy': float((P.argmax(axis=1) == Y).mean()),
    }


def calibration(P, Y, bins=10):
    """
    Reliability: of the times a forecast said x%, how often did it happen?

    All three outcomes are pooled, so one match contributes three points. A
    perfectly calibrated forecaster sits on the diagonal.
    """
    n = len(Y)
    onehot = np.zeros_like(P)
    onehot[np.arange(n), Y] = 1
    p = P.ravel()
    a = onehot.ravel()

    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() < 15:
            continue
        out.append({'lo': round(lo, 2), 'hi': round(hi, 2),
                    'predicted': round(float(p[m].mean()), 4),
                    'observed':  round(float(a[m].mean()), 4),
                    'n': int(m.sum())})
    return out


def value_return(MP, BP, Y, odds, edge=0.05):
    """
    Flat-stake return from backing only where the model disagrees upward.

    One unit on every outcome the model rates at least `edge` above the market,
    settled at the closing price actually on offer. This is the question that
    matters: is the disagreement information, or noise?
    """
    staked = returned = 0
    hits = 0
    for mp, bp, y, o in zip(MP, BP, Y, odds):
        for k in range(3):
            if mp[k] - bp[k] >= edge:
                staked += 1
                if y == k:
                    returned += o[k]
                    hits += 1
    if not staked:
        return {'bets': 0, 'roi': None, 'hit_rate': None}
    return {'bets': int(staked),
            'roi': round((returned - staked) / staked, 4),
            'hit_rate': round(hits / staked, 4)}


def run(div, seasons, min_train=300):
    name = DIVS.get(div, div)
    df = load(div, seasons)
    if df is None or df.empty:
        print(f"  {name}: no data")
        return None

    rows, model = [], None
    for i in range(len(df)):
        if i < min_train:
            continue
        if (i - min_train) % REFIT_EVERY == 0:
            cutoff = df.loc[i, 'date'] - pd.Timedelta(days=1)
            train  = df[df['date'] <= cutoff]
            if len(train) < min_train:
                continue
            model = DixonColesModel().fit(train)
        if model is None:
            continue

        g = df.loc[i]
        mp = outcome_probs(model, g['home_team'], g['away_team'])
        if mp is None or not np.isfinite(mp).all():
            continue
        if not np.isfinite([g['oh'], g['od'], g['oa']]).all():
            continue

        y = 0 if g['home_goals'] > g['away_goals'] else (1 if g['home_goals'] == g['away_goals'] else 2)
        rows.append((mp, devig(g['oh'], g['od'], g['oa']), y,
                     (g['oh'], g['od'], g['oa'])))

    if not rows:
        print(f"  {name}: nothing to score")
        return None

    MP   = np.array([r[0] for r in rows])
    BP   = np.array([r[1] for r in rows])
    Y    = np.array([r[2] for r in rows])
    ODDS = np.array([r[3] for r in rows])
    U    = np.full_like(MP, 1 / 3)

    return {'league': name, 'div': div,
            'model':   _scores(MP, Y),
            'market':  _scores(BP, Y),
            'uniform': _scores(U, Y),
            'calibration_model':  calibration(MP, Y),
            'calibration_market': calibration(BP, Y),
            'value': value_return(MP, BP, Y, ODDS)}


def report(res):
    m, b, u = res['model'], res['market'], res['uniform']
    gap = m['logloss'] - b['logloss']
    print(f"\n  {res['league']}   n={m['n']}")
    print(f"    {'':10s} {'log loss':>10s} {'Brier':>9s} {'accuracy':>10s}")
    print(f"    {'ScoreCast':10s} {m['logloss']:10.4f} {m['brier']:9.4f} {m['accuracy']:10.3f}")
    print(f"    {'Market':10s} {b['logloss']:10.4f} {b['brier']:9.4f} {b['accuracy']:10.3f}")
    print(f"    {'Uniform':10s} {u['logloss']:10.4f} {u['brier']:9.4f} {u['accuracy']:10.3f}")
    print(f"    gap to market: {gap:+.4f} log loss "
          f"({'behind' if gap > 0 else 'AHEAD OF'} the closing price)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--div', nargs='+', default=['E0'], help=f"any of {list(DIVS)}")
    ap.add_argument('--seasons', nargs='+', default=['2324', '2425', '2526'])
    ap.add_argument('--save', action='store_true',
                    help='cache results to Datasets/Models/benchmark.json for the web app')
    ap.add_argument('--all', action='store_true', help='every division with published odds')
    args = ap.parse_args()
    if args.all:
        args.div = list(DIVS)

    print("=" * 64)
    print("  ScoreCast vs the closing price")
    print(f"  divisions: {', '.join(args.div)}   seasons: {', '.join(args.seasons)}")
    print("=" * 64)

    all_res = [r for r in (run(d, args.seasons) for d in args.div) if r]
    for r in all_res:
        report(r)

    if len(all_res) > 1:
        gaps = [r['model']['logloss'] - r['market']['logloss'] for r in all_res]
        print(f"\n  mean gap across {len(all_res)} leagues: {np.mean(gaps):+.4f} log loss")

    for r in all_res:
        v = r['value']
        if v['bets']:
            print(f"  {r['league']:26s} value bets {v['bets']:4d}  "
                  f"ROI {v['roi']:+.1%}  hit {v['hit_rate']:.1%}")

    if args.save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            'generated': datetime.now().isoformat(timespec='seconds'),
            'seasons': args.seasons,
            'leagues': all_res,
        }, indent=1))
        print(f"\n  saved -> {OUT_PATH}")
    print()


if __name__ == '__main__':
    main()
