"""
Walk-forward validation for the ratings model.

Every prediction is made from a model fitted only on games that finished before
that game's date, so nothing leaks backwards.  The model is refitted in blocks
rather than per game — refitting per game changes the ratings by a rounding
error and multiplies runtime by the block size.

Scored against three baselines that the model has to beat to be worth running:
  home      — always pick the home side (a high bar: home wins ~62%)
  mean      — predict the training mean for margin and total
  elo       — standard Elo with a home bonus, walked forward the same way

Usage:
  python backtest.py                 # Euroleague, default params
  python backtest.py --comp U        # Eurocup
  python backtest.py --grid          # search decay / alpha
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from EuroleagueData import build_dataset, split, COMPETITIONS
from EuroleagueModel import RatingsModel

BLOCK = 20   # refit every N games


def _logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def elo_baseline(games, k=20, hca=60):
    """Walk-forward Elo over the same games, for comparison."""
    R, out = {}, []
    for _, g in games.iterrows():
        ra, rb = R.get(g['home'], 1500), R.get(g['away'], 1500)
        p = 1 / (1 + 10 ** (-((ra + hca) - rb) / 400))
        out.append(p)
        s = 1.0 if g['home_score'] > g['away_score'] else 0.0
        R[g['home']] = ra + k * (s - p)
        R[g['away']] = rb + k * ((1 - s) - (1 - p))
    return np.array(out)


def walk_forward(done, test_mask, decay, alpha, block=BLOCK):
    """Predict every game where test_mask is True, refitting every `block` games."""
    done = done.sort_values('date').reset_index(drop=True)
    idx  = np.where(test_mask.values)[0]

    preds, model = [], None
    for n, i in enumerate(idx):
        if n % block == 0:
            cutoff = done.loc[i, 'date'] - pd.Timedelta(days=1)
            train  = done[done['date'] <= cutoff]
            if len(train) < 200:
                preds.append(None)
                continue
            model = RatingsModel(decay=decay, alpha=alpha).fit(train, as_of=cutoff)
        if model is None:
            preds.append(None)
            continue
        preds.append(model.predict_one(done.loc[i, 'home'], done.loc[i, 'away']))

    keep = [(i, p) for i, p in zip(idx, preds) if p is not None]
    rows = done.loc[[i for i, _ in keep]].reset_index(drop=True)
    return rows.join(pd.DataFrame([p for _, p in keep]))


def score(res, train_mean_margin, train_mean_total, elo_p=None):
    y      = (res['home_score'] > res['away_score']).astype(int).values
    margin = (res['home_score'] - res['away_score']).values
    total  = (res['home_score'] + res['away_score']).values

    out = {
        'n':             len(res),
        'margin_mae':    float(np.abs(res['pred_margin'] - margin).mean()),
        'margin_mae_base': float(np.abs(margin - train_mean_margin).mean()),
        'total_mae':     float(np.abs(res['pred_total'] - total).mean()),
        'total_mae_base': float(np.abs(total - train_mean_total).mean()),
        'logloss':       _logloss(y, res['p_home'].values),
        'logloss_home':  _logloss(y, np.full(len(y), y.mean())),
        'accuracy':      float(((res['p_home'] > 0.5) == (y == 1)).mean()),
        'acc_home_base': float(y.mean()),
    }
    if elo_p is not None:
        out['logloss_elo'] = _logloss(y, elo_p)
    return out


def report(name, m):
    print(f"\n  === {name}  (n={m['n']}) ===")
    print(f"    margin MAE : {m['margin_mae']:6.2f}   baseline {m['margin_mae_base']:6.2f}"
          f"   {'BETTER' if m['margin_mae'] < m['margin_mae_base'] else 'worse'}")
    print(f"    total  MAE : {m['total_mae']:6.2f}   baseline {m['total_mae_base']:6.2f}"
          f"   {'BETTER' if m['total_mae'] < m['total_mae_base'] else 'worse'}")
    line = f"    log loss   : {m['logloss']:6.4f}   always-home {m['logloss_home']:6.4f}"
    if 'logloss_elo' in m:
        line += f"   elo {m['logloss_elo']:6.4f}"
    print(line)
    print(f"    accuracy   : {m['accuracy']:6.3f}   always-home {m['acc_home_base']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comp', default='E', choices=list(COMPETITIONS))
    ap.add_argument('--seasons', nargs='+', default=None,
                    help='season codes to test on (default: last 3)')
    ap.add_argument('--grid', action='store_true', help='search decay / alpha')
    args = ap.parse_args()

    name = COMPETITIONS[args.comp]['name']
    done, _ = split(build_dataset(args.comp, use_api=True))
    done = done.sort_values('date').reset_index(drop=True)

    seasons = args.seasons or sorted(done['season'].unique())[-3:]
    mask    = done['season'].isin(seasons)
    print(f"\n{name}: {len(done)} finished games | testing on {seasons} ({mask.sum()} games)")

    train_hist       = done[~mask]
    mean_margin      = (train_hist['home_score'] - train_hist['away_score']).mean()
    mean_total       = (train_hist['home_score'] + train_hist['away_score']).mean()

    if args.grid:
        print("\n  decay    alpha    marginMAE  totalMAE  logloss")
        best = None
        for decay in (0.0020, 0.0030, 0.0040, 0.0060):
            for alpha in (5.0, 12.0, 25.0, 50.0):
                r = walk_forward(done, mask, decay, alpha, block=40)
                m = score(r, mean_margin, mean_total)
                print(f"  {decay:.4f}  {alpha:5.1f}    {m['margin_mae']:7.2f}  "
                      f"{m['total_mae']:7.2f}  {m['logloss']:.4f}")
                if best is None or m['logloss'] < best[0]:
                    best = (m['logloss'], decay, alpha)
        print(f"\n  best by log loss: decay={best[1]}, alpha={best[2]}  ({best[0]:.4f})")
        return

    from EuroleagueModel import DECAY, ALPHA
    res = walk_forward(done, mask, DECAY, ALPHA)

    # Score Elo over exactly the games the ratings model predicted, so the two
    # log losses are comparable
    full_elo = pd.Series(elo_baseline(done), index=done.index)
    key      = done.reset_index().merge(res[['game_id']], on='game_id')['index'].values
    elo_p    = full_elo.loc[key].values

    report(name, score(res, mean_margin, mean_total, elo_p))


if __name__ == '__main__':
    main()
