"""
Fit the calibration temperature for the Dixon-Coles model.

Backtesting showed the raw model is overconfident at the extremes: it said 85%
where the observed frequency was 69%, and 6% where it was 13%. The probabilities
are spread too far toward the ends of the range.

A single temperature fixes that shape. Applied to the scoreline grid as
m ** (1/T) renormalised, T > 1 pulls everything toward uniform. One parameter,
so there is very little to overfit, and it is fitted on matches that are then
excluded from the evaluation.

The split is by date, not at random: calibrating on later matches and scoring on
earlier ones would leak the future into the fit and flatter the result.

Usage:
  python calibrate.py                # fit on all six divisions, report, save
  python calibrate.py --no-save      # report only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))

import ScorelineModel as SM
from ScorelineModel import DixonColesModel, temper
from benchmark import DIVS, load, REFIT_EVERY

OUT_PATH   = Path(__file__).resolve().parent.parent / "Datasets" / "Models" / "calibration.json"
TRAIN_FRAC = 0.65     # earliest share of matches used to fit T


def outcome_index(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def marginals(grid):
    """(home, draw, away) from a scoreline grid."""
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def collect(div, seasons, min_train=300):
    """Walk-forward over one division, returning raw grids and what happened."""
    df = load(div, seasons)
    if df is None or df.empty:
        return [], [], []

    grids, ys, dates = [], [], []
    model = None
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
        if g['home_team'] not in model._idx or g['away_team'] not in model._idx:
            continue
        grid = model.scoreline_matrix(g['home_team'], g['away_team'])
        if not np.isfinite(grid).all():
            continue
        grids.append(grid)
        ys.append(outcome_index(g['home_goals'], g['away_goals']))
        dates.append(g['date'])
    return grids, ys, dates


def _logloss_at(t, grids, Y):
    P = np.array([marginals(temper(g, t)) for g in grids])
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.log(P[np.arange(len(Y)), Y]).mean())


def _scores_at(t, grids, Y):
    P = np.array([marginals(temper(g, t)) for g in grids])
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(Y)
    oh = np.zeros_like(P); oh[np.arange(n), Y] = 1
    return {'logloss': float(-np.log(P[np.arange(n), Y]).mean()),
            'brier':   float(((P - oh) ** 2).sum(axis=1).mean() / 3),
            'accuracy': float((P.argmax(axis=1) == Y).mean())}


def reliability(t, grids, Y, bins=10):
    P = np.array([marginals(temper(g, t)) for g in grids])
    P = P / P.sum(axis=1, keepdims=True)
    n = len(Y)
    oh = np.zeros_like(P); oh[np.arange(n), Y] = 1
    p, a = P.ravel(), oh.ravel()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() < 15:
            continue
        rows.append((int(lo * 100), int(hi * 100),
                     p[m].mean() * 100, a[m].mean() * 100, int(m.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--div', nargs='+', default=list(DIVS))
    ap.add_argument('--seasons', nargs='+', default=['2324', '2425', '2526'])
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    # Fit with calibration off, or we would be tuning on top of itself
    SM.TEMPERATURE = 1.0

    print("=" * 66)
    print("  Fitting calibration temperature")
    print(f"  divisions: {', '.join(args.div)}   seasons: {', '.join(args.seasons)}")
    print("=" * 66)

    grids, ys, dates = [], [], []
    for d in args.div:
        g, y, dt = collect(d, args.seasons)
        print(f"  {DIVS.get(d, d):26s} {len(g):5d} matches")
        grids += g; ys += y; dates += dt

    if len(grids) < 500:
        raise SystemExit("not enough matches to calibrate")

    order = np.argsort(pd.to_datetime(dates).values)
    grids = [grids[i] for i in order]
    Y     = np.array(ys)[order]

    cut = int(len(grids) * TRAIN_FRAC)
    gtr, ytr = grids[:cut], Y[:cut]
    gte, yte = grids[cut:], Y[cut:]
    print(f"\n  fit on earliest {len(gtr)} matches, score on later {len(gte)}")

    res = minimize_scalar(lambda lt: _logloss_at(float(np.exp(lt)), gtr, ytr),
                          bounds=(np.log(0.5), np.log(4.0)), method='bounded')
    T = float(np.exp(res.x))
    print(f"  fitted temperature: {T:.4f}   ({'flattens' if T > 1 else 'sharpens'})")

    before = _scores_at(1.0, gte, yte)
    after  = _scores_at(T,   gte, yte)

    print(f"\n  held-out {len(gte)} matches")
    print(f"    {'':10s} {'log loss':>10s} {'Brier':>9s} {'accuracy':>10s}")
    print(f"    {'before':10s} {before['logloss']:10.4f} {before['brier']:9.4f} {before['accuracy']:10.3f}")
    print(f"    {'after':10s} {after['logloss']:10.4f} {after['brier']:9.4f} {after['accuracy']:10.3f}")
    d_ll = after['logloss'] - before['logloss']
    print(f"    change: {d_ll:+.4f} log loss  "
          f"({'BETTER' if d_ll < 0 else 'worse — do not ship'})")

    print(f"\n  reliability on held-out matches")
    print(f"    {'band':>9s} {'said':>6s} {'got':>6s}   {'said':>6s} {'got':>6s}      n")
    print(f"    {'':9s} {'(before)':>13s}   {'(after)':>13s}")
    rb = {(a, b): (p, o, n) for a, b, p, o, n in reliability(1.0, gte, yte)}
    ra = {(a, b): (p, o, n) for a, b, p, o, n in reliability(T,   gte, yte)}
    for k in sorted(set(rb) | set(ra)):
        pb, ob, nb = rb.get(k, (float('nan'),) * 3)
        pa, oa, na = ra.get(k, (float('nan'),) * 3)
        print(f"    {k[0]:3d}-{k[1]:3d}% {pb:6.0f} {ob:6.0f}   {pa:6.0f} {oa:6.0f} {na if na == na else nb:6.0f}")

    if not args.no_save:
        if d_ll >= 0:
            print("\n  calibration did not improve held-out log loss — not saving")
            return
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            'temperature': round(T, 5),
            'fitted': datetime.now().isoformat(timespec='seconds'),
            'divisions': args.div,
            'seasons': args.seasons,
            'n_fit': len(gtr),
            'n_holdout': len(gte),
            'holdout_logloss_before': round(before['logloss'], 5),
            'holdout_logloss_after':  round(after['logloss'], 5),
            'holdout_brier_before':   round(before['brier'], 5),
            'holdout_brier_after':    round(after['brier'], 5),
        }, indent=1))
        print(f"\n  saved -> {OUT_PATH}")
    print()


if __name__ == '__main__':
    main()
