"""
Fit the Dixon-Coles time-decay rate against held-out log loss.

DECAY_RATE = 0.0065 was chosen by reasoning about half-lives ("about 107 days")
and has never been scored. The equivalent parameter in the Euroleague model came
out at 0.0030 once it was actually fitted, well away from the value that seemed
sensible, so the football one is worth checking.

Each candidate is scored walk-forward: the model is refitted as the season
progresses and only ever sees matches that had already finished. Candidates are
ranked on the earlier portion and the winner is confirmed on a later portion
that took no part in choosing it, so a value that merely suits one stretch of
fixtures does not get adopted.

Usage:
  python tune_decay.py                     # default grid, all six divisions
  python tune_decay.py --div E0 I1
  python tune_decay.py --grid 0.002 0.004 0.006
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from ScorelineModel import DECAY_RATE, DixonColesModel
from benchmark import DIVS, REFIT_EVERY, load

GRID = [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0085, 0.0110]
SELECT_FRAC = 0.6      # earlier share used to rank candidates


def outcome_probs(model, home, away):
    p = model.predict(home, away)
    if p is None:
        return None
    return np.array([p['home_win'], p['draw'], p['away_win']])


def walk(df, decay, min_train=300):
    """(probabilities, outcomes, dates) for one decay value over one division."""
    P, Y, D = [], [], []
    model = None
    for i in range(len(df)):
        if i < min_train:
            continue
        if (i - min_train) % REFIT_EVERY == 0:
            cutoff = df.loc[i, 'date'] - pd.Timedelta(days=1)
            train  = df[df['date'] <= cutoff]
            if len(train) < min_train:
                continue
            model = DixonColesModel(decay_rate=decay).fit(train)
        if model is None:
            continue
        g = df.loc[i]
        p = outcome_probs(model, g['home_team'], g['away_team'])
        if p is None or not np.isfinite(p).all():
            continue
        P.append(p)
        Y.append(0 if g['home_goals'] > g['away_goals']
                 else (1 if g['home_goals'] == g['away_goals'] else 2))
        D.append(g['date'])
    return np.array(P), np.array(Y), np.array(D)


def logloss(P, Y):
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.log(P[np.arange(len(Y)), Y]).mean())


def brier(P, Y):
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    oh = np.zeros_like(P); oh[np.arange(len(Y)), Y] = 1
    return float(((P - oh) ** 2).sum(axis=1).mean() / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--div', nargs='+', default=list(DIVS))
    ap.add_argument('--seasons', nargs='+', default=['2324', '2425', '2526'])
    ap.add_argument('--grid', nargs='+', type=float, default=GRID)
    args = ap.parse_args()

    print("=" * 68)
    print(f"  Tuning DECAY_RATE   (current value {DECAY_RATE})")
    print(f"  divisions: {', '.join(args.div)}   seasons: {', '.join(args.seasons)}")
    print("=" * 68)

    frames = {}
    for d in args.div:
        df = load(d, args.seasons)
        if df is not None and len(df) > 400:
            frames[d] = df
            print(f"  {DIVS.get(d, d):26s} {len(df):5d} matches")
    if not frames:
        raise SystemExit("no data")

    print(f"\n  {'decay':>8s} {'half-life':>10s} {'select LL':>11s} {'holdout LL':>12s} {'holdout Brier':>14s}")
    results = []
    for decay in args.grid:
        sel_P, sel_Y, hold_P, hold_Y = [], [], [], []
        for d, df in frames.items():
            P, Y, D = walk(df, decay)
            if len(Y) < 50:
                continue
            cut = int(len(Y) * SELECT_FRAC)
            sel_P.append(P[:cut]);  sel_Y.append(Y[:cut])
            hold_P.append(P[cut:]); hold_Y.append(Y[cut:])
        sP, sY = np.vstack(sel_P), np.concatenate(sel_Y)
        hP, hY = np.vstack(hold_P), np.concatenate(hold_Y)
        r = {'decay': decay,
             'select': logloss(sP, sY),
             'holdout': logloss(hP, hY),
             'holdout_brier': brier(hP, hY)}
        results.append(r)
        half = np.log(2) / decay
        mark = '  <- current' if abs(decay - DECAY_RATE) < 1e-9 else ''
        print(f"  {decay:8.4f} {half:9.0f}d {r['select']:11.4f} {r['holdout']:12.4f} "
              f"{r['holdout_brier']:14.4f}{mark}")

    best = min(results, key=lambda r: r['select'])

    # Only compare against the live value if it was actually scored here. A grid
    # that excludes it would otherwise silently compare against whichever
    # candidate happens to sit nearest, and report a far smaller gain than the
    # real one.
    current = next((r for r in results
                    if abs(r['decay'] - DECAY_RATE) < 1e-9), None)
    if current is None:
        print(f"\n  note: current value {DECAY_RATE} is not in this grid, so no")
        print(f"        before/after comparison is possible from this run alone.")
        print(f"  best on selection portion : {best['decay']:.4f}")
        print(f"  its held-out log loss     : {best['holdout']:.4f}")
        return

    print(f"\n  best on selection portion : {best['decay']:.4f}")
    print(f"  its held-out log loss     : {best['holdout']:.4f}")
    print(f"  current value held-out    : {current['holdout']:.4f}")
    delta = best['holdout'] - current['holdout']
    print(f"  change if adopted         : {delta:+.4f} "
          f"({'BETTER' if delta < 0 else 'no improvement'})")

    if delta < -0.0005:
        print(f"\n  -> worth changing DECAY_RATE to {best['decay']:.4f} "
              f"in ScorelineModel.py")
    else:
        print(f"\n  -> keep {DECAY_RATE}; the fitted value is not meaningfully better")
    print()


if __name__ == '__main__':
    main()
