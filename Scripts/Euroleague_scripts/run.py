"""
Fit the ratings model and write predictions for upcoming fixtures.

Usage:
  python run.py                 # both competitions
  python run.py --comp E        # Euroleague only
  python run.py --ratings       # also print the current team ratings table

Writes Datasets/Predictions/predictions_euroleague.csv (and _eurocup.csv).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from EuroleagueData import build_dataset, split, COMPETITIONS
from EuroleagueModel import RatingsModel, TeamResolver

ROOT  = Path(__file__).resolve().parent.parent.parent
PREDS = ROOT / 'Datasets' / 'Predictions'

OUT_COLS = [
    'Date', 'Round', 'Home Team', 'Away Team',
    'Home Win %', 'Away Win %',
    'Pred Home Score', 'Pred Away Score',
    'Pred Margin', 'Pred Total',
    'Pick', 'Confidence', 'Cold Start',
]


def run_competition(comp, show_ratings=False):
    cfg  = COMPETITIONS[comp]
    name = cfg['name']
    print(f"\n-- {name}")

    df = build_dataset(comp, use_api=True)
    done, upcoming = split(df)
    print(f"   {len(done)} finished games, {len(upcoming)} upcoming fixtures")

    if done.empty:
        print("   no history — skipping")
        return None
    if upcoming.empty:
        print("   no upcoming fixtures — nothing to predict")
        return None

    model = RatingsModel().fit(done)
    print(f"   fitted on {len(model.teams_)} teams  |  "
          f"home advantage {model.home_adv_:+.2f} pts  |  residual sd {model.sd_:.1f}")

    # Fixture names come from the API, history from the CSV export; sponsor
    # renames mean they rarely match exactly.
    last_played = pd.concat([
        done[['home', 'date']].rename(columns={'home': 'team'}),
        done[['away', 'date']].rename(columns={'away': 'team'}),
    ]).groupby('team')['date'].max().to_dict()
    resolver = TeamResolver(model.teams_, last_played)
    fx = upcoming.copy()
    fx['home_resolved'] = [resolver.resolve(t) or t for t in fx['home']]
    fx['away_resolved'] = [resolver.resolve(t) or t for t in fx['away']]

    unresolved = sorted(
        set(fx.loc[~fx['home_resolved'].isin(model.teams_), 'home'])
        | set(fx.loc[~fx['away_resolved'].isin(model.teams_), 'away'])
    )
    if unresolved:
        print(f"   {len(unresolved)} club(s) with no history — rated as average:")
        for t in unresolved:
            print(f"     - {t}")

    pred = model.predict(fx.rename(columns={'home': '_h', 'away': '_a'})
                           .rename(columns={'home_resolved': 'home',
                                            'away_resolved': 'away'}))
    pred['home'] = fx['home'].values     # display the current club names
    pred['away'] = fx['away'].values

    out = pd.DataFrame({
        'Date':            pd.to_datetime(pred['date']).dt.strftime('%Y-%m-%d %H:%M'),
        'Round':           pred['round'],
        'Home Team':       pred['home'],
        'Away Team':       pred['away'],
        'Home Win %':      (pred['p_home'] * 100).round(1),
        'Away Win %':      (pred['p_away'] * 100).round(1),
        'Pred Home Score': pred['pred_home_score'].round(1),
        'Pred Away Score': pred['pred_away_score'].round(1),
        'Pred Margin':     pred['pred_margin'].round(1),
        'Pred Total':      pred['pred_total'].round(1),
    })
    out['Pick'] = out.apply(
        lambda r: r['Home Team'] if r['Home Win %'] >= 50 else r['Away Team'], axis=1)
    out['Confidence'] = out[['Home Win %', 'Away Win %']].max(axis=1)
    # A cold-started club has no rating, so its prediction is home-advantage only
    out['Cold Start'] = (~pred['known_home'] | ~pred['known_away']).values

    PREDS.mkdir(parents=True, exist_ok=True)
    path = PREDS / f"predictions_{cfg['out']}.csv"
    out[OUT_COLS].sort_values('Date').to_csv(path, index=False)
    print(f"   Saved {len(out)} predictions -> {path.name}")

    if show_ratings:
        print(f"\n   Top 10 by net rating ({name}):")
        t = model.ratings_table().head(10)
        for _, r in t.iterrows():
            print(f"     {r['net_rating']:+6.2f}  {r['team']}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comp', choices=list(COMPETITIONS), default=None)
    ap.add_argument('--ratings', action='store_true', help='print team ratings')
    args = ap.parse_args()

    comps = [args.comp] if args.comp else list(COMPETITIONS)
    print("=" * 62)
    print("  Euroleague / Eurocup predictions")
    print("=" * 62)
    for c in comps:
        run_competition(c, show_ratings=args.ratings)
    print("\n" + "=" * 62)


if __name__ == '__main__':
    main()
