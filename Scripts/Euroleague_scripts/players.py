"""
Player statistics from the Euroleague and Eurocup box scores.

The box scores are the most complete data in this project — 129,168 player-games
for the Euroleague alone, nineteen seasons deep, with no missing values in any
of the columns that matter. They have never been used.

This produces per-game and per-40-minute rates for the current season, plus
shooting splits and a recent-form window. Rates matter more than totals: a
player averaging 14 points in 32 minutes and one averaging 14 in 19 are not the
same player, and only the second has room to grow.

These are descriptive statistics, not projections. Predicting a player's next
line needs expected minutes, and nothing in this data says who is injured or who
has fallen out of the rotation — so this measures what happened rather than
guessing what will.

Usage:
  python players.py                  # both competitions, current season
  python players.py --comp U
  python players.py --season E2024
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from EuroleagueData import COMPETITIONS, DATA_DIR

OUT_PATH = Path(__file__).resolve().parent.parent.parent / 'Datasets' / 'Models' / 'players.json'

MIN_GAMES = 5      # below this the rates are noise
FORM_GAMES = 5     # recent window


def _minutes(series):
    """'17:53' -> 17.88. Blank and '0' both mean did not play."""
    s = series.astype(str).str.strip()
    ok = s.str.match(r'^\d+:\d{2}$')
    mm = pd.to_numeric(s.where(ok).str.split(':').str[0], errors='coerce')
    ss = pd.to_numeric(s.where(ok).str.split(':').str[1], errors='coerce')
    return (mm + ss / 60).fillna(pd.to_numeric(s, errors='coerce')).fillna(0.0)


def load_box(comp='E'):
    cfg = COMPETITIONS[comp]
    df = pd.read_csv(DATA_DIR / f"{cfg['csv']}_box_score.csv", low_memory=False)
    hdr = pd.read_csv(DATA_DIR / f"{cfg['csv']}_header.csv", low_memory=False,
                      usecols=['game_id', 'date', 'season_code'])
    hdr['date'] = pd.to_datetime(hdr['date'], errors='coerce')
    df = df.merge(hdr[['game_id', 'date']], on='game_id', how='left')
    # The box score carries one row per team per game alongside the players,
    # marked with dorsal 'TOTAL' and 200:00 minutes (five men times forty). Left
    # in, every team outranks every player.
    df = df[df['dorsal'].astype(str).str.strip().str.upper() != 'TOTAL']

    df['min'] = _minutes(df['minutes'])
    # A row with no minutes is a squad listing, not an appearance
    return df[df['min'] > 0].copy()


def season_table(df, season):
    d = df[df['season_code'].astype(str) == str(season)]
    if d.empty:
        return []

    g = d.groupby(['player_id', 'player'], dropna=True)
    agg = g.agg(
        team=('team_id', lambda s: s.mode().iat[0] if len(s.mode()) else ''),
        games=('game_id', 'nunique'),
        starts=('is_starter', 'sum'),
        mins=('min', 'sum'),
        pts=('points', 'sum'),
        reb=('total_rebounds', 'sum'),
        oreb=('offensive_rebounds', 'sum'),
        ast=('assists', 'sum'),
        stl=('steals', 'sum'),
        blk=('blocks_favour', 'sum'),
        tov=('turnovers', 'sum'),
        pf=('fouls_committed', 'sum'),
        val=('valuation', 'sum'),
        pm=('plus_minus', 'sum'),
        fg2m=('two_points_made', 'sum'), fg2a=('two_points_attempted', 'sum'),
        fg3m=('three_points_made', 'sum'), fg3a=('three_points_attempted', 'sum'),
        ftm=('free_throws_made', 'sum'), fta=('free_throws_attempted', 'sum'),
    ).reset_index()

    agg = agg[agg['games'] >= MIN_GAMES]
    if agg.empty:
        return []

    # Recent form: the player's last N games by date
    d_sorted = d.sort_values('date')
    form = (d_sorted.groupby('player_id')
            .tail(FORM_GAMES)
            .groupby('player_id')
            .agg(form_pts=('points', 'mean'), form_min=('min', 'mean'),
                 form_val=('valuation', 'mean'))
            .reset_index())
    agg = agg.merge(form, on='player_id', how='left')

    def per(col, by):
        return (agg[col] / agg[by]).replace([np.inf, -np.inf], 0).fillna(0)

    rows = []
    for _, r in agg.iterrows():
        fga = r['fg2a'] + r['fg3a']
        fgm = r['fg2m'] + r['fg3m']
        # True shooting charges the possession a trip to the line costs, which
        # is the only fair way to compare a slasher with a shooter.
        tsa = fga + 0.44 * r['fta']
        rows.append({
            'player': r['player'], 'team': r['team'],
            'games': int(r['games']), 'starts': int(r['starts']),
            'mpg':  round(r['mins'] / r['games'], 1),
            'ppg':  round(r['pts'] / r['games'], 1),
            'rpg':  round(r['reb'] / r['games'], 1),
            'apg':  round(r['ast'] / r['games'], 1),
            'spg':  round(r['stl'] / r['games'], 1),
            'bpg':  round(r['blk'] / r['games'], 1),
            'tpg':  round(r['tov'] / r['games'], 1),
            'vpg':  round(r['val'] / r['games'], 1),
            'pm':   round(r['pm'] / r['games'], 1),
            'pts40': round(r['pts'] / r['mins'] * 40, 1) if r['mins'] else 0,
            'reb40': round(r['reb'] / r['mins'] * 40, 1) if r['mins'] else 0,
            'ast40': round(r['ast'] / r['mins'] * 40, 1) if r['mins'] else 0,
            'fg2':  round(r['fg2m'] / r['fg2a'] * 100, 1) if r['fg2a'] else 0,
            'fg3':  round(r['fg3m'] / r['fg3a'] * 100, 1) if r['fg3a'] else 0,
            'ft':   round(r['ftm'] / r['fta'] * 100, 1) if r['fta'] else 0,
            # Attempts travel with the percentages. A 100% three-point shooter
            # who took two shots tops any unfiltered leaderboard, so whoever
            # displays these needs to know the sample behind them.
            'fg3a': int(r['fg3a']), 'fg2a': int(r['fg2a']), 'fta': int(r['fta']),
            'fg3a_pg': round(r['fg3a'] / r['games'], 1),
            'ts':   round(r['pts'] / (2 * tsa) * 100, 1) if tsa else 0,
            'form_pts': round(r.get('form_pts', 0) or 0, 1),
            'form_min': round(r.get('form_min', 0) or 0, 1),
            'form_val': round(r.get('form_val', 0) or 0, 1),
        })
    rows.sort(key=lambda x: -x['vpg'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comp', choices=list(COMPETITIONS), default=None)
    ap.add_argument('--season', default=None)
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    comps = [args.comp] if args.comp else list(COMPETITIONS)
    out = []

    print("=" * 60)
    print("  Player statistics")
    print("=" * 60)

    for c in comps:
        df = load_box(c)
        season = args.season or sorted(df['season_code'].astype(str).unique())[-1]
        rows = season_table(df, season)
        if not rows:
            print(f"  {COMPETITIONS[c]['name']}: no data for {season}")
            continue
        print(f"  {COMPETITIONS[c]['name']:11s} {season}  {len(rows):3d} players "
              f"(min {MIN_GAMES} games)  -> top: {rows[0]['player']} "
              f"{rows[0]['vpg']} val/g")
        out.append({'comp': c, 'name': COMPETITIONS[c]['name'],
                    'season': season, 'min_games': MIN_GAMES,
                    'players': rows})

    if out and not args.no_save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            'generated': datetime.now().isoformat(timespec='seconds'),
            'competitions': out,
        }, indent=1))
        print(f"\n  saved -> {OUT_PATH}")
    print()


if __name__ == '__main__':
    main()
