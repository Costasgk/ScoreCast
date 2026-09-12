"""
Data layer for the Euroleague / Eurocup models.

Two sources, because neither is sufficient alone:
  - The CSVs in 'Euroloeague Dataset/' hold 19 seasons of finished games but
    stop at the end of the season they were exported in.
  - api-live.euroleague.net serves the current season's results and the full
    upcoming fixture list, with no auth and no rate limiting worth worrying
    about.

Competitions share a schema and near-identical scoring (Euroleague totals
average 158.3, Eurocup 158.8), so one code path serves both.  Their rating
scales are NOT pooled: across the last three seasons only three clubs appear
in both, so there are no cross-competition games to calibrate one ladder
against the other.
"""

from pathlib import Path

import pandas as pd
import requests

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'Euroloeague Dataset'
API      = "https://api-live.euroleague.net/v2/competitions"
HEADERS  = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/153.0.0.0 Safari/537.36"}

COMPETITIONS = {
    'E': {'name': 'Euroleague', 'csv': 'euroleague', 'out': 'euroleague'},
    'U': {'name': 'Eurocup',    'csv': 'eurocup',    'out': 'eurocup'},
}

# One row per game, home-team perspective. team_a is the home side — verified
# from the history: team_a wins 62.7% and averages +3.81 points.
COLUMNS = ['game_id', 'date', 'season', 'round', 'phase',
           'home', 'away', 'home_score', 'away_score', 'played']


def load_history(comp='E'):
    """Finished games from the exported CSVs."""
    cfg = COMPETITIONS[comp]
    h = pd.read_csv(DATA_DIR / f"{cfg['csv']}_header.csv", low_memory=False)

    df = pd.DataFrame({
        'game_id':    h['game_id'].astype(str),
        'date':       pd.to_datetime(h['date'], errors='coerce'),
        'season':     h['season_code'],
        'round':      h['round'],
        'phase':      h['phase'],
        'home':       h['team_a'].astype(str).str.strip(),
        'away':       h['team_b'].astype(str).str.strip(),
        'home_score': pd.to_numeric(h['score_a'], errors='coerce'),
        'away_score': pd.to_numeric(h['score_b'], errors='coerce'),
    })
    df['played'] = True
    df = df.dropna(subset=['date', 'home_score', 'away_score'])
    # A handful of rows carry 0-0 placeholders in some exports
    df = df[(df['home_score'] > 0) | (df['away_score'] > 0)]
    return df[COLUMNS].sort_values('date').reset_index(drop=True)


def fetch_season(comp='E', season=None, timeout=30):
    """
    Every game the API knows about for one season, played or not.

    Returns the same columns as load_history, so the two stack directly.
    """
    cfg  = COMPETITIONS[comp]
    code = season or f"{comp}2026"
    url  = f"{API}/{comp}/seasons/{code}/games"

    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    games = r.json().get('data', [])

    rows = []
    for g in games:
        local, road = g.get('local') or {}, g.get('road') or {}
        hc, ac = local.get('club') or {}, road.get('club') or {}
        played = bool(g.get('played'))
        rows.append({
            'game_id':    str(g.get('gameCode') or g.get('identifier')),
            'date':       pd.to_datetime(g.get('localDate') or g.get('date'), errors='coerce'),
            'season':     code,
            'round':      g.get('round'),
            'phase':      (g.get('phaseType') or {}).get('alias'),
            'home':       (hc.get('name') or '').strip(),
            'away':       (ac.get('name') or '').strip(),
            # Unplayed games report 0-0; keep them as NaN so nothing trains on them
            'home_score': local.get('score') if played else None,
            'away_score': road.get('score')  if played else None,
            'played':     played,
        })

    df = pd.DataFrame(rows)
    df = df[df['home'].ne('') & df['away'].ne('')]
    return df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)


def build_dataset(comp='E', season=None, use_api=True):
    """
    Full history + the live season, de-duplicated.

    Club names drift between the CSV export and the API ("Fenerbahce Beko
    Istanbul" vs "Fenerbahce Tarfin Istanbul" — sponsor changes), so the two
    are concatenated rather than joined on name, and duplicates are dropped on
    (date, home, away).
    """
    frames = [load_history(comp)]
    if use_api:
        try:
            frames.append(fetch_season(comp, season))
        except Exception as e:
            print(f"  [!] API fetch failed ({e}) — using CSV history only")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['date', 'home', 'away'], keep='last')
    return df.sort_values('date').reset_index(drop=True)


def split(df):
    """(finished games for training, upcoming fixtures to predict)."""
    done     = df[df['played'] & df['home_score'].notna() & df['away_score'].notna()]
    upcoming = df[~df['played']]
    return done.reset_index(drop=True), upcoming.reset_index(drop=True)


if __name__ == '__main__':
    for c in COMPETITIONS:
        d = build_dataset(c)
        done, up = split(d)
        name = COMPETITIONS[c]['name']
        print(f"{name:11s} {len(done):5d} finished  {len(up):4d} upcoming  "
              f"({done['date'].min().date()} -> {done['date'].max().date()})")
