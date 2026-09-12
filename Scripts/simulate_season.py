"""
Monte-Carlo the rest of the season from the Dixon-Coles model.

A fixture probability is consulted once. A title race is followed for months,
and it is the question people actually have: who wins this, who goes down.
Both come from the same model already fitted for the fixture pages — every
remaining match is played ten thousand times and the tables are counted.

The remaining schedule comes from ScorelineModel.remaining_pairings, the same
function the fixture pages use, so a projection covers exactly the matches the
league pages list. It prefers the published ESPN/FBref calendar and, where the
format is measurably a double round-robin, tops that up with the meetings that
must happen but have not been announced yet — a calendar published only as far
as Christmas would otherwise decide the title over a partial season.

Usage:
  python simulate_season.py                    # every league, 10k runs
  python simulate_season.py --runs 50000
  python simulate_season.py --league England_Premier_League.csv
"""

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from ScorelineModel import (DixonColesModel, MAX_GOALS, load_matches,
                            remaining_pairings)

ROOT     = Path(__file__).resolve().parent.parent
CLEANED  = ROOT / 'Datasets' / 'Cleaned Datasets'
OUT_PATH = ROOT / 'Datasets' / 'Models' / 'season_sim.json'

WIN, DRAW = 3, 1

# Places that decide something, per league. Playoffs are not modelled — the
# German and Norwegian relegation play-off places are counted as survival, so
# 'relegated' here means automatic relegation only.
# Places that decide something, per league, from the shared registry. Play-offs
# are not modelled, so 'relegate' means the automatic places only — the registry
# flags which leagues have verified values and which are a sensible default.
from leagues import LEAGUES as _REG

STRUCTURE = {
    l['file']: {'name': f"{l['name']} ({l['region']})", 'key': l['key'],
                'europe': l['europe'], 'relegate': l['relegate'],
                'exact': l['exact'], 'round_robin': l['round_robin']}
    for l in _REG
}



def _published_fixtures(league_key):
    """Remaining fixtures from the ESPN/FBref calendar, as (home, away) pairs."""
    if not league_key:
        return None
    path = ROOT / 'Datasets' / 'Models' / 'fixture_dates.json'
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text()).get('leagues', {}).get(league_key)
    except Exception:
        return None
    if not cache:
        return None

    now = pd.Timestamp.now()
    out = []
    for key, when in cache.items():
        h, _, a = key.partition('|')
        w = pd.to_datetime(when, errors='coerce')
        if h and a and pd.notna(w) and w >= now:
            out.append((h, a))
    return out or None


def current_season(path, league_key=None):
    """(members, played rows, remaining ordered pairs) for the newest season."""
    df = pd.read_csv(path, index_col=0, low_memory=False)
    if 'season' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    latest = sorted(df['season'].dropna().astype(str).unique())[-1]
    cur = df[(df['season'].astype(str) == latest)
             & (df['venue'].astype(str).str.lower().str.strip() == 'home')]
    if cur.empty:
        return None

    members = sorted(set(cur['team'].astype(str)) | set(cur['opponent'].astype(str)))
    played  = cur[cur['gf'].notna() & cur['ga'].notna()]
    done    = {(str(r['team']), str(r['opponent'])) for _, r in played.iterrows()}

    # Ask the model for the remaining schedule rather than working one out here.
    # Two copies of this arithmetic drifted apart: the fixture pages learned to
    # top a short calendar up with the meetings it had not announced yet, while
    # this file kept simulating whatever the calendar happened to contain — so
    # the Premier League title was being decided over 350 games instead of 380,
    # and 2. Bundesliga over 265 of its 306. Percentages from a season missing a
    # tenth of its fixtures are wrong in a way nothing on the page reveals.
    remaining = [(h, a) for h, a in
                 zip(*(remaining_pairings(str(path), {}, league_key)
                       [c] for c in ('home_team', 'away_team')))]
    if remaining:
        members = sorted(set(members) | {t for p in remaining for t in p})
        source  = 'calendar' if _published_fixtures(league_key) else 'derived'
        return {'season': latest, 'members': members, 'played': played,
                'remaining': remaining, 'source': source}

    remaining = [p for p in itertools.permutations(members, 2) if p not in done]
    return {'season': latest, 'members': members, 'played': played,
            'remaining': remaining, 'source': 'derived'}


def standings(members, played):
    """Points, goal difference and goals for, from results so far."""
    pts = {t: 0 for t in members}
    gd  = {t: 0 for t in members}
    gf  = {t: 0 for t in members}
    pl  = {t: 0 for t in members}
    for _, r in played.iterrows():
        h, a = str(r['team']), str(r['opponent'])
        if h not in pts or a not in pts:
            continue
        hg, ag = int(r['gf']), int(r['ga'])
        pl[h] += 1; pl[a] += 1
        gf[h] += hg; gf[a] += ag
        gd[h] += hg - ag; gd[a] += ag - hg
        if hg > ag:   pts[h] += WIN
        elif hg < ag: pts[a] += WIN
        else:         pts[h] += DRAW; pts[a] += DRAW
    return pts, gd, gf, pl


def simulate(model, members, remaining, pts0, gd0, gf0, runs, rng):
    """
    Play every remaining fixture `runs` times.

    Scorelines are drawn from the model's own grid rather than from the 1X2
    summary, so a 4-0 and a 1-0 contribute differently to goal difference — and
    goal difference is what separates teams at the top and bottom.
    """
    idx = {t: i for i, t in enumerate(members)}
    n   = len(members)

    pts = np.tile(np.array([pts0[t] for t in members], dtype=np.int32), (runs, 1))
    gd  = np.tile(np.array([gd0[t] for t in members], dtype=np.int32), (runs, 1))
    gf  = np.tile(np.array([gf0[t] for t in members], dtype=np.int32), (runs, 1))

    side = MAX_GOALS + 1
    for home, away in remaining:
        # Promoted clubs have no rating in this division. Skipping their
        # fixtures would hand them zero points and a guaranteed last place, so
        # they are simulated as average sides and flagged in the output.
        grid = model.scoreline_matrix(home, away, allow_unknown=True).ravel()
        grid = np.clip(grid, 0, None)
        grid = grid / grid.sum()

        draw = rng.choice(grid.size, size=runs, p=grid)
        hg, ag = draw // side, draw % side

        h, a = idx[home], idx[away]
        gf[:, h] += hg;      gf[:, a] += ag
        gd[:, h] += hg - ag; gd[:, a] += ag - hg
        pts[:, h] += np.where(hg > ag, WIN, np.where(hg == ag, DRAW, 0))
        pts[:, a] += np.where(ag > hg, WIN, np.where(hg == ag, DRAW, 0))

    # Rank by points, then goal difference, then goals scored. Lexsort takes the
    # last key as primary, and negation turns ascending into descending.
    order = np.lexsort((-gf, -gd, -pts), axis=1)
    pos = np.empty_like(order)
    rows = np.arange(runs)[:, None]
    pos[rows, order] = np.arange(n)[None, :]
    return pts, pos


def run_league(path, runs, rng):
    fname = path.name
    cfg   = STRUCTURE.get(fname, {'name': fname, 'europe': 4, 'relegate': 3})
    info  = current_season(path, cfg.get('key'))
    if info is None:
        print(f"  skip {cfg['name']} — no season data")
        return None

    if info.get('source') != 'calendar' and not cfg.get('round_robin', True):
        # Without a published calendar, a split season or conference schedule
        # cannot have its remaining fixtures derived — the pairings arithmetic
        # would invent matches that never happen.
        print(f"  skip {cfg['name']} — no calendar and not a round-robin")
        return None

    members, remaining = info['members'], info['remaining']
    if not remaining:
        print(f"  skip {cfg['name']} — season complete ({info['season']})")
        return None

    matches = load_matches(str(path))
    done    = matches[matches['home_goals'].notna() & matches['away_goals'].notna()]
    if len(done) < 100:
        print(f"  skip {cfg['name']} — not enough history")
        return None
    model = DixonColesModel().fit(done)

    unknown = [t for t in members if t not in model._idx]
    pts0, gd0, gf0, pl0 = standings(members, info['played'])
    pts, pos = simulate(model, members, remaining, pts0, gd0, gf0, runs, rng)

    n = len(members)
    rows = []
    for i, t in enumerate(members):
        p = pos[:, i]
        rows.append({
            'team':       t,
            'played':     pl0[t],
            'points_now': pts0[t],
            'exp_points': round(float(pts[:, i].mean()), 1),
            'title':      round(float((p == 0).mean()) * 100, 2),
            'europe':     round(float((p < cfg['europe']).mean()) * 100, 2),
            'relegated':  round(float((p >= n - cfg['relegate']).mean()) * 100, 2),
            'avg_pos':    round(float(p.mean()) + 1, 2),
            'best_pos':   int(p.min()) + 1,
            'worst_pos':  int(p.max()) + 1,
            'unknown':    t in unknown,
        })
    rows.sort(key=lambda r: (-r['exp_points'], r['avg_pos']))

    print(f"  {cfg['name']:26s} {len(members):2d} teams  {len(remaining):3d} to play  "
          f"-> {rows[0]['team']} {rows[0]['title']:.0f}%")
    return {
        'league': cfg['name'], 'file': fname, 'season': info['season'],
        # A league that is not a straight round-robin decides its title in a
        # playoff or championship split the published calendar does not cover.
        # These projections are of the regular season only, and the page has to
        # say so — 72% for Nashville means topping the table, not winning MLS.
        'regular_season_only': not cfg.get('round_robin', True),
        'fixture_source': info.get('source', 'derived'),
        'teams': n, 'played': len(info['played']), 'remaining': len(remaining),
        'europe_spots': cfg['europe'], 'relegation_spots': cfg['relegate'],
        'runs': runs, 'table': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=10000)
    ap.add_argument('--league', nargs='+', default=None, help='cleaned CSV filenames')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    names = args.league or list(STRUCTURE)
    rng   = np.random.default_rng(20260911)

    print("=" * 64)
    print(f"  Season simulation  |  {args.runs:,} runs per league")
    print("=" * 64)

    out = []
    for nm in names:
        p = CLEANED / nm
        if not p.exists():
            print(f"  skip {nm} — file not found")
            continue
        r = run_league(p, args.runs, rng)
        if r:
            out.append(r)

    if out and not args.no_save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            'generated': datetime.now().isoformat(timespec='seconds'),
            'runs': args.runs,
            'leagues': out,
        }, indent=1))
        print(f"\n  saved -> {OUT_PATH}")
    print()


if __name__ == '__main__':
    main()
