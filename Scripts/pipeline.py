"""
ScoreCast Pipeline  —  Scrape → Clean → Predict
────────────────────────────────────────────────
Two data sources:
  footballdata (default) — football-data.co.uk over plain HTTP. Takes ~30s,
                           needs no browser, and writes Cleaned Datasets
                           directly, so the scrape and clean steps are skipped.
  fbref                  — the Selenium scrape in Scrapping.py. Takes ~1.5h and
                           needs a manual Cloudflare tick, but is the only
                           source carrying pk/pkatt, possession, formation and
                           xG (required by the RandomForest in Modelling.py).

Usage:
  python pipeline.py                                   # fetch all + predict (~30s)
  python pipeline.py --source fbref --all              # full browser scrape
  python pipeline.py --dry-run                         # preview without running
  python pipeline.py --leagues "Brazil Serie A" "Norway Eliteserien"
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
SCRAPPED = ROOT / 'Datasets' / 'Scrapped Datasets'
CLEANED  = ROOT / 'Datasets' / 'Cleaned Datasets'
PREDS    = ROOT / 'Datasets' / 'Predictions'

# ── Config ────────────────────────────────────────────────────────────────────
STALE_DAYS   = 7   # re-scrape if CSV is older than this
RECENT_DAYS  = 30  # if last match was within this window, treat as active

# ── League definitions ────────────────────────────────────────────────────────
# Derived from the shared registry in leagues.py — one definition for the
# fetcher, this pipeline, the simulator and the web app.
from leagues import LEAGUES as _REG

PIPELINE_LEAGUES = [
    {
        'name':         f"{l['name']} ({l['region']})",
        'key':          l['key'],
        'scrapped':     SCRAPPED / l['file'],
        'cleaned_name': l['file'],
        'pred_file':    l['pred'],
    }
    for l in _REG
]


# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging():
    log_path = ROOT / 'pipeline.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding='utf-8'),
        ],
    )

log = logging.getLogger(__name__)

# ── Smart stale check ─────────────────────────────────────────────────────────
def _needs_update(league, force=False):
    """
    Returns (should_update: bool, reason: str).

    Decision logic:
    1. No scrapped file → always update (first run)
    2. force=True       → always update
    3. File < STALE_DAYS old → skip (recently done)
    4. Has future fixtures → update (active season, new results came in)
    5. Last match < RECENT_DAYS ago → update (season just ended or between rounds)
    6. Otherwise → skip (deep off-season, nothing new expected)
    """
    path = league['scrapped']

    if not path.exists():
        return True, 'first run'
    if force:
        return True, 'forced'

    age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
    if age_days < STALE_DAYS:
        return False, f'fresh ({age_days:.0f}d old)'

    try:
        df = pd.read_csv(path, usecols=['Date'], low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        df = df[df['date'].astype(str).str.lower() != 'date']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        today = pd.Timestamp.now().normalize()

        if (df['date'] > today).any():
            return True, f'active season ({age_days:.0f}d old)'

        days_since_last = (today - df['date'].dropna().max()).days
        if days_since_last <= RECENT_DAYS:
            return True, f'recent ({days_since_last}d since last match)'

        return False, f'off-season (last match {days_since_last}d ago)'

    except Exception as e:
        return True, f'stale — could not read dates: {e}'

# ── Pipeline steps ────────────────────────────────────────────────────────────
def _divider(char='=', width=60):
    log.info(char * width)

def step_scrape(leagues, dry_run=False):
    _divider('-')
    log.info(f'  SCRAPE    {len(leagues)} leagues  (fbref via Chrome)')
    _divider('-')

    if dry_run:
        for l in leagues:
            log.info(f'  [dry]  would scrape  {l["name"]}')
        return True

    from Scrapping import scrape_data
    url_filter = {l['url'] for l in leagues}
    try:
        scrape_data(url_filter=url_filter)
        return True
    except Exception as e:
        log.error(f'  Scrape failed: {e}')
        return False

def step_fetch(leagues, dry_run=False):
    _divider('-')
    log.info(f'  FETCH     {len(leagues)} leagues  (football-data.co.uk)')
    _divider('-')

    import requests
    from FootballData import LEAGUES as FD_LEAGUES, fetch_league, CLEANED, BASE, HEADERS, _get_csv

    # Match on the registry key, not the display name: display names carry the
    # region ('Serie A (Italy)') while the fetcher's do not, and there are
    # several leagues called Serie A.
    by_key      = {c['key']: c for c in FD_LEAGUES}
    unsupported = [l for l in leagues if l.get('key') not in by_key]
    for l in unsupported:
        log.warning(f"  SKIP  {l['name']} — not published by football-data.co.uk "
                    f"(use --source fbref)")

    targets = [by_key[l['key']] for l in leagues if l.get('key') in by_key]

    if dry_run:
        for c in targets:
            log.info(f'  [dry]  would fetch  {c["name"]}')
        return True

    if not targets:
        log.error('  No leagues available from this source.')
        return False

    CLEANED.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    fixtures_main  = _get_csv(session, f'{BASE}/fixtures.csv')
    fixtures_extra = _get_csv(session, f'{BASE}/new_league_fixtures.csv')

    ok = 0
    for cfg in targets:
        try:
            df = fetch_league(session, cfg, fixtures_main, fixtures_extra)
            if df is None or df.empty:
                log.error(f'  FAILED  {cfg["name"]} — empty response')
                continue
            df = df.sort_values('date').reset_index(drop=True)
            df.to_csv(CLEANED / cfg['out'])
            played   = int(df['gf'].notna().sum() // 2)
            upcoming = int(df['gf'].isna().sum() // 2)
            log.info(f'  OK    {cfg["name"]:24s} {played:5d} played  {upcoming:3d} upcoming')
            ok += 1
        except Exception as e:
            log.error(f'  FAILED  {cfg["name"]}: {e}')

    log.info(f'  Fetched {ok}/{len(targets)} leagues')
    return ok > 0


def step_clean(leagues, dry_run=False):
    _divider('-')
    log.info(f'  CLEAN     {len(leagues)} leagues')
    _divider('-')

    from Cleaning import read_file, cleaning, export_df

    success = 0
    for league in leagues:
        if dry_run:
            log.info(f'  [dry]  would clean  {league["cleaned_name"]}')
            continue
        try:
            df = read_file(str(league['scrapped']))
            df = cleaning(df)
            export_df(str(league['scrapped']), df, league['cleaned_name'])
            success += 1
        except Exception as e:
            log.error(f'  FAILED  {league["name"]}: {e}')

    if not dry_run:
        log.info(f'  Cleaned {success}/{len(leagues)} leagues')
    return True

def step_predict(leagues, dry_run=False):
    _divider('-')
    log.info(f'  PREDICT   {len(leagues)} leagues')
    _divider('-')

    from ScorelineModel import run_league

    success = 0
    for league in leagues:
        cleaned_path = str(CLEANED / league['cleaned_name'])
        output_path  = str(PREDS   / league['pred_file'])

        if dry_run:
            log.info(f'  [dry]  would predict  {league["name"]}')
            continue
        if not os.path.exists(cleaned_path):
            log.warning(f'  SKIP  {league["name"]} — cleaned file not found')
            continue
        try:
            run_league(cleaned_path, output_path, league['name'])
            success += 1
        except Exception as e:
            log.error(f'  FAILED  {league["name"]}: {e}')

    if not dry_run:
        log.info(f'  Predicted {success}/{len(leagues)} leagues')
    return True

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ScoreCast data pipeline')
    parser.add_argument('--all',     action='store_true', help='force all leagues')
    parser.add_argument('--dry-run', action='store_true', help='preview without writing')
    parser.add_argument('--leagues', nargs='+', metavar='NAME',
                        help='run specific leagues by name')
    parser.add_argument('--source', choices=['footballdata', 'fbref'],
                        default='footballdata',
                        help='where match data comes from (default: footballdata)')
    args = parser.parse_args()

    _setup_logging()

    _divider()
    log.info('  ScoreCast Pipeline')
    log.info(f'  {datetime.now().strftime("%A, %d %B %Y  %H:%M")}')
    log.info(f'  Source: {args.source}')
    if args.dry_run:
        log.info('  MODE: dry-run - nothing will be written')
    _divider()

    # Optional league filter via --leagues
    pool = PIPELINE_LEAGUES
    if args.leagues:
        names_lower = {n.lower() for n in args.leagues}
        pool = [l for l in pool if l['name'].lower() in names_lower]
        if not pool:
            log.error(f'No leagues matched: {args.leagues}')
            log.info(f'Available: {[l["name"] for l in PIPELINE_LEAGUES]}')
            sys.exit(1)

    start = datetime.now()

    if args.source == 'footballdata':
        # A full fetch takes ~30s, so the staleness heuristics that exist to
        # avoid a 1.5h browser scrape would only add a chance of going stale.
        step_fetch(pool,   dry_run=args.dry_run)
        step_predict(pool, dry_run=args.dry_run)
    else:
        log.info('  Checking leagues...\n')
        to_update, skipped = [], []
        for league in pool:
            update, reason = _needs_update(league, force=args.all)
            tag = 'UPDATE' if update else 'SKIP  '
            log.info(f'  {tag}  {league["name"]:<28}  {reason}')
            (to_update if update else skipped).append(league)

        if not to_update:
            log.info('\n  All leagues are up to date. Nothing to do.')
            _divider()
            return

        log.info(f'\n  {len(to_update)} to update  |  {len(skipped)} skipped\n')

        step_scrape(to_update,  dry_run=args.dry_run)
        step_clean(to_update,   dry_run=args.dry_run)
        step_predict(to_update, dry_run=args.dry_run)

    elapsed = round((datetime.now() - start).seconds / 60, 1)
    _divider()
    log.info(f'  Pipeline complete  |  {elapsed} min')
    _divider()


if __name__ == '__main__':
    main()
