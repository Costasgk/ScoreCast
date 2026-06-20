"""
ScoreCast Pipeline  —  Scrape → Clean → Predict
────────────────────────────────────────────────
Usage:
  python pipeline.py                                   # auto-detect stale leagues
  python pipeline.py --all                             # force all 10 leagues
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
PIPELINE_LEAGUES = [
    {
        'name':         'Brazil Serie A',
        'url':          'https://fbref.com/en/comps/24/Serie-A-Stats',
        'scrapped':     SCRAPPED / 'Serie_A_Stats.csv',
        'cleaned_name': 'Brazil_Serie_A.csv',
        'pred_file':    'predictions_brazil_serie_a.csv',
    },
    {
        'name':         'Brazil Serie B',
        'url':          'https://fbref.com/en/comps/38/Serie-B-Stats',
        'scrapped':     SCRAPPED / 'Serie_B_Stats.csv',
        'cleaned_name': 'Brazil_Serie_B.csv',
        'pred_file':    'predictions_brazil_serie_b.csv',
    },
    {
        'name':         'Norway Eliteserien',
        'url':          'https://fbref.com/en/comps/28/Eliteserien-Stats',
        'scrapped':     SCRAPPED / 'Eliteserien_Stats.csv',
        'cleaned_name': 'Norway_Eliteserien.csv',
        'pred_file':    'predictions_norway_eliteserien.csv',
    },
    {
        'name':         'Finland Veikkausliiga',
        'url':          'https://fbref.com/en/comps/43/Veikkausliiga-Stats',
        'scrapped':     SCRAPPED / 'Veikkausliiga_Stats.csv',
        'cleaned_name': 'Finland_Veikkausliiga.csv',
        'pred_file':    'predictions_finland_veikkausliiga.csv',
    },
    {
        'name':         'Greece Super League',
        'url':          'https://fbref.com/en/comps/27/Super-League-Greece-Stats',
        'scrapped':     SCRAPPED / 'Super_League_Greece_Stats.csv',
        'cleaned_name': 'Greece_Super_League.csv',
        'pred_file':    'predictions_greece_super_league.csv',
    },
    {
        'name':         'England Premier League',
        'url':          'https://fbref.com/en/comps/9/Premier-League-Stats',
        'scrapped':     SCRAPPED / 'Premier_League_Stats.csv',
        'cleaned_name': 'England_Premier_League.csv',
        'pred_file':    'predictions_england_premier_league.csv',
    },
    {
        'name':         'Italy Serie A',
        'url':          'https://fbref.com/en/comps/11/Serie-A-Stats',
        'scrapped':     SCRAPPED / 'Serie_A_Stats_Italy.csv',
        'cleaned_name': 'Italy_Serie_A.csv',
        'pred_file':    'predictions_italy_serie_a.csv',
    },
    {
        'name':         'Spain La Liga',
        'url':          'https://fbref.com/en/comps/12/La-Liga-Stats',
        'scrapped':     SCRAPPED / 'La_Liga_Stats.csv',
        'cleaned_name': 'Spain_La_Liga.csv',
        'pred_file':    'predictions_spain_la_liga.csv',
    },
    {
        'name':         'Germany Bundesliga',
        'url':          'https://fbref.com/en/comps/20/Bundesliga-Stats',
        'scrapped':     SCRAPPED / 'Bundesliga_Stats.csv',
        'cleaned_name': 'Germany_Bundesliga.csv',
        'pred_file':    'predictions_germany_bundesliga.csv',
    },
    {
        'name':         'France Ligue 1',
        'url':          'https://fbref.com/en/comps/13/Ligue-1-Stats',
        'scrapped':     SCRAPPED / 'Ligue_1_Stats.csv',
        'cleaned_name': 'France_Ligue_1.csv',
        'pred_file':    'predictions_france_ligue_1.csv',
    },
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
    log.info(f'  STEP 1 / SCRAPE   {len(leagues)} leagues')
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

def step_clean(leagues, dry_run=False):
    _divider('-')
    log.info(f'  STEP 2 / CLEAN    {len(leagues)} leagues')
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
    log.info(f'  STEP 3 / PREDICT  {len(leagues)} leagues')
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
    args = parser.parse_args()

    _setup_logging()

    _divider()
    log.info('  ScoreCast Pipeline')
    log.info(f'  {datetime.now().strftime("%A, %d %B %Y  %H:%M")}')
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

    # Decide which leagues need updating
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

    start = datetime.now()

    step_scrape(to_update,  dry_run=args.dry_run)
    step_clean(to_update,   dry_run=args.dry_run)
    step_predict(to_update, dry_run=args.dry_run)

    elapsed = round((datetime.now() - start).seconds / 60, 1)
    _divider()
    log.info(f'  Pipeline complete  |  {elapsed} min')
    _divider()


if __name__ == '__main__':
    main()
