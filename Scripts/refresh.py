"""
One command that brings everything up to date.

Fetch results and fixtures, refit every league, project the remaining seasons,
and — weekly — rescore the model against the closing price. Designed to be run
unattended from cron, so it never prompts, never depends on a browser, and
reports what it did in a form that is readable months later in a log file.

The steps are ordered by how quickly they go stale. Predictions matter daily,
season projections shift with every round, and the benchmark moves slowly
enough that recomputing it more than weekly is waste.

Nothing is written until a step succeeds: a failed fetch leaves yesterday's
predictions in place rather than replacing them with nothing, which matters when
this runs with nobody watching.

Usage:
  python refresh.py                  # fetch + predict + simulate
  python refresh.py --full           # also rescore against the market
  python refresh.py --quick          # fetch + predict only
  python refresh.py --install-cron   # print a crontab line and exit
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY   = ROOT / 'env' / 'bin' / 'python'
LOG  = ROOT / 'refresh.log'


def _python():
    """The project venv if it exists, otherwise whatever is running this."""
    return str(PY) if PY.exists() else sys.executable


def log(msg, echo=True):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    if echo:
        print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def run(label, args, timeout=3600):
    """Run one step. Returns True on success; never raises."""
    log(f"START  {label}")
    t0 = time.time()
    try:
        p = subprocess.run([_python(), *args], cwd=str(HERE),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"TIMND  {label} — exceeded {timeout}s")
        return False
    except Exception as e:
        log(f"ERROR  {label} — {type(e).__name__}: {e}")
        return False

    mins = (time.time() - t0) / 60
    if p.returncode != 0:
        log(f"FAIL   {label} ({mins:.1f} min, exit {p.returncode})")
        for ln in (p.stderr or '').strip().splitlines()[-6:]:
            log(f"         {ln}")
        return False

    # Keep the informative lines; drop the progress noise
    for ln in (p.stdout or '').splitlines():
        t = ln.strip()
        if t.startswith(('OK', 'FAIL', 'skip', 'saved')) or 'predictions ->' in t:
            log(f"         {t}")
    log(f"OK     {label} ({mins:.1f} min)")
    return True


CRON = """# ScoreCast — refresh predictions daily at 06:15, full rescore Mondays at 04:00
15 6 * * *  cd {root} && {py} Scripts/refresh.py >> {log} 2>&1
0  4 * * 1  cd {root} && {py} Scripts/refresh.py --full >> {log} 2>&1"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='also rescore against closing odds (slow, weekly)')
    ap.add_argument('--quick', action='store_true',
                    help='fetch and predict only, skip season projections')
    ap.add_argument('--runs', type=int, default=10000,
                    help='simulations per league')
    ap.add_argument('--install-cron', action='store_true',
                    help='print a crontab entry and exit')
    args = ap.parse_args()

    if args.install_cron:
        print(CRON.format(root=ROOT, py=_python(), log=LOG))
        print("\nAdd it with:  crontab -e")
        return

    log("=" * 58, echo=False)
    log(f"REFRESH  mode={'full' if args.full else 'quick' if args.quick else 'standard'}")

    t0 = time.time()
    ok = {}

    # 1. Results and fixtures. Everything downstream reads what this writes, so
    #    if it fails there is nothing worth recomputing.
    ok['fetch'] = run('fetch results and fixtures', ['FootballData.py'], timeout=1800)
    if not ok['fetch']:
        log('ABORT  fetch failed — leaving existing predictions in place')
        log(f"REFRESH ended after {(time.time() - t0) / 60:.1f} min")
        return 1

    # 2. Full-season fixture dates from ESPN. Optional: without it the rest of
    #    the season still lists, just without kickoffs.
    ok['dates'] = run('fetch fixture dates', ['fixtures.py'], timeout=2400)

    # 3. Match predictions, football then basketball.
    ok['football'] = run('predict football fixtures', ['-c', PREDICT_SNIPPET], timeout=3600)
    ok['basket']   = run('predict basketball fixtures',
                         ['Euroleague_scripts/run.py'], timeout=1800)

    # 4. Season projections — slower, and only meaningful for round-robin leagues.
    if not args.quick:
        ok['season'] = run('project remaining seasons',
                           ['simulate_season.py', '--runs', str(args.runs)], timeout=3600)

    # 5. Scoring against the market barely moves week to week.
    if args.full:
        ok['benchmark'] = run('rescore against closing odds',
                              ['benchmark.py', '--all', '--save'], timeout=5400)

    good = sum(1 for v in ok.values() if v)
    log(f"REFRESH complete — {good}/{len(ok)} steps ok, "
        f"{(time.time() - t0) / 60:.1f} min")
    return 0 if good == len(ok) else 1


# Run inline rather than as a separate file: the per-league prediction loop is
# only used here and by the pipeline, and duplicating it invites drift.
PREDICT_SNIPPET = """
import sys, json; sys.path.insert(0, '.')
from pathlib import Path
from leagues import LEAGUES
from ScorelineModel import run_league
_dp = Path('../Datasets/Models/fixture_dates.json')
HAVE_DATES = set(json.loads(_dp.read_text()).get('leagues', {})) if _dp.exists() else set()
CLEANED = Path('../Datasets/Cleaned Datasets')
PREDS   = Path('../Datasets/Predictions'); PREDS.mkdir(parents=True, exist_ok=True)
ok = 0
for l in LEAGUES:
    src = CLEANED / l['file']
    if not src.exists():
        continue
    try:
        # A published ESPN calendar works for any format; without one, only a
        # round-robin league can have its remaining fixtures derived.
        run_league(str(src), str(PREDS / l['pred']), f"{l['name']} ({l['region']})",
                   include_unscheduled=l.get('round_robin', False) or l['key'] in HAVE_DATES,
                   league_key=l['key'])
        ok += 1
    except Exception as e:
        print(f"FAIL {l['key']}: {type(e).__name__}: {e}")
print(f"OK {ok} leagues predicted")
"""


if __name__ == '__main__':
    sys.exit(main() or 0)
