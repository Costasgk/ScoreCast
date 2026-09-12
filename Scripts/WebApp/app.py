from flask import Flask, render_template, send_file, jsonify, request, Response
import re
import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote as _urlquote
import json
import os
import sqlite3
import base64
import hmac
import functools
import yaml

# Leagues come from the shared registry so the fetcher, pipeline, simulator
# and this app cannot drift apart.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leagues import LEAGUES as REGISTRY

app = Flask(__name__)

# ── Auth ───────────────────────────────────────────────────────────────────────

def _load_config():
    path = Path(__file__).resolve().parent / 'config.yaml'
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}

_cfg             = _load_config()
_STATS_USERNAME  = _cfg.get('stats', {}).get('username', '') or os.environ.get('STATS_USERNAME', '')
_STATS_PASSWORD  = _cfg.get('stats', {}).get('password', '') or os.environ.get('STATS_PASSWORD', '')

# Whether an unprotected page is acceptable. Opening up has to be deliberate:
# the previous default let anyone through whenever no password happened to be
# configured, which is the wrong way round once this is reachable from the web.
_OPEN_STATS = os.environ.get('SCORECAST_OPEN_STATS', '').lower() in ('1', 'true', 'yes')


def _require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not _STATS_PASSWORD:
            if _OPEN_STATS:
                return f(*args, **kwargs)
            return Response(
                'Stats are disabled: no credentials configured.\n'
                'Set STATS_USERNAME and STATS_PASSWORD, or SCORECAST_OPEN_STATS=1 '
                'to allow unauthenticated access on a local machine.\n',
                503, {'Content-Type': 'text/plain'})
        auth = request.headers.get('Authorization', '')
        if auth.startswith('Basic '):
            try:
                username, password = base64.b64decode(auth[6:]).decode('utf-8').split(':', 1)
                if (hmac.compare_digest(username, _STATS_USERNAME) and
                        hmac.compare_digest(password, _STATS_PASSWORD)):
                    return f(*args, **kwargs)
            except Exception:
                pass
        return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="ScoreCast Stats"'})
    return decorated

# ── Visitor tracking ───────────────────────────────────────────────────────────

_BOTS = ('bot', 'crawler', 'spider', 'python', 'curl', 'wget', 'scrapy', 'httpclient')

def _db_path():
    # Configurable so a container can point it at a writable volume: the app
    # directory is read-only in a sane image, and a counter that resets on every
    # restart is worse than no counter.
    env = os.environ.get('SCORECAST_STATE_DIR')
    base = Path(env) if env else Path(__file__).resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    return base / 'visits.db'

def _init_db():
    with sqlite3.connect(_db_path()) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                ts   TEXT NOT NULL,
                ip   TEXT
            )
        ''')
        # migrate existing DB that has no ip column
        try:
            conn.execute('ALTER TABLE visits ADD COLUMN ip TEXT')
        except sqlite3.OperationalError:
            pass

_init_db()

_DEDUP_MINUTES = 30  # same IP on same page within this window = 1 visit

@app.before_request
def _record_visit():
    if (request.path.startswith('/static') or request.path.startswith('/api/')
            or request.path == '/stats' or request.path == '/favicon.ico'):
        return
    ua = request.headers.get('User-Agent', '').lower()
    if any(b in ua for b in _BOTS):
        return

    # respect X-Forwarded-For if behind a proxy/nginx
    raw_ip = request.headers.get('X-Forwarded-For', request.remote_addr) or ''
    ip = raw_ip.split(',')[0].strip()

    now = datetime.now(timezone.utc)
    ts  = now.strftime('%Y-%m-%d %H:%M:%S')
    dedup_cutoff = (now - timedelta(minutes=_DEDUP_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')

    with sqlite3.connect(_db_path()) as conn:
        already = conn.execute(
            'SELECT 1 FROM visits WHERE ip = ? AND path = ? AND ts >= ? LIMIT 1',
            (ip, request.path, dedup_cutoff)
        ).fetchone()
        if not already:
            conn.execute('INSERT INTO visits (path, ts, ip) VALUES (?, ?, ?)',
                         (request.path, ts, ip))

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent

def _predictions_dir():
    return _ROOT / 'Datasets' / 'Predictions'

def _models_dir():
    return _ROOT / 'Datasets' / 'Models'

def _cleaned_dir():
    return _ROOT / 'Datasets' / 'Cleaned Datasets'

# ── Display columns ────────────────────────────────────────────────────────────

DISPLAY_COLS = [
    'Date', 'Home Team', 'Away Team',
    'Home Win %', 'Draw %', 'Away Win %',
    'xG Home', 'xG Away',
    'BTTS %', 'Over 1.5 %', 'Over 2.5 %', 'Over 3.5 %',
    'Most Likely Score', 'Top 3 Scores',
]

def _load(filename):
    """
    (full frame, scheduled table, rest-of-season table, counts).

    football-data publishes only the next matchweek, so the rest of the season
    is derived from the pairings that have not been played. Those are certain to
    happen but have no date yet, and they are kept in a separate table rather
    than mixed in — a fixture with a kickoff time and one without are different
    things to a reader.
    """
    path = _predictions_dir() / filename
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if 'Scheduled' not in df.columns:
        df['Scheduled'] = df['Date'].notna()
    if 'Rated' not in df.columns:
        df['Rated'] = True

    # 'Scheduled' records where a fixture came from — football-data publishes
    # only the next matchweek — not whether it has a kickoff. That was the same
    # thing until the ESPN and FBref calendars started supplying dates for the
    # whole season; since then, splitting on the flag files a dated game under
    # 'rest of season' whenever football-data happens not to list it. Today's
    # St Pauli v Wolfsburg sat below 262 undated rows while the three other
    # Bundesliga 2 games that afternoon sat on top.
    #
    # So split on the matchweek instead of the source: whatever falls on or
    # before the last day football-data has published belongs in the first
    # table, whoever supplied it. A matchday then arrives whole.
    flagged = df['Scheduled'].astype(bool)
    window  = df.loc[flagged, 'Date'].max()
    if pd.notna(window):
        flagged = flagged | (df['Date'].notna()
                             & (df['Date'].dt.normalize() <= window.normalize()))

    sched = df[flagged].sort_values('Date')
    # No dates to sort on. Confidence order looks sensible and reads terribly:
    # the weakest side in the division owns the top of the list, because every
    # home game against them is a near-certainty. Alphabetical by home team
    # keeps a club's fixtures together, which is what the table is for.
    rest = df[~flagged]
    rest = (rest.sort_values('Date') if rest['Date'].notna().any()
            else rest.sort_values(['Home Team', 'Away Team']))

    def _dates(t):
        """
        Render the date column ourselves.

        to_html's na_rep does not reach a datetime column — pandas formats those
        through its own path, which spells a missing value 'NaT'. That was fine
        while every row in a table either had a date or none did; now that a
        calendar can be topped up with fixtures whose day is not announced yet,
        the two mix in one table and 38 rows of 'NaT' appear on the page.
        """
        if 'Date' not in t.columns or not len(t):
            return t
        t = t.copy()
        d = t['Date']
        with_time = (d.dt.hour != 0) | (d.dt.minute != 0)
        t['Date'] = (d.dt.strftime('%Y-%m-%d %H:%M')
                     .where(with_time, d.dt.strftime('%Y-%m-%d'))
                     .fillna('—'))
        return t

    cols = [c for c in DISPLAY_COLS if c in df.columns]
    # ESPN supplies kickoffs for most leagues, so the rest of the season usually
    # has dates now. Drop the column only where nothing in it is filled.
    rest_cols = cols if (len(rest) and rest['Date'].notna().any()) \
                else [c for c in cols if c != 'Date']
    return df, _dates(sched[cols]), _dates(rest[rest_cols]), {
        'scheduled': int(len(sched)), 'rest': int(len(rest)),
        'unrated':   int((~df['Rated']).sum()),
        'rest_dated': int(rest['Date'].notna().sum()) if len(rest) else 0,
    }

# ── League list ────────────────────────────────────────────────────────────────

# Football leagues come from the shared registry in Scripts/leagues.py so the
# fetcher, pipeline, simulator and web app cannot drift apart.
LEAGUES = [(l['slug'], l['pred'], f"{l['name']} ({l['region']})") for l in REGISTRY]
# ── Basketball (Euroleague / Eurocup) ──────────────────────────────────────────
# Kept separate from LEAGUES: the ratings model outputs points, not goals, so
# these pages carry different columns and a different template.

BASKETBALL = [
    ('Euroleague', 'predictions_euroleague.csv', 'EuroLeague'),
    ('Eurocup',    'predictions_eurocup.csv',    'EuroCup'),
]

BASKET_COLS = [
    'Date', 'Round', 'Home Team', 'Away Team',
    'Home Win %', 'Away Win %',
    'Pred Home Score', 'Pred Away Score', 'Pred Total', 'Pred Margin',
    'Pick', 'Confidence',
]

def _load_basket(filename):
    path = _predictions_dir() / filename
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    cold = int(df['Cold Start'].sum()) if 'Cold Start' in df.columns else 0
    show = df[[c for c in BASKET_COLS if c in df.columns]].copy()
    show['Date'] = show['Date'].dt.strftime('%Y-%m-%d %H:%M')
    return df, show, cold


# ── Home ───────────────────────────────────────────────────────────────────────

def _is_active(path):
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=['Date'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return (df['Date'] > pd.Timestamp.today()).any()
    except Exception:
        return False

# Region shown alongside each competition on the index
REGIONS = {l['slug']: l['region'] for l in REGISTRY}
REGIONS.update({'Euroleague': 'Europe', 'Eurocup': 'Europe'})

def _fixture_count(path):
    """Upcoming fixtures in a predictions file, or 0 if it has none."""
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path, usecols=['Date'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return int((df['Date'] > pd.Timestamp.today()).sum())
    except Exception:
        return 0


def _index_rows(entries):
    rows = []
    for slug, filename, title in entries:
        n = _fixture_count(_predictions_dir() / filename)
        # The region is shown separately, so 'Serie A (Brazil)' reads as a stutter
        name = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
        rows.append({'slug': slug, 'title': name,
                     'region': REGIONS.get(slug, ''), 'count': n, 'active': n > 0})
    return rows


def _grouped(rows):
    """
    Leagues grouped by country, for a grid rather than one long column.

    Thirty-six leagues in a single list put basketball two thousand pixels below
    the fold. Grouping by country gives the grid a natural cell — England has
    five divisions, most countries have one — and keeps related competitions
    together instead of alphabetically scattered.
    """
    order = {l['slug']: i for i, l in enumerate(REGISTRY)}
    by = {}
    for r in rows:
        by.setdefault(r['region'], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: order.get(r['slug'], 999))
    # Countries with more divisions first, then alphabetically
    return sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _strongest_calls():
    """
    The most confident prediction on the soonest day that has fixtures.

    Ranking every upcoming fixture by confidence alone surfaces whichever match
    the model happens to like most, which is usually months away — a EuroLeague
    tie in December is not what someone opening the site today wants to see.
    So the earliest matchday with fixtures is found first, and the strongest
    call is chosen from that day only. Today when today has games, otherwise
    the next day that does.
    """
    now   = pd.Timestamp.now()
    today = now.normalize()
    rows  = []

    for slug, filename, title in LEAGUES + BASKETBALL:
        path = _predictions_dir() / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if 'Home Win %' not in df.columns:
                continue
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])

            # football-data publishes dates without kickoff times, so today's
            # fixtures are stored at midnight. Comparing those against the
            # current time drops the whole day from mid-morning onwards, which
            # is why the hero skipped to tomorrow while today still had games.
            # A fixture with a real time is judged on that time; a date-only one
            # counts for the whole of its day.
            has_time = (df['Date'].dt.hour != 0) | (df['Date'].dt.minute != 0)
            df = df[(has_time & (df['Date'] >= now))
                    | (~has_time & (df['Date'].dt.normalize() >= today))]
            if df.empty:
                continue
            df = df.assign(_top=df[['Home Win %', 'Away Win %']].max(axis=1),
                           _day=df['Date'].dt.normalize(),
                           _league=title, _slug=slug)
            rows.append(df)
        except Exception:
            continue

    if not rows:
        return []

    allf = pd.concat(rows, ignore_index=True)
    days = sorted(allf['_day'].unique())[:2]
    return [_call_for(allf, d, today) for d in days]


def _call_for(allf, day, today):
    """The most confident fixture on one matchday."""
    same = allf[allf['_day'] == day]
    row  = same.loc[same['_top'].idxmax()]

    has_draw   = 'Draw %' in row.index and pd.notna(row.get('Draw %'))
    home_leads = row['Home Win %'] >= row['Away Win %']
    delta      = (day - today).days
    when       = ('Today' if delta == 0 else
                  'Tomorrow' if delta == 1 else
                  row['Date'].strftime('%a %d %b'))

    return {
        'league': row['_league'], 'slug': row['_slug'],
        'home': row['Home Team'], 'away': row['Away Team'],
        'date': when,
        'time': row['Date'].strftime('%H:%M') if row['Date'].hour or row['Date'].minute else '',
        'n_today': int(len(same)),
        'pct': float(row['_top']),
        'pick': row['Home Team'] if home_leads else row['Away Team'],
        'home_pct': float(row['Home Win %']),
        'draw_pct': float(row['Draw %']) if has_draw else 0.0,
        'away_pct': float(row['Away Win %']),
        'home_leads': bool(home_leads),
        'has_draw': bool(has_draw),
    }


@app.route('/', methods=['GET'])
def display_home():
    today    = date.today().strftime('%d %B %Y')
    pred_dir = _predictions_dir()
    status   = {slug: _is_active(pred_dir / filename) for slug, filename, _ in LEAGUES}
    status.update({slug: _is_active(pred_dir / filename) for slug, filename, _ in BASKETBALL})
    return render_template(
        'index.html',
        today=today, status=status,
        football=_grouped(_index_rows(LEAGUES)),
        basketball=_grouped(_index_rows(BASKETBALL)),
        n_football=len(LEAGUES), n_basketball=len(BASKETBALL),
        n_football_active=sum(1 for r in _index_rows(LEAGUES) if r['active']),
        n_basketball_active=sum(1 for r in _index_rows(BASKETBALL) if r['active']),
        calls=_strongest_calls(),
    )

# ── League routes ──────────────────────────────────────────────────────────────

def _make_route(slug, filename, title):
    def view():
        path = _predictions_dir() / filename
        if not path.exists():
            return render_template('league.html',
                                   title=title, league_slug=slug,
                                   has_predictions=False, table_html='')
        _, sched_df, rest_df, counts = _load(filename)
        table_html = sched_df.to_html(index=False, classes='', na_rep='—', border=0)
        rest_html  = (rest_df.to_html(index=False, classes='', na_rep='—', border=0)
                      if len(rest_df) else '')
        return render_template('league.html',
                               title=title, league_slug=slug,
                               has_predictions=True, table_html=table_html,
                               rest_html=rest_html, counts=counts)

    def download():
        path = _predictions_dir() / filename
        return send_file(path, as_attachment=True, download_name=filename)

    view.__name__     = f'view_{slug}'
    download.__name__ = f'download_{slug}'
    return view, download

for slug, filename, title in LEAGUES:
    view_fn, dl_fn = _make_route(slug, filename, title)
    app.add_url_rule(f'/{slug}',     view_fn.__name__, view_fn, methods=['GET'])
    app.add_url_rule(f'/{slug}-csv', dl_fn.__name__,   dl_fn,   methods=['GET'])

# ── Basketball routes ──────────────────────────────────────────────────────────

def _make_basket_route(slug, filename, title):
    def view():
        path = _predictions_dir() / filename
        if not path.exists():
            return render_template('basketball.html', title=title, league_slug=slug,
                                   has_predictions=False, table_html='', cold=0, n=0)
        _, display_df, cold = _load_basket(filename)
        table_html = display_df.to_html(index=False, classes='', na_rep='—', border=0)
        return render_template('basketball.html', title=title, league_slug=slug,
                               has_predictions=True, table_html=table_html,
                               cold=cold, n=len(display_df))

    def download():
        return send_file(_predictions_dir() / filename,
                         as_attachment=True, download_name=filename)

    view.__name__     = f'view_basket_{slug}'
    download.__name__ = f'download_basket_{slug}'
    return view, download


for slug, filename, title in BASKETBALL:
    v, d = _make_basket_route(slug, filename, title)
    app.add_url_rule(f'/{slug}',     v.__name__, v, methods=['GET'])
    app.add_url_rule(f'/{slug}-csv', d.__name__, d, methods=['GET'])


# ── Model vs Market ────────────────────────────────────────────────────────────
# Competing sites publish a headline accuracy percentage and no validation.
# This page publishes the part that actually decides whether a forecast is any
# good: calibration and sharpness, scored against the closing price.

def _benchmark():
    path = _models_dir() / 'benchmark.json'
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@app.route('/model-vs-market', methods=['GET'])
def display_benchmark():
    data = _benchmark()
    if not data:
        return render_template('benchmark.html', has_data=False)

    leagues = sorted(data['leagues'],
                     key=lambda r: r['model']['logloss'] - r['market']['logloss'])
    for r in leagues:
        r['gap_ll']    = r['model']['logloss'] - r['market']['logloss']
        r['gap_brier'] = r['model']['brier']   - r['market']['brier']
        r['beats_uniform'] = r['model']['logloss'] < r['uniform']['logloss']

    n_total = sum(r['model']['n'] for r in leagues)
    bets    = sum(r['value']['bets'] or 0 for r in leagues)
    staked  = bets
    won     = sum((r['value']['roi'] or 0) * (r['value']['bets'] or 0) for r in leagues)
    overall_roi = (won / staked) if staked else None

    return render_template(
        'benchmark.html',
        has_data=True,
        generated=data.get('generated', '')[:16].replace('T', ' '),
        seasons=', '.join(data.get('seasons', [])),
        leagues=leagues,
        n_total=n_total,
        mean_gap=sum(r['gap_ll'] for r in leagues) / len(leagues),
        best=leagues[0],
        worst=leagues[-1],
        value_bets=bets,
        value_roi=overall_roi,
    )


# ── Player statistics ──────────────────────────────────────────────────────────
# Basketball only. Football player data is not published by football-data.co.uk,
# so there is nothing equivalent to show for the 36 football leagues.

PLAYER_SORTS = {
    'val': ('vpg', 'Valuation'), 'pts': ('ppg', 'Points'),
    'reb': ('rpg', 'Rebounds'),  'ast': ('apg', 'Assists'),
    'min': ('mpg', 'Minutes'),   'ts':  ('ts',  'True shooting'),
    'fg3': ('fg3', 'Three-point %'),
    'pts40': ('pts40', 'Points per 40'),
}
MIN_3PA = 50    # attempts before a three-point percentage means anything


def _players():
    path = _models_dir() / 'players.json'
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@app.route('/players', methods=['GET'])
@app.route('/players/<comp>', methods=['GET'])
def display_players(comp=None):
    data = _players()
    if not data or not data.get('competitions'):
        return render_template('players.html', has_data=False, comps=[])

    comps  = data['competitions']
    chosen = next((c for c in comps if c['comp'] == comp), comps[0])

    sort_key = request.args.get('sort', 'val')
    field, label = PLAYER_SORTS.get(sort_key, PLAYER_SORTS['val'])

    rows = list(chosen['players'])
    if field == 'fg3':
        # Otherwise the leaderboard is whoever took two threes and made both
        rows = [r for r in rows if r.get('fg3a', 0) >= MIN_3PA]
    rows.sort(key=lambda r: -(r.get(field) or 0))

    return render_template(
        'players.html', has_data=True,
        generated=data.get('generated', '')[:16].replace('T', ' '),
        comps=[{'comp': c['comp'], 'name': c['name']} for c in comps],
        c=chosen, rows=rows[:60], sort=sort_key, sort_label=label,
        sorts=PLAYER_SORTS, min_3pa=MIN_3PA,
    )


# ── Head to head ───────────────────────────────────────────────────────────────

def _h2h(cleaned_path, a, b, limit=12):
    """
    Every recorded meeting between two clubs in this competition.

    Only league matches are counted — the cleaned files already exclude cup and
    continental games — so the record answers "in this league" rather than
    "ever", which is the comparison a fixture page implies.
    """
    df = pd.read_csv(cleaned_path, index_col=0, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    home = df[df['venue'].astype(str).str.lower().str.strip() == 'home']
    m = home[(((home['team'] == a) & (home['opponent'] == b)) |
              ((home['team'] == b) & (home['opponent'] == a)))
             & home['gf'].notna() & home['ga'].notna()]
    m = m.sort_values('date', ascending=False)

    a_w = b_w = dr = a_g = b_g = 0
    a_home_w = b_home_w = 0
    meetings = []
    for _, r in m.iterrows():
        hg, ag = int(r['gf']), int(r['ga'])
        h, aw = str(r['team']), str(r['opponent'])
        if h == a:
            a_g += hg; b_g += ag
        else:
            a_g += ag; b_g += hg
        if hg == ag:
            dr += 1
            res = 'D'
        elif (hg > ag) == (h == a):
            a_w += 1
            if h == a: a_home_w += 1
            res = 'A'
        else:
            b_w += 1
            if h == b: b_home_w += 1
            res = 'B'
        meetings.append({
            'date': r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else '',
            'home': h, 'away': aw, 'hg': hg, 'ag': ag, 'res': res,
            'season': str(r['season']) if 'season' in r and pd.notna(r.get('season')) else '',
        })

    n = len(m)
    return {
        'played': n, 'a_wins': a_w, 'b_wins': b_w, 'draws': dr,
        'a_goals': a_g, 'b_goals': b_g,
        'a_pct': round(a_w / n * 100, 1) if n else 0,
        'd_pct': round(dr / n * 100, 1) if n else 0,
        'b_pct': round(b_w / n * 100, 1) if n else 0,
        'a_home_wins': a_home_w, 'b_home_wins': b_home_w,
        'meetings': meetings[:limit],
    }


@app.route('/h2h/<slug>', methods=['GET'])
def head_to_head(slug):
    row = next(((s_, f, t) for s_, f, t in LEAGUES if s_ == slug), None)
    if not row:
        return 'League not found', 404
    _, pred_file, title = row

    cf = CLEANED_FILES.get(slug)
    cp = _cleaned_dir() / cf if cf else None
    if not cp or not cp.exists():
        return render_template('h2h.html', title=title, slug=slug,
                               teams=[], a=None, b=None, rec=None, fixture=None)

    df = pd.read_csv(cp, index_col=0, low_memory=False)
    teams = sorted(set(df['team'].dropna().astype(str))
                   | set(df['opponent'].dropna().astype(str)))

    a = request.args.get('a') or ''
    b = request.args.get('b') or ''
    rec = _h2h(str(cp), a, b) if (a in teams and b in teams and a != b) else None

    # If these two are due to meet, show the model's line for that fixture
    fixture = None
    pp = _predictions_dir() / pred_file
    if rec and pp.exists():
        try:
            pdf = pd.read_csv(pp)
            hit = pdf[((pdf['Home Team'] == a) & (pdf['Away Team'] == b)) |
                      ((pdf['Home Team'] == b) & (pdf['Away Team'] == a))]
            if len(hit):
                fixture = hit.iloc[0].to_dict()
        except Exception:
            pass

    return render_template('h2h.html', title=title, slug=slug, teams=teams,
                           a=a, b=b, rec=rec, fixture=fixture)


# ── Season projection ──────────────────────────────────────────────────────────

def _season_sim():
    path = _models_dir() / 'season_sim.json'
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@app.route('/season', methods=['GET'])
@app.route('/season/<slug>', methods=['GET'])
def display_season(slug=None):
    data = _season_sim()
    if not data or not data.get('leagues'):
        return render_template('season.html', has_data=False, leagues=[])

    leagues = data['leagues']
    by_slug = {}
    for L in leagues:
        s = L['league'].replace(' ', '-')
        L['slug'] = s
        by_slug[s] = L

    chosen = by_slug.get(slug) or leagues[0]
    return render_template(
        'season.html',
        has_data=True,
        generated=data.get('generated', '')[:16].replace('T', ' '),
        runs=data.get('runs', 0),
        leagues=[{'slug': L['slug'], 'name': L['league'],
                  'regular_only': bool(L.get('regular_season_only'))} for L in leagues],
        L=chosen,
    )


# ── Simulator ──────────────────────────────────────────────────────────────────

def _model_path(pred_filename):
    model_file = pred_filename.replace('predictions_', 'model_').replace('.csv', '.json')
    return _models_dir() / model_file

def _load_model(pred_filename):
    path = _model_path(pred_filename)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def _simulate(model, home_team, away_team):
    if home_team not in model['attack'] or away_team not in model['attack']:
        return None

    lam = float(np.exp(model['attack'][home_team] + model['defense'][away_team] + model['home_adv']))
    mu  = float(np.exp(model['attack'][away_team]  + model['defense'][home_team]))
    rho = model['rho']

    g = np.arange(6)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= max(1e-6, 1 - lam * mu * rho)
    m[1, 0] *= max(1e-6, 1 + mu  * rho)
    m[0, 1] *= max(1e-6, 1 + lam * rho)
    m[1, 1] *= max(1e-6, 1 - rho)
    m /= m.sum()

    total = g[:, None] + g[None, :]
    flat  = {f'{i}-{j}': round(float(m[i, j]) * 100, 1) for i in range(6) for j in range(6)}
    top3  = sorted(flat.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        'home_win':    round(float(np.tril(m, -1).sum()) * 100, 1),
        'draw':        round(float(np.trace(m))           * 100, 1),
        'away_win':    round(float(np.triu(m,  1).sum()) * 100, 1),
        'xg_home':     round(lam, 2),
        'xg_away':     round(mu,  2),
        'btts':        round(float(m[1:, 1:].sum())       * 100, 1),
        'over_1_5':    round(float(m[total > 1].sum())    * 100, 1),
        'over_2_5':    round(float(m[total > 2].sum())    * 100, 1),
        'over_3_5':    round(float(m[total > 3].sum())    * 100, 1),
        'most_likely': top3[0][0],
        'top3':        top3,
        'matrix':      m.tolist(),
    }

# ── Team DNA ───────────────────────────────────────────────────────────────────

CLEANED_FILES = {l['slug']: l['file'] for l in REGISTRY}

def _team_stats(cleaned_path):
    """Per-team stats from cleaned CSV (all venues combined, last 2 seasons)."""
    try:
        df = pd.read_csv(cleaned_path, index_col=0, low_memory=False)
    except Exception:
        return {}
    df['gf']   = pd.to_numeric(df['gf'],   errors='coerce')
    df['ga']   = pd.to_numeric(df['ga'],   errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    done = df[
        df['gf'].notna() & df['ga'].notna() &
        (df['gf'] % 1 == 0) & (df['ga'] % 1 == 0) &
        df['date'].notna()
    ].copy()
    if 'year' in done.columns:
        done['year'] = pd.to_numeric(done['year'], errors='coerce')
        mx = done['year'].max()
        done = done[done['year'] >= mx - 1]
    stats = {}
    for team, grp in done.groupby('team'):
        n = len(grp)
        if n < 3:
            continue
        w    = (grp['result'].str.upper() == 'W').sum()
        # Last 5 matches sorted oldest→newest (so left=oldest, right=most recent)
        form = (grp.sort_values('date', ascending=True)
                   ['result'].str.upper().tail(5).tolist())
        stats[team] = {
            'games':   n,
            'win_pct': round(float(w / n * 100), 1),
            'gf_avg':  round(float(grp['gf'].mean()), 2),
            'ga_avg':  round(float(grp['ga'].mean()), 2),
            'form':    form,
        }
    return stats

def _actual_table(cleaned_path):
    """Build current-season standings (last 12 months of completed results)."""
    try:
        df = pd.read_csv(cleaned_path, index_col=0, low_memory=False)
    except Exception:
        return {}
    df['gf']   = pd.to_numeric(df['gf'], errors='coerce')
    df['ga']   = pd.to_numeric(df['ga'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    done = df[
        df['gf'].notna() & df['ga'].notna() &
        (df['gf'] % 1 == 0) & (df['ga'] % 1 == 0) &
        df['date'].notna()
    ].copy()
    cutoff = done['date'].max() - pd.DateOffset(months=12)
    done   = done[done['date'] >= cutoff]
    table  = {}
    for team, grp in done.groupby('team'):
        n = len(grp)
        if n < 3:
            continue
        w  = int((grp['result'].str.upper() == 'W').sum())
        d  = int((grp['result'].str.upper() == 'D').sum())
        l  = int((grp['result'].str.upper() == 'L').sum())
        gf = int(grp['gf'].sum())
        ga = int(grp['ga'].sum())
        table[team] = {
            'played': n, 'wins': w, 'draws': d, 'losses': l,
            'gf': gf, 'ga': ga, 'gd': gf - ga, 'pts': w * 3 + d,
        }
    return table

@app.route('/rankings/<slug>')
def rankings(slug):
    row = next(((s, f, t) for s, f, t in LEAGUES if s == slug), None)
    if not row:
        return 'League not found', 404
    _, filename, title = row

    model = _load_model(filename)
    if not model:
        return render_template('rankings.html', title=title, slug=slug,
                               rows=[], no_model=True)

    cf     = CLEANED_FILES.get(slug)
    actual = {}
    if cf:
        cp = _cleaned_dir() / cf
        if cp.exists():
            actual = _actual_table(str(cp))

    # Only compare teams present in both model AND current actual table
    common = [t for t in model['teams'] if t in actual]
    if not common:
        # Fall back to all model teams if no actual data
        common = model['teams']

    strength = {t: model['attack'][t] - model['defense'][t] for t in common}
    s_vals   = list(strength.values())
    s_min, s_max = min(s_vals), max(s_vals)
    s_rng = (s_max - s_min) or 1.0

    # Model ranking — re-ranked among current-season teams only
    model_ranked = sorted(common, key=lambda t: strength[t], reverse=True)
    model_rank   = {t: i + 1 for i, t in enumerate(model_ranked)}

    # Actual standings
    actual_ranked = sorted(
        common,
        key=lambda t: (actual.get(t, {}).get('pts', 0),
                       actual.get(t, {}).get('gd',  0),
                       actual.get(t, {}).get('gf',  0)),
        reverse=True,
    )
    actual_rank = {t: i + 1 for i, t in enumerate(actual_ranked)}

    rows = []
    for t in model_ranked:
        mr  = model_rank[t]
        ar  = actual_rank.get(t)
        a   = actual.get(t, {})
        # positive delta → outperforming model prediction
        # negative delta → underperforming model prediction
        delta    = (mr - ar) if ar is not None else None
        str_pct  = round((strength[t] - s_min) / s_rng * 100, 1)
        rows.append({
            'name':        t,
            'name_url':    _urlquote(t, safe=''),
            'model_rank':  mr,
            'str_pct':     str_pct,
            'actual_rank': ar,
            'pts':         a.get('pts'),
            'played':      a.get('played'),
            'wins':        a.get('wins'),
            'draws':       a.get('draws'),
            'losses':      a.get('losses'),
            'gd':          a.get('gd'),
            'delta':       delta,
        })

    with_delta  = [r for r in rows if r['delta'] is not None]
    biggest_up  = max(with_delta, key=lambda r: r['delta'],  default=None)
    biggest_dn  = min(with_delta, key=lambda r: r['delta'],  default=None)
    if biggest_up and biggest_up['delta'] < 2:  biggest_up = None
    if biggest_dn and biggest_dn['delta'] > -2: biggest_dn = None

    return render_template('rankings.html', title=title, slug=slug,
                           rows=rows, no_model=False,
                           biggest_up=biggest_up, biggest_dn=biggest_dn)

@app.route('/teams/<slug>')
def teams_overview(slug):
    row = next(((s, f, t) for s, f, t in LEAGUES if s == slug), None)
    if not row:
        return 'League not found', 404
    _, filename, title = row

    model = _load_model(filename)
    if not model:
        return render_template('teams.html', title=title, slug=slug,
                               teams=[], no_model=True)

    actual = {}
    cf = CLEANED_FILES.get(slug)
    if cf:
        cp = _cleaned_dir() / cf
        if cp.exists():
            actual = _team_stats(str(cp))

    tms   = model['teams']
    atks  = [model['attack'][t]  for t in tms]
    defs  = [model['defense'][t] for t in tms]
    a_min, a_max = min(atks), max(atks)
    d_min, d_max = min(defs), max(defs)
    a_rng = (a_max - a_min) or 1.0
    d_rng = (d_max - d_min) or 1.0

    rows = []
    for t in tms:
        atk_pct = round((model['attack'][t]  - a_min) / a_rng * 100, 1)
        def_pct = round((d_max - model['defense'][t]) / d_rng * 100, 1)
        s = actual.get(t, {})
        rows.append({
            'name':     t,
            'name_url': _urlquote(t, safe=''),
            'atk_pct':  atk_pct,
            'def_pct':  def_pct,
            'games':    s.get('games'),
            'win_pct':  s.get('win_pct'),
            'gf_avg':   s.get('gf_avg'),
            'ga_avg':   s.get('ga_avg'),
            'form':     s.get('form', []),
        })

    rows.sort(key=lambda x: x['atk_pct'], reverse=True)
    return render_template('teams.html', title=title, slug=slug,
                           teams=rows, no_model=False)

# ── Simulator ──────────────────────────────────────────────────────────────────

def _compute_league_accuracy(slug, filename):
    model = _load_model(filename)
    if not model:
        return None
    cf = CLEANED_FILES.get(slug)
    if not cf:
        return None
    cp = _cleaned_dir() / cf
    if not cp.exists():
        return None
    try:
        df = pd.read_csv(cp, index_col=0, low_memory=False)
    except Exception:
        return None

    df['gf']   = pd.to_numeric(df['gf'], errors='coerce')
    df['ga']   = pd.to_numeric(df['ga'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    done = df[
        df['gf'].notna() & df['ga'].notna() &
        (df['gf'] % 1 == 0) & (df['ga'] % 1 == 0) &
        df['date'].notna() &
        (df['venue'] == 'Home')
    ].copy()
    if done.empty:
        return None

    cutoff = done['date'].max() - pd.DateOffset(months=12)
    done   = done[done['date'] >= cutoff]
    if done.empty:
        return None

    total   = 0
    correct = 0
    by_actual = {'H': [0, 0], 'D': [0, 0], 'A': [0, 0]}  # [correct, total]

    for _, row in done.iterrows():
        home = str(row['team'])
        away = str(row.get('opponent', ''))
        if not away or home not in model['attack'] or away not in model['attack']:
            continue
        sim = _simulate(model, home, away)
        if not sim:
            continue

        probs     = {'H': sim['home_win'], 'D': sim['draw'], 'A': sim['away_win']}
        predicted = max(probs, key=probs.get)

        r      = str(row['result']).upper()
        actual = 'H' if r == 'W' else ('D' if r == 'D' else 'A')

        total += 1
        by_actual[actual][1] += 1
        if predicted == actual:
            correct += 1
            by_actual[actual][0] += 1

    if total == 0:
        return None

    return {
        'total':    total,
        'correct':  correct,
        'accuracy': round(correct / total * 100, 1),
        'by_actual': {
            k: {
                'correct': v[0], 'total': v[1],
                'pct': round(v[0] / v[1] * 100, 1) if v[1] > 0 else 0,
            }
            for k, v in by_actual.items()
        },
    }

@app.route('/accuracy')
def accuracy():
    results        = []
    total_matches  = 0
    total_correct  = 0
    agg = {'H': [0, 0], 'D': [0, 0], 'A': [0, 0]}

    for slug, filename, title in LEAGUES:
        data = _compute_league_accuracy(slug, filename)
        if not data:
            continue
        results.append({'title': title, 'slug': slug, **data})
        total_matches += data['total']
        total_correct += data['correct']
        for k in ('H', 'D', 'A'):
            agg[k][0] += data['by_actual'][k]['correct']
            agg[k][1] += data['by_actual'][k]['total']

    overall    = round(total_correct / total_matches * 100, 1) if total_matches else 0
    agg_by_actual = {
        k: {
            'correct': v[0], 'total': v[1],
            'pct': round(v[0] / v[1] * 100, 1) if v[1] else 0,
        }
        for k, v in agg.items()
    }
    return render_template('accuracy.html',
                           results=results,
                           total_matches=total_matches,
                           total_correct=total_correct,
                           overall=overall,
                           agg_by_actual=agg_by_actual)

@app.route('/simulator')
def simulator():
    leagues_data = []
    for slug, filename, title in LEAGUES:
        model = _load_model(filename)
        if model:
            leagues_data.append({'slug': slug, 'title': title, 'teams': model['teams']})
    return render_template('simulator.html', leagues=leagues_data)

@app.route('/api/simulate')
def api_simulate():
    slug      = request.args.get('league', '')
    home_team = request.args.get('home', '')
    away_team = request.args.get('away', '')

    filename = next((f for s, f, _ in LEAGUES if s == slug), None)
    if not filename:
        return jsonify({'error': 'League not found'}), 404

    model = _load_model(filename)
    if not model:
        return jsonify({'error': 'Model not available — run ScorelineModel.py first'}), 404

    if home_team == away_team:
        return jsonify({'error': 'Home and away teams must be different'}), 400

    result = _simulate(model, home_team, away_team)
    if not result:
        return jsonify({'error': 'One or both teams not found in model'}), 400

    return jsonify(result)


@app.route('/picks')
def best_picks():
    today = pd.Timestamp.today().normalize()
    all_picks = []

    for slug, filename, title in LEAGUES:
        path = _predictions_dir() / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            continue

        upcoming = df[df['Date'] >= today]
        for _, row in upcoming.iterrows():
            h = float(row.get('Home Win %') or 0)
            d = float(row.get('Draw %')     or 0)
            a = float(row.get('Away Win %') or 0)
            best = max(h, d, a)
            if best < 65:
                continue

            if best == h:
                outcome, outcome_code = 'Home Win', 'H'
            elif best == d:
                outcome, outcome_code = 'Draw',     'D'
            else:
                outcome, outcome_code = 'Away Win', 'A'

            dt = row['Date']
            if pd.notna(dt):
                if dt.date() == today.date():
                    date_label = 'Today'
                elif dt.date() == (today + pd.Timedelta(days=1)).date():
                    date_label = 'Tomorrow'
                else:
                    date_label = dt.strftime('%a %d %b')
            else:
                date_label = '—'

            all_picks.append({
                'date':         dt,
                'date_label':   date_label,
                'home':         str(row.get('Home Team', '')),
                'away':         str(row.get('Away Team', '')),
                'league':       title,
                'slug':         slug,
                'outcome':      outcome,
                'outcome_code': outcome_code,
                'confidence':   round(best, 1),
                'xg_home':      row.get('xG Home', '—'),
                'xg_away':      row.get('xG Away', '—'),
                'score':        row.get('Most Likely Score', '—'),
                'over_2_5':     row.get('Over 2.5 %', '—'),
                'btts':         row.get('BTTS %', '—'),
            })

    all_picks.sort(key=lambda x: (x['date'] if pd.notna(x['date']) else pd.Timestamp.max,
                                  -x['confidence']))

    # Group by date label preserving order
    grouped = {}
    for p in all_picks:
        grouped.setdefault(p['date_label'], []).append(p)

    return render_template('picks.html', grouped=grouped, total=len(all_picks))


# ── Stats ──────────────────────────────────────────────────────────────────────

@app.route('/stats')
@_require_auth
def stats():
    now = datetime.now(timezone.utc)
    cutoff_14 = (now - timedelta(days=13)).strftime('%Y-%m-%d 00:00:00')
    today_str  = now.strftime('%Y-%m-%d 00:00:00')

    with sqlite3.connect(_db_path()) as conn:
        total      = conn.execute('SELECT COUNT(*) FROM visits').fetchone()[0]
        unique_ips = conn.execute('SELECT COUNT(DISTINCT ip) FROM visits').fetchone()[0]
        today      = conn.execute(
            'SELECT COUNT(*) FROM visits WHERE ts >= ?', (today_str,)
        ).fetchone()[0]
        today_ips  = conn.execute(
            'SELECT COUNT(DISTINCT ip) FROM visits WHERE ts >= ?', (today_str,)
        ).fetchone()[0]
        daily = [
            {'day': r[0], 'cnt': r[1], 'uips': r[2]}
            for r in conn.execute('''
                SELECT substr(ts, 1, 10) AS day, COUNT(*) AS cnt, COUNT(DISTINCT ip) AS uips
                FROM visits WHERE ts >= ?
                GROUP BY day ORDER BY day
            ''', (cutoff_14,)).fetchall()
        ]
        pages = [
            {'path': r[0], 'cnt': r[1], 'uips': r[2]}
            for r in conn.execute('''
                SELECT path, COUNT(*) AS cnt, COUNT(DISTINCT ip) AS uips
                FROM visits
                GROUP BY path ORDER BY cnt DESC LIMIT 10
            ''').fetchall()
        ]
        ip_breakdown = [
            {'ip': r[0], 'visits': r[1], 'pages': r[2], 'first': r[3][:16], 'last': r[4][:16]}
            for r in conn.execute('''
                SELECT ip,
                       COUNT(*)           AS visits,
                       COUNT(DISTINCT path) AS pages,
                       MIN(ts)            AS first_seen,
                       MAX(ts)            AS last_seen
                FROM visits
                WHERE ip IS NOT NULL
                GROUP BY ip
                ORDER BY visits DESC
            ''').fetchall()
        ]

    max_day = max((row['cnt'] for row in daily), default=1)
    return render_template('stats.html',
                           total=total, unique_ips=unique_ips,
                           today=today, today_ips=today_ips,
                           daily=daily, max_day=max_day,
                           pages=pages, ip_breakdown=ip_breakdown)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug)
