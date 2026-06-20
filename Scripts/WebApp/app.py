from flask import Flask, render_template, send_file, jsonify, request, Response
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

def _require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not _STATS_PASSWORD:
            return f(*args, **kwargs)  # no config set → allow (local dev only)
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
    return Path(__file__).resolve().parent / 'visits.db'

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
    path = _predictions_dir() / filename
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    return df, df[cols]

# ── League list ────────────────────────────────────────────────────────────────

LEAGUES = [
    ('Serie-A-Brazil',        'predictions_brazil_serie_a.csv',          'Serie A (Brazil)'),
    ('Serie-B-Brazil',        'predictions_brazil_serie_b.csv',          'Serie B (Brazil)'),
    ('Premier-League-England','predictions_england_premier_league.csv',  'Premier League (England)'),
    ('Serie-A-Italy',         'predictions_italy_serie_a.csv',           'Serie A (Italy)'),
    ('La-Liga-Spain',         'predictions_spain_la_liga.csv',           'La Liga (Spain)'),
    ('Ligue-1-France',        'predictions_france_ligue_1.csv',          'Ligue 1 (France)'),
    ('Bundesliga-Germany',    'predictions_germany_bundesliga.csv',       'Bundesliga (Germany)'),
    ('Super-League-Greece',   'predictions_greece_super_league.csv',     'Super League (Greece)'),
    ('Eliteserien-Norway',    'predictions_norway_eliteserien.csv',       'Eliteserien (Norway)'),
    ('Veikkausliiga-Finland', 'predictions_finland_veikkausliiga.csv',    'Veikkausliiga (Finland)'),
]

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

@app.route('/', methods=['GET'])
def display_home():
    today    = date.today().strftime('%B %d, %Y')
    pred_dir = _predictions_dir()
    status   = {slug: _is_active(pred_dir / filename) for slug, filename, _ in LEAGUES}
    return render_template('index.html', today=today, status=status)

# ── League routes ──────────────────────────────────────────────────────────────

def _make_route(slug, filename, title):
    def view():
        path = _predictions_dir() / filename
        if not path.exists():
            return render_template('league.html',
                                   title=title, league_slug=slug,
                                   has_predictions=False, table_html='')
        _, display_df = _load(filename)
        table_html = display_df.to_html(index=False, classes='', na_rep='—', border=0)
        return render_template('league.html',
                               title=title, league_slug=slug,
                               has_predictions=True, table_html=table_html)

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

CLEANED_FILES = {
    'Serie-A-Brazil':         'Brazil_Serie_A.csv',
    'Serie-B-Brazil':         'Brazil_Serie_B.csv',
    'Premier-League-England': 'England_Premier_League.csv',
    'Serie-A-Italy':          'Italy_Serie_A.csv',
    'La-Liga-Spain':          'Spain_La_Liga.csv',
    'Ligue-1-France':         'France_Ligue_1.csv',
    'Bundesliga-Germany':     'Germany_Bundesliga.csv',
    'Super-League-Greece':    'Greece_Super_League.csv',
    'Eliteserien-Norway':     'Norway_Eliteserien.csv',
    'Veikkausliiga-Finland':  'Finland_Veikkausliiga.csv',
}

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
