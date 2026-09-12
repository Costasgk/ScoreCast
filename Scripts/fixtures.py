"""
Real dates for the rest of the season, from ESPN via soccerdata.

football-data publishes only the next matchweek, so ScoreCast derived the rest
of the season from the round-robin pairings — correct about who plays whom, but
with no date to show. ESPN publishes the full calendar, which turns those
pairings into actual fixtures.

Two problems have to be solved to use it. ESPN fingerprints plain HTTP clients,
so the request has to go through soccerdata's TLS client rather than requests.
And ESPN names clubs differently from football-data ("VfL Bochum" against
"Bochum", "SpVgg Greuther Fürth" against "Greuther Furth"), so names are matched
by token overlap with a stem fallback rather than by string equality.

Anything that cannot be matched is left alone: a fixture keeps its derived form
with no date rather than being attached to the wrong club.

Usage:
  python fixtures.py                 # refresh the date cache for every mapped league
  python fixtures.py --league GER-Bundesliga 2
"""

import argparse
import difflib
import json
import re
import sys
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parent.parent
CLEANED  = ROOT / 'Datasets' / 'Cleaned Datasets'
OUT_PATH = ROOT / 'Datasets' / 'Models' / 'fixture_dates.json'

# ScoreCast league key -> ESPN league name as configured in soccerdata.
# Only leagues ESPN actually publishes a full calendar for; the rest keep their
# derived, dateless pairings.
ESPN_LEAGUES = {
    'england_premier_league': 'ENG-Premier League',
    'england_championship':   'ENG-Championship',
    'england_league_one':     'ENG-League One',
    'england_league_two':     'ENG-League Two',
    'germany_bundesliga':     'GER-Bundesliga',
    'germany_2_bundesliga':   'GER-Bundesliga 2',
    'italy_serie_a':          'ITA-Serie A',
    'italy_serie_b':          'ITA-Serie B',
    'spain_la_liga':          'ESP-La Liga',
    'spain_segunda':          'ESP-Segunda',
    'france_ligue_1':         'FRA-Ligue 1',
    'france_ligue_2':         'FRA-Ligue 2',
    'netherlands_eredivisie': 'NED-Eredivisie',
    'portugal_primeira':      'POR-Primeira Liga',
    'belgium_pro_league':     'BEL-Pro League',
    'turkey_super_lig':       'TUR-Super Lig',
    'greece_super_league':    'GRE-Super League',
    'scotland_premiership':   'SCO-Premiership',
    'brazil_serie_a':         'BRA-Serie A',
    'usa_mls':                'USA-MLS',
    'japan_j1':               'JPN-J1 League',
    'norway_eliteserien':     'NOR-Eliteserien',
    'sweden_allsvenskan':     'SWE-Allsvenskan',
    'denmark_superliga':      'DEN-Superliga',
    'austria_bundesliga':     'AUT-Bundesliga',
    'switzerland_super':      'SUI-Super League',
    'poland_ekstraklasa':     'POL-Ekstraklasa',
}

# Leagues that run on a calendar year rather than autumn-to-spring. Asking ESPN
# for '2026-2027' returns the wrong window for these — mostly finished matches —
# which is why Norway showed eight fixtures when a hundred remained.
# soccerdata derives ESPN's scrape window from a league's season_start and
# season_end. For calendar-year competitions that arithmetic lands years off —
# Norway came back with the 2020 season whatever season string was passed — so
# these go to FBref, which resolves them correctly.
ESPN_WRONG_SEASON = {
    'norway_eliteserien', 'sweden_allsvenskan', 'usa_mls',
    'brazil_serie_a', 'japan_j1',
    # ESPN returned a partial calendar for Switzerland (35 of ~180) and keeps
    # failing on Poland, so both take the FBref route instead.
    'switzerland_super', 'poland_ekstraklasa', 'denmark_superliga',
}


# Leagues ESPN does not publish. FBref covers them, but only through a real
# browser (Cloudflare), so it is a fallback rather than the default: one league
# takes as long as ESPN takes for ten.
FBREF_LEAGUES = {
    'norway_eliteserien':    ('NOR-Eliteserien', '2026'),
    'sweden_allsvenskan':    ('SWE-Allsvenskan', '2026'),
    'usa_mls':               ('USA-MLS', '2026'),
    'brazil_serie_a':        ('BRA-Serie A', '2026'),
    'japan_j1':              ('JPN-J1 League', '2026'),
    'finland_veikkausliiga': ('FIN-Veikkausliiga', '2026'),
    'denmark_superliga':     ('DEN-Superliga', '2026-2027'),
    'argentina_liga':        ('ARG-Liga Profesional', '2026'),
    'ireland_premier':       ('IRL-Premier Division', '2026'),
    'mexico_liga_mx':        ('MEX-Liga MX', '2026-2027'),
    'china_super_league':    ('CHN-Super League', '2026'),
    'romania_superliga':     ('ROU-Liga I', '2026-2027'),
    'russia_premier':        ('RUS-Premier League', '2026-2027'),
    'scotland_championship': ('SCO-Championship2', '2026-2027'),
    # ESPN published only 35 Swiss dates and keeps 502-ing on Poland
    'switzerland_super':     ('SUI-Super League', '2026-2027'),
    'poland_ekstraklasa':    ('POL-Ekstraklasa', '2026-2027'),
}


# Club-type abbreviations that carry no identity and differ between sources
STOP = {
    'FC', 'SC', 'SV', 'VFL', 'VFB', 'TSV', 'SPVGG', 'BSC', 'CF', 'AC', 'AS',
    'SSC', 'US', 'CD', 'RC', 'FK', 'BK', 'IF', 'IK', 'GIF', 'AIK', 'SK', 'NK',
    'HNK', 'KV', 'KAA', 'RSC', 'SD', 'UD', 'SL', 'CS', 'CA', 'CLUB', 'DE',
}


# Letters NFKD will not decompose, because they are distinct letters rather
# than a base plus a diacritic. Without this, 'Brøndby' normalises to 'BRNDBY'
# and matches nothing — the o is dropped rather than kept.
_TRANSLIT = str.maketrans({
    'ø': 'o', 'Ø': 'O', 'æ': 'ae', 'Æ': 'AE', 'å': 'a', 'Å': 'A',
    'đ': 'd', 'Đ': 'D', 'ð': 'd', 'Ð': 'D', 'ł': 'l', 'Ł': 'L',
    'ß': 'ss', 'þ': 'th', 'Þ': 'TH', 'ı': 'i', 'ẞ': 'SS',
})


def _norm(s):
    s = str(s).translate(_TRANSLIT)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().upper()
    return re.sub(r'\s+', ' ', re.sub(r'[^A-Z0-9 ]', ' ', s)).strip()


def _initials(s):
    """'Queens Park Rangers' -> 'QPR', so an acronym can match the full name."""
    parts = [w for w in _norm(s).split() if w not in STOP]
    return ''.join(w[0] for w in parts) if len(parts) > 1 else ''


def _tokens(s):
    return {t for t in _norm(s).split()
            if t not in STOP and not t.isdigit() and len(t) > 2}


# Clubs whose two sources disagree on more than spelling — a different language
# or a different part of the name entirely. Token tricks cannot bridge these.
ALIASES = {
    'FC COLOGNE': 'KOLN', 'COLOGNE': 'KOLN',
    'STADE RENNAIS': 'RENNES',
    'ATHLETIC CLUB': 'BILBAO',
    'SPORTING CP': 'SP LISBON',
    'DEPORTIVO': 'LA CORUNA',
    'ESPANYOL': 'ESPANOL',
    'INTERNAZIONALE': 'INTER',
    'ATLETICO MADRID': 'ATH MADRID',
    'ATLETICO BILBAO': 'ATH BILBAO',
    'F C KOBENHAVN': 'COPENHAGEN', 'KOBENHAVN': 'COPENHAGEN',
    'ISTANBUL BASAKSEHIR': 'BUYUKSEHYR',
    'BRISTOL ROVERS': 'BRISTOL RVS',
    'LAFC': 'LOS ANGELES',
}


def _related(a, b):
    """
    Whether two name tokens refer to the same club.

    'Man' and 'Manchester', 'Inter' and 'Internazionale', 'M'gladbach' and
    'Mönchengladbach' are all the same club written shorter or longer, which
    containment catches. 'Rennes' and 'Rennais' share only a stem, so a common
    prefix is allowed too — four characters, which is long enough that
    unrelated clubs in the same division do not collide.
    """
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        return True
    # Three characters is enough for a club-name stem: 'Weds' and 'Wednesday',
    # 'Rennais' and 'Rennes'. Shorter than that and unrelated clubs collide.
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i >= 3 and min(len(a), len(b)) >= 3


def resolve(name, pool):
    """
    Match an ESPN club name to the name football-data uses, or None.

    Exact first, then whole-token overlap, then a stem comparison — 'Karlsruher
    SC' and 'Karlsruhe' share no whole token but are obviously the same club.
    Returning None is a valid answer: a wrong match would silently move a
    fixture to another team.
    """
    n = _norm(name)
    for p in pool:
        if _norm(p) == n:
            return p

    alias = ALIASES.get(n)
    if alias:
        for p in pool:
            if alias in _norm(p):
                return p

    # Acronyms: 'QPR' against 'Queens Park Rangers', 'AGF' against the full club
    # name. Checked both directions since either source may be the short one.
    ini = _initials(name)
    for p in pool:
        pn = _norm(p)
        if (ini and ini == pn.replace(' ', '')) or (_initials(p) == n.replace(' ', '')):
            return p

    want = _tokens(name)
    if not want:
        return None

    # Score on related tokens rather than identical ones, so a shortened or
    # lengthened club name still matches its own entry.
    # An exact token match is far stronger evidence than a fuzzy one, and
    # scoring them equally is what made 'Real Sociedad' tie with 'Villarreal' —
    # REAL is a substring of VILLARREAL. Exact counts 1.0, fuzzy 0.6, so a real
    # word match always outranks an accidental overlap.
    def token_score(w, tp):
        if w in tp:
            return 1.0
        return 0.6 if any(_related(w, t) for t in tp) else 0.0

    scored = []
    for p in pool:
        tp = _tokens(p)
        if not tp:
            continue
        total = sum(token_score(w, tp) for w in want)
        scored.append((total / min(len(want), len(tp)), p))
    if not scored:
        return None

    scored.sort(reverse=True)
    top = scored[0][0]
    winners = [p for sc, p in scored if sc == top]
    runner = next((sc for sc, _ in scored if sc < top), 0.0)

    # What makes a match trustworthy is the margin, not the absolute score.
    # 'Olympiacos' scores only 0.6 against 'Olympiakos' — one fuzzy token — but
    # every other club in the division scores 0.0, so it is unambiguous. A 0.75
    # cutoff rejected that while accepting nothing safer, which is how
    # Mönchengladbach, Karlsruher SC and three Turkish clubs went missing.
    if top < 0.6:
        return None
    if len(winners) == 1 and top - runner >= 0.25:
        return winners[0]
    if len(winners) == 1 and top >= 0.99:
        return winners[0]

    # Tied on tokens. Rather than refuse outright — which lost Olympiacos,
    # Mönchengladbach and Karlsruher SC — fall back to raw string similarity,
    # which separates 'Olympiakos' from 'Olympiacos Volou' the way a reader
    # would. Only a tie *there* is true ambiguity.
    sims = sorted(((difflib.SequenceMatcher(None, n, _norm(w)).ratio(), w)
                   for w in winners), reverse=True)
    if len(sims) > 1 and sims[0][0] - sims[1][0] > 0.05:
        return sims[0][1]

    # Tied because the tokeniser discarded what separates them: 'Paris FC' and
    # 'Paris SG' both reduce to {PARIS}. The dropped part is often an
    # abbreviation of the query's remaining words — SG for Saint-Germain — so
    # compare those before giving up.
    rest = [w for w in _norm(name).split() if w not in want and w not in STOP]
    abbrev = ''.join(w[0] for w in _norm(name).split() if w not in STOP)
    for w in winners:
        shorts = [t for t in _norm(w).split() if len(t) <= 3 and t not in STOP]
        for sh in shorts:
            if sh == abbrev or any(sh == ''.join(x[0] for x in rest) for _ in [0]):
                return w
            # 'SG' against SAINT GERMAIN: initials of the unmatched words
            unmatched = [x for x in _norm(name).split() if x not in _tokens(w)]
            if sh == ''.join(x[0] for x in unmatched if x not in STOP):
                return w
    # Still ambiguous — refuse rather than attach a fixture to the wrong club.
    return None


def our_teams(league):
    path = CLEANED / league['file']
    if not path.exists():
        return []
    df = pd.read_csv(path, index_col=0, low_memory=False)
    if 'season' not in df.columns:
        return []
    latest = sorted(df['season'].dropna().astype(str).unique())[-1]
    cur = df[df['season'].astype(str) == latest]
    return sorted(set(cur['team'].astype(str)) | set(cur['opponent'].astype(str)))


def fbref_schedule(fbref_league, season):
    """FBref via a real browser — slow, but it reaches leagues ESPN skips."""
    import soccerdata as sd
    s = sd.FBref(leagues=fbref_league, seasons=season).read_schedule().reset_index()
    s['date'] = pd.to_datetime(s['date'], errors='coerce')
    if 'time' in s.columns:
        t = s['time'].astype(str).str.extract(r'(\d{1,2}:\d{2})')[0]
        s['date'] = pd.to_datetime(
            s['date'].dt.strftime('%Y-%m-%d') + ' ' + t.fillna('00:00'),
            errors='coerce').fillna(s['date'])
    return s.dropna(subset=['date'])


def espn_schedule(espn_league, season):
    import soccerdata as sd
    s = sd.ESPN(leagues=espn_league, seasons=season).read_schedule().reset_index()
    s['date'] = pd.to_datetime(s['date'], errors='coerce', utc=True)
    return s.dropna(subset=['date'])


def build(league, season='2026-2027', use_fbref=True):
    """{'home|away': 'YYYY-MM-DD HH:MM'} for one league, or None."""
    pool = our_teams(league)
    if not pool:
        return None

    espn_name = ESPN_LEAGUES.get(league['key'])
    fb = FBREF_LEAGUES.get(league['key'])
    sched = None

    if espn_name and league['key'] not in ESPN_WRONG_SEASON:
        try:
            sched = espn_schedule(espn_name, season)
        except Exception as e:
            print(f"  {league['key']}: ESPN failed — {type(e).__name__}: {str(e)[:60]}")

    if sched is None and fb and use_fbref:
        try:
            sched = fbref_schedule(fb[0], fb[1])
        except Exception as e:
            print(f"  {league['key']}: FBref failed — {type(e).__name__}: {str(e)[:60]}")

    if sched is None:
        if espn_name or fb:
            print(f"  {league['key']}: no schedule from any source")
        else:
            print(f"  {league['key']}: not configured for ESPN or FBref")
        return None

    names = sorted(set(sched['home_team']) | set(sched['away_team']))
    mapping = {n: resolve(n, pool) for n in names}
    unmatched = [n for n, v in mapping.items() if v is None]

    out = {}
    for _, r in sched.iterrows():
        h, a = mapping.get(r['home_team']), mapping.get(r['away_team'])
        if not h or not a:
            continue
        out[f"{h}|{a}"] = r['date'].strftime('%Y-%m-%d %H:%M')

    print(f"  {league['name']} ({league['region']}): {len(out)} dated fixtures, "
          f"{len(names) - len(unmatched)}/{len(names)} clubs matched"
          + (f"  unmatched: {', '.join(unmatched)}" if unmatched else ""))
    return out


def main():
    from leagues import LEAGUES
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', default='2026-2027')
    ap.add_argument('--league', nargs='+', default=None, help='ScoreCast league keys')
    ap.add_argument('--no-fbref', action='store_true',
                    help='skip the browser-driven fallback (much faster)')
    args = ap.parse_args()

    targets = [l for l in LEAGUES
               if (l['key'] in ESPN_LEAGUES or l['key'] in FBREF_LEAGUES)
               and (not args.league or l['key'] in args.league)]

    print("=" * 64)
    print(f"  Fixture dates from ESPN  |  {len(targets)} leagues")
    print("=" * 64)

    all_dates, ok = {}, 0
    for l in targets:
        d = build(l, args.season, use_fbref=not args.no_fbref)
        if d:
            all_dates[l['key']] = d
            ok += 1

    if all_dates:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Merge rather than replace. Running this for a single league used to
        # write a file containing only that league, silently discarding every
        # other calendar already fetched.
        merged = {}
        if OUT_PATH.exists():
            try:
                merged = json.loads(OUT_PATH.read_text()).get('leagues', {})
            except Exception:
                merged = {}
        merged.update(all_dates)
        OUT_PATH.write_text(json.dumps({
            'generated': datetime.now().isoformat(timespec='seconds'),
            'season': args.season,
            'leagues': merged,
        }, indent=1))
        all_dates = merged
        total = sum(len(v) for v in all_dates.values())
        print(f"\n  {ok}/{len(targets)} leagues, {total} dated fixtures")
        print(f"  saved -> {OUT_PATH}")
    print()


if __name__ == '__main__':
    main()
