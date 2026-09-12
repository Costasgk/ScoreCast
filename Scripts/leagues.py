"""
The league registry — one definition, every consumer.

The league list used to be repeated in FootballData.py, pipeline.py, app.py and
simulate_season.py. Four copies was already awkward at nine leagues; at thirty-six
it would guarantee drift. Everything now derives from LEAGUES below.

Fields
  key        stable identifier
  name       display name
  region     country or area, shown beside the name
  slug       web route, e.g. /Premier-League-England
  source     'main'  -> /mmz4281/<season>/<div>.csv, carries shots and odds
             'extra' -> /new/<country>.csv, results only
  file       cleaned CSV name
  pred       predictions CSV name
  europe     places that qualify for continental football
  relegate   automatic relegation places
  tier       1 = top flight, 2 = second tier
  exact      True when europe/relegate are verified for that competition;
             False where they are a reasonable default. Play-offs are never
             modelled, so 'relegate' means the automatic places only.
"""


# Leagues that do NOT play a straight double round-robin. Measured, not assumed:
# for each league's last completed season, games-per-team was compared with
# 2*(n-1). Championship splits, Apertura/Clausura and conference schedules all
# fail that test, and the season simulator cannot derive their remaining
# fixtures from the pairings alone — so it declines to guess.
NOT_ROUND_ROBIN = {
    'belgium_pro_league', 'greece_super_league', 'scotland_premiership',
    'scotland_championship', 'argentina_liga', 'usa_mls', 'mexico_liga_mx',
    'denmark_superliga', 'finland_veikkausliiga', 'austria_bundesliga',
    'switzerland_super', 'romania_superliga', 'ireland_premier',
}


def _m(key, name, region, slug, div, file, europe, relegate, tier=1, exact=True):
    return {'key': key, 'name': name, 'region': region, 'slug': slug,
            'source': 'main', 'div': div, 'file': file,
            'pred': f'predictions_{key}.csv',
            'europe': europe, 'relegate': relegate, 'tier': tier, 'exact': exact,
            'round_robin': key not in NOT_ROUND_ROBIN}


def _x(key, name, region, slug, country, league, file, europe, relegate, tier=1, exact=False):
    return {'key': key, 'name': name, 'region': region, 'slug': slug,
            'source': 'extra', 'country': country, 'league': league, 'file': file,
            'pred': f'predictions_{key}.csv',
            'europe': europe, 'relegate': relegate, 'tier': tier, 'exact': exact,
            'round_robin': key not in NOT_ROUND_ROBIN}


LEAGUES = [
    # ── Top flights with shot data and odds ──────────────────────────────────
    _m('england_premier_league', 'Premier League',   'England',     'Premier-League-England',   'E0',  'England_Premier_League.csv',  4, 3),
    _m('italy_serie_a',          'Serie A',          'Italy',       'Serie-A-Italy',            'I1',  'Italy_Serie_A.csv',           4, 3),
    _m('spain_la_liga',          'La Liga',          'Spain',       'La-Liga-Spain',            'SP1', 'Spain_La_Liga.csv',           4, 3),
    _m('germany_bundesliga',     'Bundesliga',       'Germany',     'Bundesliga-Germany',       'D1',  'Germany_Bundesliga.csv',      4, 2),
    _m('france_ligue_1',         'Ligue 1',          'France',      'Ligue-1-France',           'F1',  'France_Ligue_1.csv',          3, 2),
    _m('netherlands_eredivisie', 'Eredivisie',       'Netherlands', 'Eredivisie-Netherlands',   'N1',  'Netherlands_Eredivisie.csv',  3, 2),
    _m('portugal_primeira',      'Primeira Liga',    'Portugal',    'Primeira-Liga-Portugal',   'P1',  'Portugal_Primeira_Liga.csv',  3, 2),
    _m('belgium_pro_league',     'Pro League',       'Belgium',     'Pro-League-Belgium',       'B1',  'Belgium_Pro_League.csv',      3, 1, 1, False),
    _m('turkey_super_lig',       'Super Lig',        'Turkey',      'Super-Lig-Turkey',         'T1',  'Turkey_Super_Lig.csv',        3, 3, 1, False),
    _m('greece_super_league',    'Super League',     'Greece',      'Super-League-Greece',      'G1',  'Greece_Super_League.csv',     3, 2, 1, False),
    _m('scotland_premiership',   'Premiership',      'Scotland',    'Premiership-Scotland',     'SC0', 'Scotland_Premiership.csv',    3, 1),

    # ── Second tiers: no continental places, promotion is the prize ──────────
    _m('england_championship',   'Championship',     'England',     'Championship-England',     'E1',  'England_Championship.csv',    2, 3, 2),
    _m('england_league_one',     'League One',       'England',     'League-One-England',       'E2',  'England_League_One.csv',      2, 4, 2),
    _m('england_league_two',     'League Two',       'England',     'League-Two-England',       'E3',  'England_League_Two.csv',      3, 2, 2),
    _m('england_national',       'National League',  'England',     'National-League-England',  'EC',  'England_National_League.csv', 1, 4, 2, False),
    _m('germany_2_bundesliga',   '2. Bundesliga',    'Germany',     '2-Bundesliga-Germany',     'D2',  'Germany_2_Bundesliga.csv',    2, 2, 2),
    _m('italy_serie_b',          'Serie B',          'Italy',       'Serie-B-Italy',            'I2',  'Italy_Serie_B.csv',           2, 3, 2, False),
    _m('spain_segunda',          'Segunda Division', 'Spain',       'Segunda-Spain',            'SP2', 'Spain_Segunda.csv',           2, 4, 2, False),
    _m('france_ligue_2',         'Ligue 2',          'France',      'Ligue-2-France',           'F2',  'France_Ligue_2.csv',          2, 2, 2, False),
    _m('scotland_championship',  'Championship',     'Scotland',    'Championship-Scotland',    'SC1', 'Scotland_Championship.csv',   1, 1, 2, False),

    # ── Results-only leagues (no shots, no odds) ─────────────────────────────
    _x('brazil_serie_a',         'Serie A',          'Brazil',      'Serie-A-Brazil',      'BRA', 'Serie A',          'Brazil_Serie_A.csv',        4, 4),
    _x('argentina_liga',         'Liga Profesional', 'Argentina',   'Liga-Argentina',      'ARG', 'Liga Profesional', 'Argentina_Liga.csv',        4, 2),
    _x('usa_mls',                'MLS',              'USA',         'MLS-USA',             'USA', 'MLS',              'USA_MLS.csv',               0, 0),
    _x('mexico_liga_mx',         'Liga MX',          'Mexico',      'Liga-MX-Mexico',      'MEX', 'Liga MX',          'Mexico_Liga_MX.csv',        0, 0),
    _x('japan_j1',               'J1 League',        'Japan',       'J1-League-Japan',     'JPN', 'J1 League',        'Japan_J1_League.csv',       3, 3),
    _x('china_super_league',     'Super League',     'China',       'Super-League-China',  'CHN', 'Super League',     'China_Super_League.csv',    2, 2),
    _x('norway_eliteserien',     'Eliteserien',      'Norway',      'Eliteserien-Norway',  'NOR', 'Eliteserien',      'Norway_Eliteserien.csv',    3, 2),
    _x('sweden_allsvenskan',     'Allsvenskan',      'Sweden',      'Allsvenskan-Sweden',  'SWE', 'Allsvenskan',      'Sweden_Allsvenskan.csv',    3, 2),
    _x('denmark_superliga',      'Superliga',        'Denmark',     'Superliga-Denmark',   'DNK', 'Superliga',        'Denmark_Superliga.csv',     3, 2),
    _x('finland_veikkausliiga',  'Veikkausliiga',    'Finland',     'Veikkausliiga-Finland','FIN','Veikkausliiga',    'Finland_Veikkausliiga.csv', 3, 2),
    _x('austria_bundesliga',     'Bundesliga',       'Austria',     'Bundesliga-Austria',  'AUT', 'Bundesliga',       'Austria_Bundesliga.csv',    3, 1),
    _x('switzerland_super',      'Super League',     'Switzerland', 'Super-League-Swiss',  'SWZ', 'Super League',     'Switzerland_Super_League.csv', 3, 1),
    _x('poland_ekstraklasa',     'Ekstraklasa',      'Poland',      'Ekstraklasa-Poland',  'POL', 'Ekstraklasa',      'Poland_Ekstraklasa.csv',    3, 2),
    _x('romania_superliga',      'Superliga',        'Romania',     'Superliga-Romania',      'ROU', 'Superliga',        'Romania_Superliga.csv',        3, 2),
    _x('russia_premier',         'Premier League',   'Russia',      'Premier-League-Russia','RUS','Premier League',   'Russia_Premier_League.csv', 3, 2),
    _x('ireland_premier',        'Premier Division', 'Ireland',     'Premier-Division-Ireland','IRL','Premier Division','Ireland_Premier_Division.csv', 2, 1),
]

BY_KEY  = {l['key']: l for l in LEAGUES}
BY_SLUG = {l['slug']: l for l in LEAGUES}
BY_FILE = {l['file']: l for l in LEAGUES}


def full_name(l):
    """'Premier League (England)' — region disambiguates the many Serie As."""
    return f"{l['name']} ({l['region']})"


def by_region():
    """Leagues grouped for display, top flights before second tiers."""
    out = {}
    for l in sorted(LEAGUES, key=lambda x: (x['tier'], x['region'], x['name'])):
        out.setdefault(l['region'], []).append(l)
    return out
