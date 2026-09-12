# ScoreCast — Session Notes

## Goal
Extend ScoreCast from a simple W/D/L classifier into a full scoreline distribution predictor using the Dixon-Coles Poisson model.

---

## Files Changed

### `Scripts/Scrapping.py` — full rewrite + optimisations
- Replaced `requests` with `undetected_chromedriver` (real Chrome browser) to bypass Cloudflare IUAM which now blocks all plain HTTP requests to FBref
- Chrome version pinned: `version_main=148`
- Headless mode removed — Chrome runs visibly so Cloudflare doesn't detect a bot
- Added `_make_driver()`, `_get_html()`, `_needs_human()`, `_try_dismiss_overlay()` helpers
- Added **persistent Chrome profile** (`~/.scorecast_chrome_profile`) — after first Cloudflare verification, subsequent runs are fully automated
- Added **image/font blocking** (`prefs`) — pages load faster since we only need HTML
- Added `RESCRAPE_RECENT = 2` — always re-scrapes last 2 seasons even if cached
- Added **incremental saves** — CSV written after every season, crash-safe
- Added **skip cached seasons** — reads existing CSV at league start, skips years already scraped (except last 2)
- Reduced `time.sleep(3)` → `time.sleep(2)` per team
- Dynamic year ranges: top 5 leagues go back 16 years, others 12 years, Serie B 7 years
- Year range updated to start from `datetime.now().year` (2026)
- All `pd.read_html` calls wrapped in `StringIO` (required for pandas 3.x)
- Error messages truncated to 120 chars to prevent HTML spam in terminal

**Result:** Scraped 591 teams across 11 leagues in ~8.3 hours. All CSVs now cover up to 2026.

---

### `Scripts/Cleaning.py` — bug fix + robustness
- **Critical bug fixed (line 65):** `df['ga'] = pd.to_numeric(df['gf'], errors='coerce')` → `df['ga'] = pd.to_numeric(df['ga'], errors='coerce')`. This was copying goals-for into goals-against, making every match look like a draw and corrupting all cleaned CSVs.
- `read_file()`: added `index_col=0` — scraped CSVs are saved with a Pandas index column
- `read_file()`: added column lowercasing (`df.columns = [c.lower() for c in df.columns]`)
- `read_file()`: added FBref embedded header row filter — FBref tables include repeated `<tr class="thead">` rows inside tbody which `pd.read_html` picks up as data rows where `date = "Date"`
- `cleaning()`: added `pd.to_numeric(..., errors='coerce')` for all numeric columns before `.mean()` — columns are read as string dtype and need explicit conversion

---

### `Scripts/ScorelineModel.py` — new file
Full Dixon-Coles Poisson model. Reads cleaned CSVs, writes prediction CSVs for all upcoming fixtures.

**Model: `DixonColesModel`**
- Fits attack strength + defense strength per team via maximum likelihood (L-BFGS-B)
- Exponential time-weighting: `DECAY_RATE = 0.0065` (half-life ~107 days) — recent matches weighted far more than old ones
- 5-year training cutoff — older data has near-zero weight and inflates parameter space
- Dixon-Coles low-score correction (`rho`) — adjusts 0-0, 1-0, 0-1, 1-1 probabilities
- Bounded optimisation: attack/defense (-4, 4), home_adv (-1.5, 1.5), rho (-0.5, 0.4)

**Data helpers**
- `load_matches()`: league-team filter — teams with fewer than 10 home appearances excluded (removes European/cup opponents)
- `upcoming_fixtures()`: 12-month window from dataset's max date — prevents old null-goal rows being treated as upcoming

**Output per fixture (48 columns)**
- Date, Home Team, Away Team
- Home Win %, Draw %, Away Win %
- xG Home, xG Away
- BTTS % (both teams to score)
- Over 1.5 %, Over 2.5 %, Over 3.5 %
- Most Likely Score (e.g. "1-0")
- Top 3 Scores (e.g. "1-0 (12.7%), 2-0 (12.5%), 1-1 (10.1%)")
- p_0_0 through p_5_5 — 36 scoreline probability columns

**Decision: actual goals (not xG)**
xG data is 0% available for 2024, 2025, 2026 — the seasons that carry almost all model weight due to time-decay. xG is available for 2019-2022 but those seasons contribute <1% to the model. Hybrid approach unnecessary.

---

## Key Concepts Explained

**Dixon-Coles model:** Each team has attack/defense parameters. Goals modelled as Poisson-distributed with rate λ (home xG) = exp(attack_home + defense_away + home_adv). Scoreline grid computed as outer product of two Poisson PMFs, corrected at low scores.

**Exponential decay:** Matches weighted by `exp(-0.0065 × days_ago)`. A match from 1 year ago has 8% weight; from 3 years ago has 0.07% weight. Needed because team quality changes every season — old results are irrelevant.

**Time-weighting effect on training:**
- 2026 (current season): ~90% weight
- 2025: ~8%
- 2024: ~0.8%
- 2023 and older: <0.1%

---

## Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| actual goals vs xG | actual goals | xG missing for 2024-2026 |
| hybrid approach | no | would be actual goals for 98% anyway |
| training window | 5 years | older data has near-zero weight |
| league filter | min 10 home games | removes cup opponents |
| scoreline grid | 0-0 to 5-5 (36 cells) | covers all realistic scorelines |

---

## Pending / Next Steps

1. **Run `Cleaning.py`** on all 11 leagues now that scraping is complete
2. **Run `ScorelineModel.py`** on all 11 leagues to generate prediction CSVs
3. **Wire into Flask app** — new routes and UI to display scoreline heatmaps, BTTS, Over/Under, Most Likely Score
4. **Optional future improvement:** add `comp` column through Cleaning.py and filter to league-only games in `load_matches()` — currently cup games between league teams (e.g. Corinthians vs Flamengo in Copa do Brasil) slip through the min_home_games filter

---

## Leagues Covered
| League | CSV |
|--------|-----|
| Brazil Serie A | Serie_A_Stats.csv |
| Brazil Serie B | Serie_B_Stats.csv |
| Eliteserien (Norway) | Eliteserien_Stats.csv |
| Veikkausliiga (Finland) | Veikkausliiga_Stats.csv |
| Super League Greece | Super_League_Greece_Stats.csv |
| Premier League | Premier_League_Stats.csv |
| Serie A Italy | Serie_A_Stats_Italy.csv |
| La Liga | La_Liga_Stats.csv |
| Bundesliga | Bundesliga_Stats.csv |
| Ligue 1 | Ligue_1_Stats.csv |
| J1 League | J1_League_Stats.csv |

---

## Session — 11 Sep 2026: football-data.co.uk source + fixes

### New: `Scripts/FootballData.py`
FBref now returns **403** to plain HTTP (Cloudflare), so the Selenium scrape is
the only way in there — ~1.5h plus a manual verification tick.
`football-data.co.uk` publishes the same results as plain CSV over HTTP and
returns 200, so the whole dataset now lands in ~30 seconds.

Writes straight into `Cleaned Datasets` in the long one-row-per-team-per-match
shape, so the scrape **and** clean steps are skipped entirely.

| Covers | Not covered |
|---|---|
| 9 leagues, 5 seasons, + upcoming fixtures | Brazil Serie B, J1 League |
| goals, shots, shots-on-target, referee | pk/pkatt, possession, formation, xG |

Dixon-Coles uses only goals, so it is fully served. The RandomForest in
`Modelling.py` needs `pk`/`pkatt` — **`rf.pkl` still requires the FBref path**.

### `pipeline.py`
- `--source footballdata` (**default**, ~1 min) or `--source fbref` (~1.5h)
- New `step_fetch()`; staleness heuristics skipped on the fast path
- Step labels were numbered wrong once FETCH replaced SCRAPE+CLEAN — now named

### Fixes
- **Cup games excluded** — `comp` + `round` now survive `Cleaning.read_file`, and
  `ScorelineModel.load_matches` keeps only rounds starting `Matchweek`.
  Filtering on *round* not *comp name* matters: Norway's `Tippeligaen` was
  renamed `Eliteserien` in 2017, and a name-based filter silently deletes 960
  valid league rows. Dropped 1,152 Libertadores/Sudamericana rows from Brazil.
- **`Modelling.py` train cutoff** — was hardcoded `'2023-07-19'`, making any
  retrain a silent no-op. Now splits relative to the data's own max date.
- **Chrome pin** — `version_main=148` unpinned; local Chrome is 153.
- **Scraper re-run duplication** — re-scraped seasons are now *replaced*, not
  appended (`_merge_seasons`). Previously every run duplicated recent seasons.
- **Scrape window** `SEASONS = 5`; sleeps 2s -> 0.5s; `page_load_strategy=eager`.
  FBref path is ~1.5h now instead of 8.3h.
- **`requirements.txt`** — was still pre-Selenium. Added selenium,
  undetected-chromedriver, tqdm, scipy, setuptools (uc imports `distutils`,
  gone in py3.12).

### Verified
9/10 leagues predicted (Serie B correctly skipped), 65 upcoming fixtures,
all 15 Flask routes return 200, probabilities sum to 99.9-100.0.

---

# Session 2 — 36 leagues, market benchmark, containerisation

## Data pipeline

### `Scripts/FootballData.py` — new
Fetches all 36 leagues from football-data.co.uk over plain HTTP in ~30s, replacing
the 8.3h Cloudflare-blocked browser scrape as the primary source.
- Season labels sorted **as strings**, not `pd.to_numeric` — seven countries use
  `2012/2013` format and numeric coercion silently dropped them.
- `_season_code()` derives the European season from fixture dates so 2026-27
  fixtures are not filed under 2526.

### `Scripts/leagues.py` — new
Single registry for all 36 leagues. The league list had been duplicated across four
files and had already drifted. Carries `round_robin`, **measured** by comparing
games-per-team against 2(n-1) rather than assumed.

### `Scripts/fixtures.py` — new
Full-season calendars from ESPN and FBref via soccerdata; football-data publishes
only the coming week. Expanded fixtures from ~65 to ~8,000.
- `_TRANSLIT` for ø/æ/å/đ/ð/ł/ß — NFKD **deletes** these rather than transliterating
  (`Brøndby` became `BRNDBY`).
- Name resolver is margin-based: a winner must beat the runner-up by ≥0.25. Earlier
  ambiguity guards oscillated; root cause was `REAL` being a substring of
  `VILLARREAL`. Final form recovers 12/12 true matches while blocking 6/6 false ones.
- Cache writes **merge**. `--league X` previously wiped every other league.
- All five initial FBref league names were wrong guesses; the real 158 names were
  read from soccerdata's cached `leagues.html`.

## Model

- `DECAY_RATE` 0.0065 → **0.0020** (347-day half-life), tuned on held-out seasons via
  `tune_decay.py`. Mean gap to the closing line improved +0.0406 → +0.0308 with every
  league improving, which is what makes it a real effect rather than a fit to noise.
- Calibration by temperature scaling was implemented, **measured, and rejected** — it
  made held-out log loss worse by +0.0011. The guard refuses to save. The "85%→69%"
  miscalibration that motivated it came from a bin with n=26.

### `Scripts/benchmark.py` — new
Walk-forward scoring against de-vigged closing odds: log loss, Brier, calibration
bins, value return. Disk cache with a 6h TTL for the in-progress season.

### `Scripts/simulate_season.py` — new
10,000-run Monte Carlo of the remaining season, sampling scorelines rather than
outcomes so goal difference moves correctly.

## Bugs found and fixed

- **`Modelling.py` hardcoded `'2023-07-19'` cutoff** made retraining a silent no-op.
- **Cup-round filter** on competition name would have deleted 960 Norwegian rows
  (Tippeligaen → Eliteserien rename); now filters on `round` starting "Matchweek".
- **Round-robin assumption fabricated fixtures for 13 leagues**, including a Greek
  title race that had already been published. Now measured, not assumed.
- **Team-total rows** (`dorsal='TOTAL'`) outranked every player in basketball leaders;
  3P% leaders had a single attempt. Filtered, and attempt counts carried.
- **Auth failed OPEN** when no password was configured — now returns 503 unless
  `SCORECAST_OPEN_STATS=1` is set explicitly.
- **Missing fixtures, three independent causes** (all found chasing one report of
  "games missing from all leagues"):
  1. `app.py` split its two tables on the `Scheduled` flag, which records *provenance*
     (football-data's weekly feed) rather than whether a fixture has a date. Once
     ESPN/FBref supplied season-wide dates the two diverged, filing 34 dated fixtures
     across 21 leagues under "rest of season".
  2. `remaining_pairings` returned **only** the published calendar, with no fallback.
     Calendars run to the end of what has been announced, so **637 fixtures** appeared
     nowhere at all — not played, not predicted. 2. Bundesliga held 268 of its 306.
     `simulate_season.py` had the same gap independently: the Premier League title was
     being decided over 350 games instead of 380.
  3. **The visible one.** `.sc-table-wrap` sets `overflow-x: auto`, which forces
     `overflow-y` to compute to `auto` — making the wrapper the nearest scrolling
     ancestor. The sticky `thead th { top: 3.4rem }`, intended to park under the fixed
     topbar during page scroll, was therefore measured from the top of the *table* and
     held an opaque header over the first row. **Every table on every page silently
     lost its first row**: captions said 5 and readers counted 4. Fixed with `top: 0`.

## Deployment

`Dockerfile`, `compose.yml`, `.dockerignore`, `requirements-web.txt` — the image
serves but does not fetch, so it needs no browser. Data bind-mounted read-only at
`/data`, state at `/state`, runs as UID 10001, health check hits a real page.

### Verified
36 leagues predicted; 22 now fixture-complete (every ordered pair accounted for) where
previously none were; 0 header/row overlaps across 8 page types; captions match
rendered row counts on all 8 leagues spot-checked.
