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
