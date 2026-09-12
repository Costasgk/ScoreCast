import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO
import time

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm


RESCRAPE_RECENT = 1   # always re-scrape the last N seasons even if cached
PAGE_SETTLE     = 0.5 # pause after each page load, seconds
TEAM_DELAY      = 0.5 # pause between teams, seconds
SEASONS         = 5   # seasons back to scrape — older data is decayed to ~0 by the models

# ── League display names ───────────────────────────────────────────────────────

LEAGUE_NAMES = {
    "Serie-A-Stats":               "Brazil Serie A",
    "Eliteserien-Stats":           "Eliteserien (Norway)",
    "Veikkausliiga-Stats":         "Veikkausliiga (Finland)",
    "Super-League-Greece-Stats":   "Super League Greece",
    "Premier-League-Stats":        "Premier League (England)",
    "Serie-A-Stats-Italy":         "Serie A (Italy)",
    "La-Liga-Stats":               "La Liga (Spain)",
    "Bundesliga-Stats":            "Bundesliga (Germany)",
    "Ligue-1-Stats":               "Ligue 1 (France)",
}


# ── Browser driver ─────────────────────────────────────────────────────────────

def _make_driver():
    profile_dir = os.path.join(os.path.expanduser("~"), ".scorecast_chrome_profile")
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-data-dir={profile_dir}")
    # Block images, fonts, and media — we only need HTML/JS to pass Cloudflare
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.fonts": 2,
        "profile.managed_default_content_settings.media_stream": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.page_load_strategy = "eager"   # don't wait on subresources
    driver = uc.Chrome(options=options, use_subprocess=True)
    driver.set_page_load_timeout(60)
    return driver


def _needs_human(driver):
    """Return a reason string if the page requires manual action, else empty string."""
    title = driver.title.lower()
    if "just a moment" in title:
        return "Cloudflare verification — tick the checkbox in Chrome"
    if "challenge" in title:
        return "Cloudflare challenge in Chrome — complete it to continue"
    # Note: FBref's adblock overlay is in the DOM but doesn't block the HTML data,
    # so we don't pause for it — _try_dismiss_overlay handles it silently.
    return ""


def _try_dismiss_overlay(driver):
    """Best-effort click on common overlay dismiss buttons."""
    selectors = [
        "button.ad-blocking-message__btn",   # FBref adblock button
        "[id*='adblock'] button",
        "[class*='adblock'] button",
        "[class*='ad-block'] button",
        "button[data-tier='1']",              # consent / cookie popups
        "#onetrust-accept-btn-handler",
        ".qc-cmp2-summary-buttons button",
    ]
    for sel in selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            btn.click()
            time.sleep(1)
            return True
        except Exception:
            pass
    return False


def _get_html(driver, url, wait_css=None, timeout=30):
    driver.get(url)

    # Wait for Cloudflare JS challenge to auto-clear
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: "just a moment" not in d.title.lower()
        )
    except Exception:
        pass

    # Try auto-dismissing overlays (adblock warning, cookie consent)
    _try_dismiss_overlay(driver)

    # If page still needs human action, pause and ask
    reason = _needs_human(driver)
    if reason:
        tqdm.write(f"\n  [ACTION NEEDED] {reason}")
        tqdm.write(f"  Handle it in the Chrome window, then press Enter...")
        input()
        time.sleep(2)
        _try_dismiss_overlay(driver)   # try again after user acts

    if wait_css:
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_css))
            )
        except Exception:
            pass

    time.sleep(PAGE_SETTLE)
    return driver.page_source


# ── Helpers ────────────────────────────────────────────────────────────────────

SCRAPPED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Datasets", "Scrapped Datasets"
)

# Identifies one team's match uniquely
KEY_COLS = ["year", "team", "date", "opponent", "venue"]


def _merge_seasons(existing_df, season_dfs):
    """Replace re-scraped seasons instead of appending to them.

    Appending leaves the cached rows for a season sitting next to the fresh
    ones, so every re-run duplicates the last RESCRAPE_RECENT seasons.
    """
    parts = []
    if existing_df is not None and len(existing_df):
        fresh_years = set()
        for sdf in season_dfs:
            if "year" in sdf.columns:
                fresh_years |= {int(y) for y in sdf["year"].dropna().unique()}
        keep = existing_df
        if fresh_years and "year" in keep.columns:
            keep = keep[~pd.to_numeric(keep["year"], errors="coerce").isin(fresh_years)]
        parts.append(keep)

    parts.extend(season_dfs)
    merged = pd.concat(parts, ignore_index=True)

    key = [c for c in KEY_COLS if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    return merged


def _divider(char="=", width=62):
    print(char * width)

def _league_header(name, idx, total, n_seasons):
    print()
    _divider()
    print(f"  LEAGUE {idx}/{total}  |  {name}")
    print(f"  Seasons to scrape: {n_seasons}")
    _divider()


# ── Core scrape ────────────────────────────────────────────────────────────────

def scrape_data(url_filter=None):

    # SerieA_argentina = "https://fbref.com/en/comps/21/Primera-Division-Stats"
    SerieA_brazil      = "https://fbref.com/en/comps/24/Serie-A-Stats"
    LeagueA_norway     = "https://fbref.com/en/comps/28/Eliteserien-Stats"
    LeagueA_finland    = "https://fbref.com/en/comps/43/Veikkausliiga-Stats"
    PL_england         = "https://fbref.com/en/comps/9/Premier-League-Stats"
    SerieA_italy       = "https://fbref.com/en/comps/11/Serie-A-Stats"
    LaLiga_spain       = "https://fbref.com/en/comps/12/La-Liga-Stats"
    LigueA_france      = "https://fbref.com/en/comps/13/Ligue-1-Stats"
    Bundesliga_germany = "https://fbref.com/en/comps/20/Bundesliga-Stats"
    SuperLeague_greece = "https://fbref.com/en/comps/27/Super-League-Greece-Stats"

    top5 = {PL_england, SerieA_italy, LaLiga_spain, LigueA_france, Bundesliga_germany}

    urls = [
        SerieA_brazil,
        LeagueA_norway, LeagueA_finland,
        SuperLeague_greece, PL_england,
        SerieA_italy, LaLiga_spain,
        Bundesliga_germany, LigueA_france,
    ]

    if url_filter is not None:
        urls = [u for u in urls if u in url_filter]

    driver       = _make_driver()
    current_year = datetime.now().year
    grand_total  = 0
    run_start    = time.time()

    print()
    print("  ScoreCast Scraper  |  FBREF via Chrome")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}  |  {len(urls)} leagues")

    try:
        for league_idx, url in enumerate(urls, start=1):

            standings_url = url
            last_split    = standings_url.split('/')[-1]

            years = list(range(current_year, current_year - SEASONS, -1))

            if url == SerieA_italy:
                last_split = last_split + "-Italy"

            display_key  = last_split.replace('_', '-')
            league_name  = LEAGUE_NAMES.get(display_key, last_split)
            csv_path     = os.path.join(SCRAPPED_DIR, last_split.replace('-', '_') + ".csv")
            os.makedirs(SCRAPPED_DIR, exist_ok=True)

            # ── Resume: load what's already scraped ────────────────────────────
            existing_df   = None
            scraped_years = set()
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path, index_col=0, low_memory=False)
                    existing_df.columns = [c.lower() for c in existing_df.columns]
                    if 'year' in existing_df.columns:
                        scraped_years = {int(y) for y in existing_df['year'].dropna().unique()}
                except Exception:
                    existing_df = None

            _league_header(league_name, league_idx, len(urls), len(years))
            if scraped_years:
                print(f"  Cached seasons  : {sorted(scraped_years)}")
                print(f"  Scraping only   : seasons missing or last {RESCRAPE_RECENT} years")

            new_season_dfs  = []
            league_scraped  = 0
            league_skipped  = 0

            for year_idx, year in enumerate(years, start=1):

                if not standings_url:
                    tqdm.write(f"  [!] No previous-season link — stopping at {year}")
                    break

                # ── Load standings (always needed for the prev-season nav link) ─
                soup = None
                for attempt in range(3):
                    html  = _get_html(driver, standings_url,
                                      wait_css="table.stats_table", timeout=30)
                    soup  = BeautifulSoup(html, "html.parser")
                    if soup.select("table.stats_table"):
                        break
                    wait = 5 + attempt * 5
                    tqdm.write(f"  [!] No table (attempt {attempt+1}/3) — waiting {wait}s")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  [!] Could not load {year} standings — skipping season")
                    continue

                standings_table = soup.select("table.stats_table")[0]
                links           = standings_table.find_all('a')
                href            = [l.get('href') for l in links]
                href_squad      = [l for l in href if '/squads/' in l]
                team_urls       = [f'https://fbref.com{l}' for l in href_squad]

                prev_tag      = soup.select("a.prev")
                standings_url = f'https://fbref.com{prev_tag[0].get("href")}' if prev_tag else ""

                # ── Skip seasons already in the CSV (except recent ones) ────────
                if year in scraped_years and year < current_year - RESCRAPE_RECENT:
                    tqdm.write(f"  {year} ({year_idx}/{len(years)}): cached — skipping")
                    continue

                season_scraped = 0
                season_skipped = 0
                season_matches = []

                # ── Team progress bar ──────────────────────────────────────────
                bar_fmt = (
                    "  {desc}  {percentage:3.0f}%|{bar:22}|"
                    " {n_fmt}/{total_fmt} teams"
                    " [{elapsed}<{remaining}]  {postfix}"
                )
                with tqdm(team_urls, desc=f"  {year} ({year_idx}/{len(years)})",
                          bar_format=bar_fmt, unit="", leave=True) as pbar:

                    for team_url in pbar:
                        team_name = team_url.split('/')[-1].replace('-Stats','').replace('-',' ')
                        pbar.set_postfix_str(team_name[:28])

                        try:
                            html    = _get_html(driver, team_url,
                                                wait_css="table", timeout=30)
                            matches = pd.read_html(StringIO(html), match='Scores & Fixtures')[0]

                            soup_team     = BeautifulSoup(html, "html.parser")
                            href_all      = [l.get('href') for l in soup_team.find_all('a')]
                            href_shooting = [l for l in href_all
                                             if l and 'all_comps/shooting' in l]

                            if not href_shooting:
                                tqdm.write(f"    [-] {team_name}: no shooting page")
                                season_skipped += 1
                                continue

                            html_sh  = _get_html(driver,
                                                 f'https://fbref.com/{href_shooting[0]}',
                                                 wait_css="table", timeout=30)
                            shooting = pd.read_html(StringIO(html_sh), match='Shooting')[0]

                            if isinstance(shooting.columns, pd.MultiIndex):
                                shooting.columns = shooting.columns.droplevel(0)

                            team_data = pd.concat(
                                [matches,
                                 shooting[shooting.columns.difference(matches.columns)]],
                                axis=1,
                            )
                            team_data['Year'] = year
                            team_data['Team'] = team_name
                            season_matches.append(team_data)
                            season_scraped += 1

                        except ValueError as e:
                            tqdm.write(f"    [-] {team_name}: {str(e)[:120]}")
                            season_skipped += 1
                        except Exception as e:
                            tqdm.write(f"    [!] {team_name}: {str(e)[:120]}")
                            season_skipped += 1

                        time.sleep(TEAM_DELAY)

                tqdm.write(
                    f"  Season {year}: {season_scraped} scraped"
                    + (f", {season_skipped} skipped" if season_skipped else "")
                )
                league_scraped += season_scraped
                league_skipped += season_skipped

                # ── Incremental save after every season ────────────────────────
                if season_matches:
                    season_df = pd.concat(season_matches, ignore_index=True)
                    season_df.columns = [c.lower() for c in season_df.columns]
                    new_season_dfs.append(season_df)

                    saved_df = _merge_seasons(existing_df, new_season_dfs)
                    saved_df.to_csv(csv_path)
                    tqdm.write(f"  Saved {len(saved_df):,} rows -> {os.path.basename(csv_path)}")

            # ── Final export ───────────────────────────────────────────────────
            if not new_season_dfs and existing_df is None:
                print(f"\n  [!] No data — skipping export for {league_name}")
                continue

            last_split = last_split.replace('-', '_')
            match_df   = pd.read_csv(csv_path, index_col=0) if os.path.exists(csv_path) \
                         else pd.concat(new_season_dfs, ignore_index=True)

            grand_total += league_scraped
            elapsed_min  = (time.time() - run_start) / 60

            print()
            _divider("-")
            print(f"  {league_name} complete")
            print(f"  Teams scraped : {league_scraped}"
                  + (f"  |  skipped: {league_skipped}" if league_skipped else ""))
            print(f"  Rows saved    : {len(match_df):,}")
            print(f"  File          : {last_split}.csv")
            print(f"  Elapsed       : {elapsed_min:.1f} min")
            _divider("-")

    finally:
        driver.quit()

    # ── Grand summary ──────────────────────────────────────────────────────────
    total_min = (time.time() - run_start) / 60
    print()
    _divider()
    print(f"  SCRAPE COMPLETE")
    print(f"  Total teams scraped : {grand_total}")
    print(f"  Total time          : {total_min:.1f} min")
    print(f"  Finished            : {datetime.now().strftime('%H:%M:%S')}")
    _divider()


if __name__ == "__main__":
    scrape_data()