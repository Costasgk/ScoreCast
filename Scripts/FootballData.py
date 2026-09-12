"""
football-data.co.uk fetcher — a fast alternative to the FBref scrape.

FBref sits behind Cloudflare and needs a real browser (see Scrapping.py), which
costs ~1.5 hours and a manual verification tick.  football-data.co.uk publishes
the same results as plain CSV over HTTP, so the whole dataset lands in seconds.

Writes straight into 'Cleaned Datasets' in the long, one-row-per-team-per-match
shape that ScorelineModel.load_matches expects, skipping the scrape/clean steps.

Trade-off: this source carries goals, shots and shots-on-target, but no
penalties, possession, formation or xG.  That is everything the Dixon-Coles
scoreline model uses, but NOT enough for the RandomForest in Modelling.py,
which needs pk/pkatt — keep using Scrapping.py when you need those.

Usage:
  python FootballData.py              # all leagues
  python FootballData.py --leagues "England Premier League"
"""

import argparse
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT    = Path(__file__).resolve().parent.parent
CLEANED = ROOT / 'Datasets' / 'Cleaned Datasets'

BASE     = "https://www.football-data.co.uk"
SEASONS  = ["2526", "2425", "2324", "2223", "2122"]
HEADERS  = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/153.0.0.0 Safari/537.36"}

from leagues import LEAGUES as REGISTRY

# Adapter: the registry is the single source of truth, these are the fields the
# fetcher needs from it.
LEAGUES = [
    {**l, 'out': l['file'], 'kind': l['source']} for l in REGISTRY
]


def _get_csv(session, url):
    """Fetch one CSV, or None if it isn't published."""
    r = session.get(url, timeout=30)
    if r.status_code != 200 or len(r.text) < 500:
        return None
    # These files are latin-1 with a UTF-8 BOM on the first header cell
    df = pd.read_csv(io.StringIO(r.text), encoding="latin-1", on_bad_lines="skip")
    # Read as latin-1, so a UTF-8 BOM arrives as the literal chars "ï»¿"
    df.columns = [c.lstrip("﻿").lstrip("ï»¿").strip() for c in df.columns]
    return df


def _to_long(df, comp):
    """Turn one-row-per-match into two rows (home view + away view)."""
    out = pd.DataFrame({
        "season":   df["season"] if "season" in df else None,
        "date":     df["date"],
        "time":     df.get("time"),
        "comp":     comp,
        # Files hold league fixtures only, no cup rounds. ScorelineModel keeps
        # rows whose round starts with "Matchweek", so label them as such.
        "round":    "Matchweek",
        "referee":  df.get("referee"),
        "home":     df["home"].astype(str).str.strip(),
        "away":     df["away"].astype(str).str.strip(),
        "hg":       pd.to_numeric(df["hg"], errors="coerce"),
        "ag":       pd.to_numeric(df["ag"], errors="coerce"),
        "hs":       pd.to_numeric(df.get("hs"), errors="coerce") if "hs" in df else pd.NA,
        "as_":      pd.to_numeric(df.get("as"), errors="coerce") if "as" in df else pd.NA,
        "hst":      pd.to_numeric(df.get("hst"), errors="coerce") if "hst" in df else pd.NA,
        "ast":      pd.to_numeric(df.get("ast"), errors="coerce") if "ast" in df else pd.NA,
    })
    out = out[out["home"].notna() & out["away"].notna() & (out["home"] != "nan")]

    def side(venue):
        is_home = venue == "Home"
        d = pd.DataFrame({
            "season":   out["season"],
            "year":     out["date"].dt.year,
            "date":     out["date"],
            "time":     out["time"],
            "comp":     out["comp"],
            "round":    out["round"],
            "venue":    venue,
            "referee":  out["referee"],
            "team":     out["home"] if is_home else out["away"],
            "opponent": out["away"] if is_home else out["home"],
            "gf":       out["hg"]  if is_home else out["ag"],
            "ga":       out["ag"]  if is_home else out["hg"],
            "sh":       out["hs"]  if is_home else out["as_"],
            "sot":      out["hst"] if is_home else out["ast"],
        })
        diff = d["gf"] - d["ga"]
        d["result"] = diff.map(lambda x: pd.NA if pd.isna(x) else ("W" if x > 0 else "L" if x < 0 else "D"))
        return d

    return pd.concat([side("Home"), side("Away")], ignore_index=True)


def _season_code(dates):
    """
    European season label from a date: 2026-09 -> '2627', 2026-03 -> '2526'.

    The fixtures feed carries no season column, and its games belong to the
    season that is about to start, not the one the history ends in. Labelling
    them with the newest history code merges promoted clubs into last season.
    """
    d = pd.to_datetime(dates, errors="coerce")
    start = d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)
    return (start % 100).map("{:02d}".format) + ((start + 1) % 100).map("{:02d}".format)


def _norm_main(df, season=None):
    """mmz4281 files: Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HST, AST."""
    cols = {c.lower(): c for c in df.columns}
    take = lambda k: df[cols[k]] if k in cols else None
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(take("date"), dayfirst=True, errors="coerce"),
        "time": take("time"),
        "referee": take("referee"),
        "home": take("hometeam"), "away": take("awayteam"),
        "hg": take("fthg"), "ag": take("ftag"),
    })
    for k in ("hs", "as", "hst", "ast"):
        if k in cols:
            out[k] = df[cols[k]]
    return out[out["date"].notna()]


def _norm_extra(df):
    """new/<country> files: Season, Date, Time, Home, Away, HG, AG (no shot data)."""
    cols = {c.lower(): c for c in df.columns}
    take = lambda k: df[cols[k]] if k in cols else None
    out = pd.DataFrame({
        "season": take("season").astype(str) if "season" in cols else None,
        "date": pd.to_datetime(take("date"), dayfirst=True, errors="coerce"),
        "time": take("time"),
        "referee": None,
        "home": take("home"), "away": take("away"),
        "hg": take("hg"), "ag": take("ag"),
    })
    return out[out["date"].notna()]


def fetch_league(session, cfg, fixtures_main, fixtures_extra):
    frames = []

    if cfg["kind"] == "main":
        for s in SEASONS:
            df = _get_csv(session, f"{BASE}/mmz4281/{s}/{cfg['div']}.csv")
            if df is not None:
                frames.append(_norm_main(df, season=s))
        if fixtures_main is not None:
            fx = fixtures_main[fixtures_main["Div"].astype(str).str.strip() == cfg["div"]]
            if len(fx):
                nf = _norm_main(fx)
                nf["season"] = _season_code(nf["date"]).values
                frames.append(nf)
    else:
        df = _get_csv(session, f"{BASE}/new/{cfg['country']}.csv")
        if df is not None:
            # Some rows carry a trailing space in the league name
            df = df[df["League"].astype(str).str.strip() == cfg["league"]]
            # These files go back ~15 years; keep the same window as the
            # main-league files so every league trains on comparable depth.
            # Some countries label seasons '2026', others '2012/2013'. Both
            # sort chronologically as plain strings; parsing them as numbers
            # silently dropped every split-year league.
            lab = df["Season"].astype(str).str.strip()
            keep = sorted(lab.dropna().unique())[-len(SEASONS):]
            df = df[lab.isin(keep)]
            frames.append(_norm_extra(df))
        if fixtures_extra is not None:
            fx = fixtures_extra[fixtures_extra["League"].astype(str).str.strip() == cfg["league"]]
            if len(fx):
                nf = _norm_extra(fx)
                # These leagues run on calendar years, so the fixture's own year
                # is its season.
                nf["season"] = pd.to_datetime(nf["date"], errors="coerce").dt.year.astype("Int64").astype(str)
                frames.append(nf)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "home", "away"], keep="last")
    return _to_long(merged, cfg["name"])


def main():
    ap = argparse.ArgumentParser(description="Fetch match data from football-data.co.uk")
    ap.add_argument("--leagues", nargs="+", metavar="NAME")
    args = ap.parse_args()

    pool = LEAGUES
    if args.leagues:
        want = {n.lower() for n in args.leagues}
        pool = [l for l in LEAGUES if l["name"].lower() in want]
        if not pool:
            raise SystemExit(f"No leagues matched. Available: {[l['name'] for l in LEAGUES]}")

    CLEANED.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    fixtures_main  = _get_csv(session, f"{BASE}/fixtures.csv")
    fixtures_extra = _get_csv(session, f"{BASE}/new_league_fixtures.csv")

    print(f"\n  football-data.co.uk  |  {len(pool)} leagues  |  seasons {SEASONS[-1]}-{SEASONS[0]}\n")
    for cfg in pool:
        df = fetch_league(session, cfg, fixtures_main, fixtures_extra)
        if df is None or df.empty:
            print(f"  FAIL  {cfg['name']}")
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(CLEANED / cfg["out"])
        played   = int(df["gf"].notna().sum() // 2)
        upcoming = int(df["gf"].isna().sum() // 2)
        print(f"  OK    {cfg['name']:24s} {played:5d} played  {upcoming:3d} upcoming  "
              f"-> {cfg['out']}")


if __name__ == "__main__":
    main()
