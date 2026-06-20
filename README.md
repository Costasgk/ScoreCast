# ScoreCast

![ScoreCast](ScoreCast.png)

A football match prediction web app powered by the **Dixon-Coles Poisson model** — covering 11 leagues across Europe, South America, and Asia.

🌐 **Live at [costas.pythonanywhere.com](https://costas.pythonanywhere.com)**

---

## What it does

ScoreCast scrapes historical match data from [FBref](https://fbref.com), fits a Dixon-Coles model per league, and generates predictions for all upcoming fixtures. For each match it produces:

- Win / Draw / Loss probabilities
- Expected goals (xG) for each team
- Full scoreline distribution (0-0 through 5-5)
- Most likely score & top 3 scorelines
- Both Teams to Score %
- Over 1.5 / 2.5 / 3.5 goals %

## Leagues covered

| League | Country |
|--------|---------|
| Premier League | England |
| Serie A | Italy |
| La Liga | Spain |
| Ligue 1 | France |
| Bundesliga | Germany |
| Super League | Greece |
| Serie A | Brazil |
| Serie B | Brazil |
| Eliteserien | Norway |
| Veikkausliiga | Finland |
| J1 League | Japan |

## Features

- **Best Picks** — fixtures where the model has 65%+ confidence, grouped by date
- **Simulator** — pick any two teams from any league and run a custom matchup
- **Rankings** — model strength ranking vs actual league standings, with over/underperformer highlights
- **Team DNA** — attack/defence ratings, win %, goals scored/conceded, last 5 form
- **Model Accuracy** — backtested accuracy across all leagues for the last 12 months
- **Visitor Stats** — lightweight analytics dashboard at `/stats`

## How it works

```
Scrapping.py  →  Cleaning.py  →  ScorelineModel.py  →  Flask web app
   FBref            clean &           Dixon-Coles          serves
  scraper          validate           Poisson fit         predictions
```

1. **Scraping** — `undetected_chromedriver` bypasses Cloudflare on FBref; incremental saves per season; persistent Chrome profile so Cloudflare only needs solving once
2. **Cleaning** — normalises columns, filters bad rows, exports per-league CSVs
3. **Modelling** — fits attack/defence parameters per team via maximum likelihood (L-BFGS-B); exponential time-decay (half-life ~107 days) weights recent matches far more than old ones; Dixon-Coles low-score correction adjusts 0-0, 1-0, 0-1, 1-1 probabilities
4. **Pipeline** — `pipeline.py` orchestrates all three steps with smart staleness detection

## Running locally

```bash
# 1. Clone and set up environment
git clone https://github.com/Costasgk/ScoreCast.git
cd ScoreCast
python -m venv env
env\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Run the pipeline (scrape → clean → predict)
cd Scripts
python pipeline.py

# 3. Start the web app
cd WebApp
python app.py
```

The app runs at `http://127.0.0.1:5000`.

> **Note:** First-time scraping takes several hours (FBref rate limits). Subsequent runs are incremental and much faster.

## Project structure

```
Scripts/
├── Scrapping.py        # FBref scraper
├── Cleaning.py         # data cleaning
├── ScorelineModel.py   # Dixon-Coles model
├── pipeline.py         # orchestrates scrape → clean → predict
└── WebApp/
    ├── app.py          # Flask app
    ├── templates/      # HTML templates
    └── static/         # CSS, fonts, favicon

Datasets/
├── Scrapped Datasets/  # raw FBref output
├── Cleaned Datasets/   # cleaned per-league CSVs
├── Predictions/        # upcoming fixture predictions
└── Models/             # fitted model parameters (JSON)
```

## Deployment

Hosted on **PythonAnywhere** (free tier). To redeploy after regenerating predictions:

1. Rebuild the deployment zip locally
2. Upload to PythonAnywhere Files tab
3. Unzip and hit Reload on the Web tab

## License

MIT — free to use, modify, and distribute.
