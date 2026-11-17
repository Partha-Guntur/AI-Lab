# Realized-Volatility Forecasting for Indian Equities: HAR-RV, EGARCH and LSTM

AI Lab project, M.Sc. Quantitative Finance, Ahmedabad University
Partha Guntur · Rajvi Panchal

Next-day realized volatility of 95 NIFTY 100 stocks is forecast from 5-minute NSE data (2016 – April 2026)
with five models and compared overall, by liquidity group and by market regime.

| Model | What it uses |
|---|---|
| HAR-RV | daily / weekly / monthly realized variance, linear (Corsi, 2009) |
| log-HAR | same, in logs |
| **HAR-RV-LSTM** | same three RV components over 22 days, pooled LSTM with stock embedding |
| EGARCH-LSTM | EGARCH(1,1)-t volatility and shock from daily returns, pooled LSTM |
| HAR-EGARCH-LSTM | both input sets (ablation) |

**Headline (test, 1 Jan 2025 – 9 Apr 2026, 29,774 stock-days):** QLIKE HAR-RV 0.2286, log-HAR 0.1965,
HAR-RV-LSTM 0.1935, EGARCH-LSTM 0.2709. The LSTM's gain over log-HAR grows from 0.3% (most liquid third)
to 3.0% (least liquid third).

## Folder layout

```
Project/
├── README.md, requirements.txt
├── data/                      raw 5-minute CSVs (about 1 GB; 100 stocks + NIFTY 50, NIFTY BANK, INDIA VIX)
├── src/volforecast/           all reusable code (config, cleaning, HAR, EGARCH, LSTM, evaluation, plotting)
├── notebooks/                 run in order 01 → 06
├── data_processed/            clean daily panel, EGARCH features, liquidity groups, universe
├── results/
│   ├── forecasts/             forecast for every stock-day (validation and test)
│   ├── tables/                every table in the paper as CSV
│   └── models/                trained LSTM weights (3 seeds per model)
├── figures/                   all charts (PNG to view, PDF for LaTeX)
├── paper/                     report: Volatility_Forecasting_Paper.pdf, single-file .tex, multi-file LaTeX source
├── presentation/              slides (PDF + .tex), slide-by-slide script, full project transcript, video script
├── simulator_app/             Streamlit simulator (app.py), technical version, offline backup page
└── docs/                      Milestone 1 proposal
```

## How to run

```
pip install -r requirements.txt
```

**Simulator (for the presentation):** in a terminal (not a notebook)

```
cd simulator_app
streamlit run app.py
```

It opens at http://localhost:8501. If Streamlit fails, double-click `simulator_app/offline_backup.html`;
it opens in any browser with no installation (same data, technical model names).

**Full pipeline:** run notebooks 01 → 06. Notebook 01 reads `data/` (about 2 minutes); 04 and 05 train
3 networks each (20–40 minutes each on a laptop). Notebook 06 rebuilds every table and figure from the
saved forecasts and can be run on its own. The index files (NIFTY 50, NIFTY BANK, INDIA VIX) in `data/`
have no volume, so the cleaning step drops them automatically; results are unchanged.

**Paper:** `cd paper && latexmk -pdf main.tex`, or upload `Volatility_Forecasting_Report.tex` plus the
`figures/*.pdf` files to Overleaf.

**Slides:** `cd presentation && pdflatex Volatility_Presentation.tex` (reads figures from `../figures/`).

## Data cleaning in one paragraph

Bars outside 09:15–15:25 are dropped; rows before the listing date (IRFC, SBILIFE, HDFCLIFE, MAZDOCK)
are removed; zero-volume quotes carry no price; isolated spikes reversed in the next bar are removed
(e.g. VBL 14-Jun-2023, 322 → 644 → 322); days with fewer than 60 traded bars or with RV = 0 are dropped.
RV uses intraday returns only, so overnight splits, bonus issues and demergers never enter the target.
Details: `src/volforecast/config.py` and `results/tables/data_cleaning_log.csv`.

## Changes from the first version of the project

| Issue in the first version | Fix |
|---|---|
| Liquidity slope divided signed volume by its own near-zero mean, so the groups were close to random | 5-minute Amihud ratio on training years only |
| EGARCH estimated on 2016–2026, so test-year information leaked into the features | parameters estimated on 2016–2022, then run forward with fixed parameters |
| RV included overnight gaps and bad prints, inflating HAR errors | intraday-only RV and the cleaning rules above |
| One training run per LSTM | three seeds, averaged, seed spread reported |
| Volatility regimes defined with the realized outcome | ex-ante regimes from training-period cut-offs; ex-post kept as descriptive |
| No log-HAR benchmark | added; it is the fair benchmark for a model with log inputs |
