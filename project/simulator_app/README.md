# Volatility Forecast Simulator (Streamlit)

Interactive version of the project results for 95 NIFTY 100 stocks (test period 1 Jan 2025 – 9 Apr 2026).

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

It opens in your browser at http://localhost:8501.

## Files

- `app.py`: the plain-language app (five tabs plus a scorecard). Run it from a terminal, not a notebook.
- `offline_backup.html`: backup if Streamlit will not start; double-click to open in any browser.
- `app_technical.py`: the earlier technical version (`streamlit run app_technical.py`).
- `simulator_data.csv`: one row per stock and test day. `rv`, `har`, `lhar`, `lstm`, `eg` are daily
  volatility in basis points (realized, and the four model forecasts); `ret` is the intraday return
  in basis points. Built from `results/forecasts/*_test_forecasts.csv` and the daily panel.

## Free hosting (optional)

Push this folder to a GitHub repository and deploy it at https://share.streamlit.io to get a public link.
