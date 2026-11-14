# LinkedIn Video Script: Volatility Forecast Simulator (1 minute)

About 155 words, about 60 seconds at a normal pace.
Lines in *[brackets]* are what to do on screen; do not read them aloud.

---

*[0:00 — Open the simulator on tab **1 · Forecast vs reality**, stock **RELIANCE**.]*

"Can deep learning forecast stock volatility better than classic econometrics? We tested four models on 95 NIFTY 100 stocks, using ten years of 5-minute NSE data.

The black line is the volatility that actually happened. The coloured lines are each model's next-day forecast, made only from past data."

*[0:15 — Change the stock to **TRENT**.]*

"Overall, our LSTM has the lowest error: 15% below the standard HAR model, and it is the best model on 66 of the 95 stocks."

*[0:25 — Click tab **5 · Does liquidity matter?**]*

"Each dot is a stock. For liquid names, a simple model is enough. As liquidity falls, the LSTM's edge grows, to about 3%."

*[0:37 — Click tab **2 · Options desk**, notional **1 crore**.]*

"In real money: on a 1-crore straddle, better forecasts cut pricing errors from about 36,700 to 30,600 rupees a day."

*[0:47 — Click tab **4 · Portfolio sizing**.]*

"And position risk stays within 5% of budget."

*[0:53 — Scroll down to the **All 95 stocks** table.]*

"Better forecasts, better option prices, steadier risk. Paper and code in the comments."

---

## Numbers used (for checking)

| Figure | Value | Source |
|---|---|---|
| Data | 95 NIFTY 100 stocks, 5-minute NSE data | Paper, Section 3 |
| LSTM vs HAR-RV | QLIKE 15.4% lower | Paper, Table 4 |
| Best model on | 66 of 95 stocks | Simulator, all-stocks table |
| Gain over log-HAR, least liquid third | 3.0% | Paper, Table 7 |
| Straddle pricing error, ₹1 cr | HAR-RV ₹36,749, LSTM ₹30,590 per day | Simulator, all-stocks table |
| Risk off budget | LSTM 5% | Simulator, all-stocks table |
