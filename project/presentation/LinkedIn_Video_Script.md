# LinkedIn Video Script: Volatility Forecast Simulator (1 minute)

About 160 words, about 60 seconds. Lines in *[brackets]* are what to do on screen; do not read them.
Every number below appears on screen in the simulator at that moment.

---

*[0:00 — Tab **1 · Forecast vs reality**. Stock: **RELIANCE**.]*

"This is a volatility forecast simulator we built for 95 NIFTY 100 stocks. The black line is the volatility that actually happened; the coloured lines are each model's next-day forecast. For Reliance, the most liquid stock, the simple log-HAR model wins: QLIKE 0.177."

*[0:15 — Change stock to **ADANIGREEN**.]*

"For a less liquid stock like Adani Green, our HAR-RV-LSTM wins: QLIKE 0.260 against 0.365 for HAR-RV."

*[0:25 — Click tab **5 · Does liquidity matter?**]*

"Across stocks, the LSTM's gain over log-HAR is 3.0% in the least liquid third, and only 0.2% in the most liquid."

*[0:35 — Click tab **2 · Options desk**. Notional: **1 crore**.]*

"For an options desk quoting a 1-crore straddle on Adani Green, the average pricing error drops from 75,412 rupees a day with HAR-RV to 55,487 with the LSTM."

*[0:48 — Scroll down to the **All 95 stocks** table.]*

"Across all 95 stocks, the LSTM is best on 66, and keeps position risk within 5% of budget. Link in the comments."

---

## Numbers on screen (for checking)

| Said | Value | Where in the simulator |
|---|---|---|
| Reliance, log-HAR QLIKE | 0.177 | Tab 1, RELIANCE, yellow box |
| Adani Green, LSTM vs HAR-RV QLIKE | 0.260 vs 0.365 | Tab 1, ADANIGREEN, side panel |
| Gain over log-HAR by liquidity | 3.0% (least liquid) vs 0.2% (most liquid) | Tab 5, yellow box |
| Straddle pricing error, ₹1 cr, Adani Green | HAR-RV ₹75,412, LSTM ₹55,487 per day | Tab 2, ADANIGREEN, side panel |
| Best model on | 66 stocks | All-stocks table, "Best on" |
| Risk off budget | 5% (LSTM) | All-stocks table, "Risk off budget" |
