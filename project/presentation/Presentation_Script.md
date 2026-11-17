# Slide-by-Slide Presentation Script

**Deck:** `Volatility_Presentation.pdf` (13 slides)
**Target time:** about 15 to 17 minutes, roughly 1 to 1.5 minutes per slide.

How to use this:
- **Say** is what to speak.
- **Point at** tells you where to direct the audience on the graph.
- **If asked** gives short answers to likely questions on that slide.

Split suggestion for two speakers: Partha takes slides 1 to 7 (question, data, theory), Rajvi takes slides 8 to 13 (evaluation, results, conclusion).

---

## Slide 1 · Title

**Say:**
"Good morning. We are Partha Guntur and Rajvi Panchal. Our project is about forecasting the volatility of Indian stocks. We compare a classical econometric model, called HAR, with deep-learning models built on LSTM networks, and we ask whether the answer changes with how liquid a stock is."

---

## Slide 2 · Research Question

**Say:**
"Our question is simple: can an LSTM forecast tomorrow's realized volatility of Indian stocks better than the HAR model, and does that depend on liquidity?

Why care about volatility? It is the main input to option pricing, Value-at-Risk, exchange margins and position sizing. The problem is that you cannot observe it directly. You have to measure it from prices first, and then forecast it.

We test two competing hypotheses. H1 says a flexible model like a neural network should gain most in illiquid stocks, because their volatility is noisy and a straight-line model fits them badly. H2 says the opposite: a neural network only helps in liquid stocks, where the data is clean enough to learn from, and in illiquid stocks it just learns noise.

Our sample is 95 stocks from the NIFTY 100, using 5-minute NSE data. We train on 2016 to 2022, validate on 2023 to 2024, and test on 1 January 2025 to 9 April 2026. That test set is 29,774 stock-days that no model ever saw."

**If asked, "why this split?":**
"Seven training years so the model sees the COVID crash and different market conditions. Two validation years only to decide when to stop training. And the most recent 15 months as an untouched test, 314 trading days, which is enough for the statistical tests to have power. Roughly 67, 20 and 13 percent."

---

## Slide 3 · Theory: Realized Volatility

**Say:**
"First, how do we measure volatility. We assume the log price moves with a drift and a random part scaled by sigma, the instantaneous volatility. The true variance of a day is the integral of sigma squared over that day, called integrated variance. We cannot see it, but we can estimate it.

If we take many small returns during the day, square them and add them up, we get realized variance, RV. As the number of returns grows, RV converges to the true integrated variance.

We use 5-minute returns. Sampling more often gives more data, but below about 5 minutes prices bounce between bid and ask, and you start measuring that noise instead of real movement. So 5 minutes is the standard trade-off. That gives up to 75 returns per trading day. We exclude overnight gaps, and I will come back to why."

**Point at the graph (right side):**
"This chart shows the average volatility of our 95 stocks over time, smoothed over 5 days, in annualized percent. The shaded bands are our three periods: training, validation and test.

Notice three things. First, volatility clusters: long calm stretches, then sudden spikes. Second, the biggest spike is the COVID crash in March 2020, marked here, and there is another spike at the June 2024 election result. Third, after each spike volatility decays slowly, not instantly. That slow decay is called long memory, and it is exactly what our models try to capture. Also note the test period on the right is relatively calm."

**If asked, "why exclude overnight returns?":**
"Stock splits, bonus issues and demergers happen overnight. If the data is not adjusted, they look like a 50 percent crash. Using only intraday returns means corporate actions can never contaminate the target."

---

## Slide 4 · Measuring Liquidity

**Say:**
"To test our hypotheses we need to rank stocks by liquidity. We use the Amihud illiquidity ratio on 5-minute bars: the absolute return divided by the rupee value traded, averaged over all bars. In words, it is how much the price moves per rupee traded. If small trades move the price a lot, the stock is illiquid.

Two design choices matter. We compute it only on the training years, 2016 to 2022, so the groups contain no information from the test period. And we checked it against average traded value: the rank correlation is minus 0.91, so it agrees almost perfectly with the obvious measure.

We split the 95 stocks into three groups: 32 high, 31 medium and 32 low liquidity."

**Point at the graph:**
"Each bar is one stock, ordered from most liquid on the left to least liquid on the right. The y-axis is on a log scale, so each gridline is ten times the previous one. Colours show the three groups. Reliance is the most liquid stock; at the far right are names like Solar Industries and CG Power. The illiquidity of the last stock is roughly 8,000 times that of Reliance, which shows how wide the range is even inside the NIFTY 100."

**Point at the small table:**
"And the groups really behave differently. Average annual volatility rises from 25 percent for liquid stocks to 28.5 and then 33 percent for illiquid ones. The last column is the day-to-day autocorrelation of log volatility: 0.70 for liquid stocks and 0.65 for the others. Lower persistence means volatility is harder to forecast in illiquid stocks."

---

## Slide 5 · Theory: The HAR Model

**Say:**
"Our benchmark is HAR, the Heterogeneous Autoregressive model of Corsi, 2009. It predicts tomorrow's RV from three numbers: today's RV, the average over the last 5 days, and the average over the last 22 days. A daily, a weekly and a monthly component. We estimate one regression per stock by ordinary least squares.

The idea behind it is the heterogeneous market hypothesis: the market has day traders, weekly traders and monthly investors, and each reacts to volatility at their own horizon. Three simple averages are enough to mimic long memory with only four parameters. That is why HAR is so hard to beat.

We also use log-HAR: the same regression on the log of RV, with a small correction when converting the forecast back. Logs stop a few extreme days from dragging the coefficients around."

**Point at the coefficient table:**
"This table shows the average coefficients across stocks. In logs, the daily, weekly and monthly weights are 0.35, 0.25 and 0.21, summing to about 0.80, which means volatility is highly persistent. In levels the weights are less stable from stock to stock, because a few crisis days in training distort them. This turns out to matter a lot in the results."

**If asked, "why add log-HAR?":**
"Because our neural network also sees log inputs. Comparing it only with plain HAR would mix up two effects: the benefit of logs and the benefit of the network. log-HAR separates them."

---

## Slide 6 · Theory: EGARCH

**Say:**
"The second family of features comes from EGARCH, Nelson 1991, fitted to daily returns with fat-tailed Student-t errors. EGARCH models the log of variance, so variance is always positive without any constraints.

Each parameter has a meaning. Alpha is the size effect: big shocks raise tomorrow's variance. Gamma is the asymmetry: if it is negative, bad news raises volatility more than good news, which is called the leverage effect. Beta is persistence.

An important point: we estimate EGARCH on 2016 to 2022 only, then freeze the parameters and run it forward. So the test-period features never use future information. The outputs, estimated volatility sigma and the standardized shock z, become the inputs of our EGARCH-LSTM."

**Point at the parameter table:**
"Across the 95 stocks, gamma averages minus 0.042, and it is negative for 92 percent of stocks, exactly as theory predicts. Beta averages 0.92, so volatility is very persistent. And nu, the tail parameter, is about 4.8. A normal distribution would have nu equal to infinity, so this confirms daily returns have fat tails."

---

## Slide 7 · Theory: LSTM Network

**Say:**
"Our main model is an LSTM, a Long Short-Term Memory network. It is built for sequences. It reads the data one day at a time and keeps a memory, called the cell state, c. Three gates control that memory. The forget gate decides what old information to drop, the input gate decides what new information to store, and the output gate decides what to pass on. That is what these equations describe.

Our design: the input is the last 22 trading days. For HAR-RV-LSTM those are the three log HAR features, so the network sees exactly the same information as the HAR benchmark. If it wins, it wins because of how it uses the information, not because it got more. For EGARCH-LSTM the inputs are the two EGARCH features.

We use two LSTM layers of 64 units, plus a small learned stock embedding, and one model for all 95 stocks. One stock alone has only about 1,700 training days, too few for a neural network, so pooling gives it about 152,000 examples. The embedding tells the network which stock it is looking at. The output is passed through an exponential so the forecast is always positive, and we train on the QLIKE loss. We train each network with three random seeds and average them."

**Point at the training graph:**
"These are the training curves, one panel per model. The x-axis is the training epoch, one full pass through the data. The y-axis is the QLIKE loss. Dashed lines are training loss, solid lines are validation loss, and there are three lines for the three seeds.

Look at the level. The HAR-RV-LSTM, on the left, reaches a validation loss of about 0.226 within a few epochs. The EGARCH-LSTM in the middle levels off much higher, around 0.27. So already during training we can see that daily-return features carry less information. We use early stopping: when the validation loss stops improving for 8 epochs, we stop and keep the best version, which is why the curves have different lengths."

**If asked, "is it overfitting?":**
"Training loss keeps falling while validation flattens, which is the classic sign. That is exactly why we use early stopping on the validation set, plus dropout of 0.2 and weight decay. And the three seeds land within 0.002 of each other on the test set."

---

## Slide 8 · Theory: Evaluating Forecasts

**Say (Rajvi):**
"Now, how do we decide which model is better.

We use two main losses. MSE, mean squared error, is the squared gap between actual and forecast. Because of the square, a few crisis days dominate it.

Our main metric is QLIKE: actual over forecast, minus the log of that ratio, minus one. It is zero only when the forecast is perfect. We prefer it for three reasons. First, our 'actual' value, RV, is itself only an estimate of true volatility, and Patton proved in 2011 that QLIKE still ranks forecasts correctly in that situation. Second, it depends on the ratio, so calm and wild stocks count equally and a few crisis days do not swamp the result. Third, it punishes under-prediction more than over-prediction, which matches reality, because underestimating risk is the costlier mistake.

To check whether a difference is real or just luck, we use the Diebold-Mariano test. Each day we average the loss difference between two models across all stocks, and test whether its mean is zero, using Newey-West standard errors because neighbouring days are correlated. We confirm it with a 5-day block bootstrap.

Finally, Mincer-Zarnowitz: we regress actual log volatility on the forecast. A slope near one means the forecast is well calibrated, and R-squared tells us how much of the day-to-day movement it explains."

---

## Slide 9 · Results: Out-of-Sample Accuracy

**Say:**
"Here are the results on the 29,774 test stock-days."

**Point at the graph:**
"Three panels: QLIKE, MAE and MSE. Each bar is one model, and lower is better. The four colours are HAR-RV, log-HAR, HAR-RV-LSTM and EGARCH-LSTM. In every panel the HAR-RV-LSTM bar is the shortest, and the EGARCH-LSTM bar is the tallest."

**Point at the table:**
"In numbers: QLIKE is 0.2286 for HAR-RV, 0.1965 for log-HAR, 0.1935 for HAR-RV-LSTM and 0.2709 for EGARCH-LSTM. So our HAR-RV-LSTM is the most accurate: 15.4 percent better than HAR-RV.

But here is the key piece of analysis. log-HAR alone is already 14 percent better than HAR-RV. So of the 15.4 points, about 14 come just from using logs, and the neural network adds the last 1.6 percent on top of log-HAR. Many papers compare a neural network only with plain HAR and report a big win that is mostly the log transformation. We wanted to separate the two.

The calibration columns support this. Mincer-Zarnowitz R-squared is highest for the LSTM, 0.470. And the last column shows bias: plain HAR-RV over-predicts by 21 percent on average, log-HAR by 8 percent, the LSTM by only 6 percent. EGARCH-LSTM over-predicts by 39 percent: daily closing prices simply lose the information inside the day."

---

## Slide 10 · Results: Are the Differences Significant?

**Point at the graph first:**
"This chart shows the cumulative advantage of our model through the test period. The x-axis is time. The y-axis adds up, day by day, how much lower the LSTM's loss was than the other model's. A line that keeps rising means the LSTM keeps winning day after day, not just on a few lucky days.

The left panel compares against HAR-RV and EGARCH-LSTM: both lines rise steadily all year. The right panel compares against log-HAR, and note its scale is much smaller. The line is roughly flat until about June 2025, then climbs steadily. So the gain over log-HAR is small, but it is consistent in the second half."

**Point at the table:**
"Statistically: against HAR-RV the t-statistic is 6.18; against log-HAR it is 2.80 with an adjusted p-value of 0.010; against EGARCH-LSTM it is 9.89. All three are significant, and the bootstrap confidence intervals exclude zero.

One more test: we also trained a network with both HAR and EGARCH inputs. It is not better than HAR-RV-LSTM, t equals 0.24. So once the network knows the intraday volatility history, EGARCH adds nothing."

**If asked, "how robust is 1.6 percent?":**
"Every one of the three random seeds beats log-HAR on its own, test QLIKE between 0.1926 and 0.1959 against 0.1965. And the LSTM beats log-HAR on 67 of the 95 stocks, and beats HAR-RV on all 95."

---

## Slide 11 · Analysis: Does Liquidity Matter?

**Say:**
"This is our core research question."

**Point at the left graph:**
"The left chart shows test QLIKE for each model inside each liquidity group. The green HAR-RV-LSTM bar is the lowest in all three groups, and EGARCH-LSTM gets noticeably worse as liquidity falls."

**Point at the right graph:**
"The right chart is per stock. Each dot is one stock. The x-axis is liquidity rank, 1 being the most liquid, on the left. The y-axis is how much the LSTM reduces the error compared with log-HAR, in percent. Dots above zero are stocks where the LSTM wins.

On the left, among the liquid stocks, the dots sit around zero, some even below: ICICI Bank and SBI are about 5 percent worse with the LSTM. Moving right, more dots sit above zero, and the biggest gains are all illiquid stocks: Lodha at 19 percent, Max Healthcare at 14 and Adani Green at 10."

**Point at the table:**
"Grouped together: the gain over log-HAR is 0.3 percent for liquid stocks, which is not significant, 1.4 percent for medium, and 3.0 percent for illiquid stocks, with t-statistics of 0.38, 2.43 and 3.96.

So the advantage grows as liquidity falls. This supports H1. Our explanation is that illiquid stocks have noisier, less persistent volatility. A fixed linear rule fits them poorly, while a network that reads the whole 22-day pattern can adapt. For the most liquid stocks, log-HAR is already as good."

---

## Slide 12 · Analysis: Market Conditions and Stability

**Point at the graph:**
"This chart shows, for the test period, the average actual volatility across all 95 stocks in black, and the average forecasts of HAR-RV, log-HAR and the LSTM. All in annualized percent.

Look at the calm second half of 2025: the HAR-RV line sits clearly above the black line. That is the 21 percent over-prediction we saw earlier. The LSTM and log-HAR lines follow the black line much more closely. Now look at the spikes, for example in April 2025 and early 2026: every model lags a little, because no model knows a shock is coming."

**Point at the table:**
"First, regimes we can know in advance. Using market volatility cut-offs from the training years, the test had 253 calm days and 61 normal days, and no turbulent days at all. The LSTM beats log-HAR in both.

But on the third of days with the highest actual volatility, the LSTM is slightly worse than log-HAR, 0.292 against 0.281. That is an honest weakness: on spike days it reacts a little slower.

Over time, splitting the test into four sub-periods, the LSTM wins three of the four. The only exception is January to April 2025, which includes the April tariff shock, where log-HAR is marginally ahead."

---

## Slide 13 · Conclusion and Applications

**Say:**
"To conclude. The HAR-RV-LSTM gives the most accurate volatility forecasts: 15.4 percent better than HAR-RV and 1.6 percent better than log-HAR, both statistically significant. Most of the improvement over HAR-RV comes from using logs; the network adds a smaller but real gain. That gain grows as liquidity falls, up to 3 percent in the least liquid stocks, supporting H1. And EGARCH features add nothing beyond intraday volatility.

Where can this be used? In options: the fair price of an at-the-money straddle is about 0.8 times sigma times the stock price times root T, so better sigma means better pricing and hedging. In risk: Value-at-Risk and futures margins are a multiple of forecast sigma times the position. In portfolios: position size equals risk budget divided by forecast sigma, which is volatility targeting. The practical rule from our results: for liquid large caps, log-HAR is enough; for a broad portfolio with less liquid names, use the LSTM.

Our limitations: one fairly calm test year, and today's NIFTY 100 members only. Next steps would be adding India VIX, jump measures and market-wide volatility, and testing through a crisis.

We have also built an interactive simulator that shows these results for every stock, which we can demonstrate now. Thank you."

*[If showing the simulator, switch to the Streamlit app; the walkthrough is in Part B of `Full_Project_Transcript.md`.]*

---

## Likely questions and short answers

**Why QLIKE and not R² or accuracy?**
Volatility is continuous and RV is a noisy estimate of it. QLIKE is proven to rank forecasts correctly under a noisy proxy, and it is not dominated by a few extreme days. We also report MSE, MAE and Mincer-Zarnowitz R².

**Is 1.6 percent economically meaningful?**
It is small on average but consistent, significant, true for every seed, and it rises to 3 percent where it matters most, in illiquid stocks where execution and pricing errors are most expensive. In our simulator, on ₹1 crore of daily straddle notional, the average pricing error falls from about ₹36,700 with HAR-RV to ₹30,600 with the LSTM.

**Why not a Transformer or a bigger network?**
One pooled LSTM with about 60,000 parameters against 152,000 training sequences is already enough to match a strong benchmark. A larger model would risk overfitting, and the comparison with log-HAR shows the remaining signal is small.

**Did you have look-ahead bias?**
No. Splits are by the date being predicted; liquidity groups, scaling and EGARCH parameters all use training years only; the test set was never used for any decision.

**Why did your results change from the first version?**
We found three issues: dirty RV with overnight gaps and bad prints, a mis-scaled liquidity measure, and EGARCH fitted on the full sample including test data. Fixing them made HAR much stronger and EGARCH-LSTM weaker. HAR-RV-LSTM remained the best model.

**Why is the test period calm, and does that bias things?**
That was simply the market in 2025 and early 2026. It means our results speak mostly to calm and normal markets; testing through a crisis is our main future step.
