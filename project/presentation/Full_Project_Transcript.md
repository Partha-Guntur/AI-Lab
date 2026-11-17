# Full Project Transcript

**Realized Volatility Forecasting in Indian Equities: HAR-RV, EGARCH and LSTM Models across Liquidity Groups**
Partha Guntur and Rajvi Panchal, M.Sc. Quantitative Finance, Ahmedabad University

How to use this file: read the plain text aloud. Lines in *[brackets]* are what to show on screen and are not read.
Part A explains the whole project (about 20 to 25 minutes spoken). Part B walks through the Streamlit site (about 6 to 8 minutes).
Every number comes from the result tables in `results/tables/` or from the Streamlit app.

---

# PART A — THE PROJECT

## A1. What the project is about

"Our project asks one question: can a deep-learning model predict tomorrow's stock market volatility better than the standard econometric model, and does the answer depend on how liquid the stock is?

Volatility simply means how much a price moves. It matters because it is the main input to almost every risk decision in finance: the price of an option, the margin an exchange asks for, the Value-at-Risk of a bank, and how much of each stock a portfolio manager holds. The difficulty is that volatility cannot be seen directly. You have to measure it from prices, and then forecast it.

We work with 95 stocks from the NIFTY 100, using 5-minute price data from the National Stock Exchange from January 2016 to April 2026.

We test two competing hypotheses. Hypothesis one says a flexible model like a neural network should help most in illiquid stocks, because their volatility is noisy and a simple linear rule fits them badly. Hypothesis two says the opposite: a neural network should help only in liquid stocks, because there the data are clean enough to learn from, and in illiquid stocks it would only learn noise. Indian large caps are a good laboratory for this, because average daily turnover in our sample ranges from about 9 crore rupees to about 1,360 crore rupees."

## A2. The overall architecture of the project

*[Show the project folder or the pipeline diagram from the slides.]*

"The project is a pipeline with six stages, and each stage is one notebook.

Stage one cleans the raw 5-minute data and builds one number per stock per day, called realized variance. It also ranks the stocks by liquidity.
Stage two fits the two econometric benchmarks, HAR-RV and log-HAR.
Stage three fits an EGARCH model to daily returns to create extra features.
Stage four trains our main model, the HAR-RV-LSTM.
Stage five trains the EGARCH-LSTM and one combined model as a check.
Stage six compares everything on data no model has seen, and runs the statistical tests.

All reusable code sits in one package, with separate files for configuration, data cleaning, the HAR models, EGARCH, the LSTM, and evaluation. Every setting, such as dates and cleaning thresholds, lives in one configuration file, so the whole project can be reproduced by running notebooks one to six in order."

## A3. The data and how we cleaned it

"We started with 18.8 million 5-minute bars for 100 stocks. Each bar has an open, high, low, close and volume. Raw high-frequency data is never clean, so we applied seven rules. I will explain each one and why.

One. We keep only the regular trading session, 9:15 a.m. to 3:30 p.m. That gives up to 75 bars per day. This removes special sessions like Diwali muhurat trading, which behave differently. After this filter, and after starting from November 2015, about 17.7 million bars remain.

Two. We remove rows before a stock's listing date. For four stocks, IRFC, SBI Life, HDFC Life and Mazagon Dock, the files contained placeholder or bond prices from before the IPO. These are not real stock prices.

Three. A bar with zero volume has no trade behind it, so its price is a stale quote. We ignore the price of 41,063 such bars.

Four. We remove isolated price spikes: a move of more than 5 percent in one bar that is almost fully reversed in the next bar. The clearest example is Varun Beverages on 14 June 2023, where the price went from 322 to 644 and back to 322 within ten minutes. That is a data error, not a market move. We found 24 such spikes. Genuine large moves, like the Adani stocks in February 2023, are not reversed, so they are kept.

Five. We drop days with fewer than 60 traded bars out of 75. These are half days or data outages, and a volatility number from a partial day is biased. That removed 1,455 stock-days.

Six. We drop 188 days where realized variance is exactly zero. That happens when a stock is frozen at a price band all day, so there is nothing to measure.

Seven, and this is the most important one: our volatility measure uses only returns inside the trading day. The overnight gap from yesterday's close to today's open is excluded. The reason is that stock splits, bonus issues and demergers all happen overnight, and if the data is not adjusted they look like a 50 percent crash. By using intraday returns only, corporate actions can never contaminate our target.

After cleaning we have 234,226 stock-days. As a sanity check, the ten most volatile days that remain are all real events: the Adani sell-off in February 2023 and the COVID crash of March 2020."

## A4. The target: realized variance

"The quantity we forecast is realized variance, or RV. For each day we take the 5-minute log returns, square them, and add them up. The theory behind this is simple: if you could observe prices continuously, the sum of squared returns would equal the true variance of the day, called integrated variance. With 5-minute data we get a close estimate. We use 5 minutes because it is the standard trade-off. Sampling more often gives more data points, but below about 5 minutes you start measuring bid-ask bounce instead of real price movement.

When we show volatility in percent per year, it is the square root of 252 times RV."

## A5. Why we split the data this way

"We split by time, never randomly, because in forecasting you must never train on the future.

Training is January 2016 to December 2022, seven years, with 152,935 stock-days. We wanted seven years because the training set must contain different market conditions: the calm of 2017, the 2018 NBFC stress, and most importantly the COVID crash of 2020. A model that has never seen a crisis cannot learn how volatility behaves in one.

Validation is 2023 and 2024, two years, 46,599 stock-days. The validation set has one job only: to tell the neural network when to stop training. It includes the Adani episode and the 2024 election result day, so the stopping rule is tested on turbulent days too.

Testing is 1 January 2025 to 9 April 2026, about 15 months, 314 trading days and 29,774 stock-days. This period was never used for anything: not for training, not for stopping, not for any design choice. We kept it at 15 months for two reasons. First, it is the most recent data, which is what a real user cares about. Second, 314 days is long enough for the statistical tests to have power, while leaving as much history as possible for training. Roughly, the split is 67 percent train, 20 percent validation, 13 percent test.

One technical detail: we assign each observation to a period by the date of the target, meaning the day being predicted, not the day the prediction is made. This stops the last training day from quietly using a validation-day outcome.

Our universe is 95 of the 100 stocks. Five recently listed companies, such as Hyundai and Jio Financial, did not have at least 250 training days, so they were excluded."

## A6. Measuring liquidity

"To test our hypotheses we need to sort stocks by liquidity. We use the Amihud illiquidity ratio, computed on 5-minute bars: the absolute return divided by the rupee value traded, averaged over all bars. It measures how much the price moves per rupee traded. A high number means even small trades move the price, so the stock is illiquid.

Two design choices matter here. First, we compute it using training years only, 2016 to 2022, so the grouping contains no information from the test period. Second, we validated it: its rank correlation with average traded value is minus 0.91, and the ranking makes sense. Reliance, SBI, Infosys and ICICI Bank are the most liquid; CG Power, Solar Industries and Trent are the least.

We then cut the 95 stocks into three equal groups: 32 high liquidity, 31 medium, 32 low.

*[Show the liquidity ranking chart and the descriptive table.]*

The groups really are different. Average annualized volatility is 25 percent for the liquid group, 28.5 percent for the medium group and 33 percent for the illiquid group. And volatility is less persistent in illiquid stocks: the day-to-day autocorrelation of log RV is 0.70 for liquid stocks and 0.65 for the others. Less persistence means harder to forecast."

## A7. Feature selection

"Our features come from two sources.

The first set is the three HAR features: today's RV, the average RV over the last 5 days, and the average over the last 22 days. So a daily, a weekly and a monthly component. These come from the Heterogeneous Market Hypothesis: the market contains day traders, weekly traders and monthly investors, and each reacts to volatility at their own horizon. Three simple averages capture the long memory of volatility with very few numbers. We chose these because they are the proven standard, and because giving the neural network exactly the same information as the benchmark makes the comparison fair. If the network wins, it wins because of how it uses the information, not because it was given more.

The second set comes from an EGARCH model fitted to daily returns: the model's estimate of today's volatility, and the standardized shock, which is today's return divided by that volatility. The shock keeps its sign, so it tells the model whether the news was good or bad.

For the neural networks we take logs of the volatility features and standardize them using training-period means only. Logs matter because raw RV is extremely skewed: a few crisis days are a hundred times larger than a normal day. In logs the distribution is close to a bell curve, which is far easier to learn from."

## A8. The models

"We compare four main models and one extra.

**Model one, HAR-RV.** A linear regression of tomorrow's RV on the daily, weekly and monthly RV, estimated separately for each stock. This is the standard benchmark from Corsi, 2009.

**Model two, log-HAR.** The same regression, but in logs, with a small correction when converting the forecast back. Logs stop a handful of extreme days from distorting the coefficients. We added this because it is the fair benchmark for a neural network that also sees log inputs. The coefficients show why it matters: in logs, the daily, weekly and monthly weights are 0.35, 0.25 and 0.21, adding to about 0.80, which means volatility is highly persistent. In levels they are 0.23, 0.37 and 0.13 and vary far more from stock to stock.

**EGARCH.** This is not a forecasting model in our comparison; it is a feature generator. EGARCH models the log of variance from daily returns, and it has an asymmetry term so that bad news can raise volatility more than good news. That is called the leverage effect. In our data the asymmetry is negative for 92 percent of stocks, exactly as theory predicts, persistence averages 0.92, and the tail parameter is about 4.8, which means fat tails. Critically, we estimate EGARCH on the training years only and then run it forward with fixed parameters, so its test-period features never use future data.

**Model three, HAR-RV-LSTM.** Our main model. An LSTM is a neural network built for sequences. It has a memory cell and three gates, forget, input and output, which decide what to remember and what to discard as it reads a sequence day by day. We feed it the last 22 trading days of the three HAR features.

**Model four, EGARCH-LSTM.** The same network, but fed the two EGARCH features instead. It therefore sees only information from daily closing prices.

**The extra model, HAR-EGARCH-LSTM,** gets both sets of features. It is an ablation: it tests whether EGARCH adds anything on top of HAR."

## A9. Why this network architecture

"Each design choice has a reason.

A sequence length of 22 days matches the monthly horizon of HAR, so the network sees the same history the benchmark uses.

Two LSTM layers with 64 units each: large enough to learn non-linear patterns, small enough not to memorize. We have about 152,000 training sequences and the model has roughly 60,000 parameters, so data comfortably outnumber parameters.

One pooled model for all 95 stocks, instead of 95 separate networks. A single stock has only about 1,700 training days, far too few for a neural network. Pooling gives it 152,000 examples and lets it learn patterns that are common across stocks.

A stock embedding of 8 numbers per stock. Because the model is pooled, it needs to know which stock it is looking at, since Reliance and Adani Green have very different normal volatility levels. The embedding is a small learned fingerprint for each stock.

Dropout of 0.2 to reduce over-fitting.

The output layer is an exponential, so the forecast is always positive. Variance cannot be negative, and our loss function requires a positive forecast.

The loss function is QLIKE, which I will explain in a moment. We train on the same loss we evaluate on.

Training uses the Adam optimizer with a learning rate of 0.001 that halves when progress stalls, batches of 256, and gradient clipping. We use early stopping: after each pass through the data we check the validation loss, and if it has not improved for 8 passes we stop and keep the best version.

Finally, we train every network three times with different random seeds and average the three forecasts. Neural networks give slightly different answers each time you train them, and we wanted to be sure our result is not luck.

*[Show the training curves chart.]*

This chart shows training. The dashed lines are training loss and the solid lines are validation loss, for all three seeds. The HAR-RV-LSTM reaches a validation loss of about 0.226 within a few passes, with its best epochs at 12, 10 and 16. The EGARCH-LSTM flattens out much higher, around 0.27. Already during training we can see that daily-return features carry less information."

## A10. The evaluation metrics

"We use three loss functions. For all of them, lower is better.

**MSE, mean squared error.** The squared gap between forecast and actual. Because of the square, a few extreme days dominate it.

**MAE, mean absolute error.** The plain average gap. It treats all days more evenly.

**QLIKE.** This is our main metric. It is the ratio of actual to forecast, minus the log of that ratio, minus one. It equals zero only when the forecast is perfect. We prefer it for three reasons. First, the actual value we compare against, RV, is itself only an estimate of true volatility, and it has been proven that QLIKE still ranks forecasts correctly in that situation. Second, it depends on the ratio, not the difference, so a calm stock and a wild stock count equally and crisis days do not swamp everything. Third, it punishes under-prediction more than over-prediction, which matches reality: underestimating risk is the costlier mistake.

Then three statistical tools.

**The Diebold-Mariano test** answers: is the difference between two models real, or could it be chance? Each day we average the loss difference across all stocks, then test whether the mean of those 314 daily differences is zero. We use Newey-West standard errors because losses on neighbouring days are correlated. A t-statistic above about 2 means the difference is statistically significant. We also adjust the p-values with the Holm method, because we run several comparisons.

**The block bootstrap** is a second opinion. We resample the test period in blocks of 5 days, 2,000 times, and build a 95 percent confidence interval for the difference. If the interval does not include zero, the result is robust.

**The Mincer-Zarnowitz regression** checks calibration. We regress actual log RV on forecast log RV. A slope near one means the forecast is well calibrated, and the R-squared tells us what share of day-to-day movements the forecast explains."

## A11. Main results

*[Show the test losses bar chart and the main results table.]*

"Now the results, on 29,774 test stock-days.

Test QLIKE is 0.2286 for HAR-RV, 0.1965 for log-HAR, 0.1935 for our HAR-RV-LSTM, and 0.2709 for EGARCH-LSTM.

So the HAR-RV-LSTM is the most accurate model. Its error is 15.4 percent lower than HAR-RV and 1.6 percent lower than log-HAR. It also has the lowest MAE, 1.40 against 1.66 for HAR-RV in units of ten-thousandths, and the lowest MSE.

There is an important piece of analysis here. Log-HAR alone is already 14.0 percent better than HAR-RV. So of the 15.4 points of improvement, about 14 come simply from working in logs, and the neural network itself adds the last 1.6. We think it is important to say this clearly, because many papers compare a neural network only against plain HAR and report a big gain that is mostly the log transformation.

The EGARCH-LSTM is the worst model, 18.5 percent worse than plain HAR-RV. Daily closing prices simply contain less information about volatility than intraday data.

And the combined model, with both HAR and EGARCH inputs, scores 0.1931 against 0.1935. That difference is not significant, with a t-statistic of 0.24. So once you know the intraday volatility history, EGARCH adds nothing."

## A12. Are the differences statistically real?

*[Show the cumulative QLIKE chart and the Diebold-Mariano table.]*

"HAR-RV-LSTM against HAR-RV: t-statistic 6.18, p-value below 0.001.
Against log-HAR: t-statistic 2.80, adjusted p-value 0.010, and the bootstrap interval runs from 0.0011 to 0.0054, which excludes zero.
Against EGARCH-LSTM: t-statistic 9.89.

So all three gains are statistically significant. On MAE the gain over log-HAR is also significant, with a t of 4.22. On MSE it is not, with a p-value of 0.26, and that is expected, because MSE is driven by a few extreme days where no model does well.

This chart shows the cumulative advantage of our model over the test period. A line that rises steadily means the model keeps winning day after day, rather than getting lucky on a few days. Against HAR-RV the line rises all year. Against log-HAR it is flat until about June 2025 and rises after that.

By stock: the LSTM beats HAR-RV on all 95 stocks, and beats log-HAR on 67 of 95, which is 70 percent.

By seed: the three separately trained networks score 0.1926, 0.1938 and 0.1959. Every one of them beats log-HAR at 0.1965, so the result is not a lucky training run."

## A13. Error analysis: bias and calibration

*[Show the forecast versus actual chart and the Mincer-Zarnowitz table.]*

"Next, how are the models wrong?

Plain HAR-RV over-predicts. Its average forecast is 21 percent above actual volatility. The reason is that in levels, a few crisis days in the training data pull the coefficients up, and the model then forecasts too high in calm periods. You can see it in this chart: the blue HAR line sits above the black actual line through the calm second half of 2025.

Log-HAR over-predicts by 8 percent, and the LSTM by 6 percent, the smallest bias. EGARCH-LSTM over-predicts by 39 percent.

In the Mincer-Zarnowitz regressions, the LSTM has the highest R-squared, 0.470, meaning it explains 47 percent of the day-to-day variation in log volatility, against 0.462 for log-HAR, 0.402 for HAR-RV and 0.348 for EGARCH-LSTM. The slopes are 0.93 to 0.95 for the first three models, close to the ideal of one, and 0.79 for EGARCH-LSTM, which means its forecasts swing too much."

## A14. The liquidity result, our main research finding

*[Show the QLIKE by liquidity chart and the per-stock scatter.]*

"Now the core question. In every liquidity group, the HAR-RV-LSTM is the best model. Its QLIKE is 0.1915 for liquid stocks, 0.1898 for medium and 0.1989 for illiquid stocks.

But the interesting part is its gain over log-HAR.
In the most liquid third, the gain is 0.3 percent, with a t-statistic of 0.38. That is not significant; the two models are equally good.
In the middle third, the gain is 1.4 percent, t-statistic 2.43, significant.
In the least liquid third, the gain is 3.0 percent, t-statistic 3.96, strongly significant.

So the advantage of the neural network grows as liquidity falls. This supports hypothesis one. The explanation is that in illiquid stocks volatility is noisier and less persistent, so a fixed linear rule fits worse, and a flexible model that reads the whole 22-day pattern has more to gain.

In this scatter plot each dot is one stock, ordered from most liquid on the left to least liquid on the right. Dots above zero are stocks where the LSTM wins. The dots drift upward to the right, and the rank correlation between the gain and illiquidity is 0.29.

The EGARCH-LSTM does the opposite: it gets worse as liquidity falls, from 0.248 to 0.309."

## A15. Market conditions and robustness

"We ran four more checks.

**Market regimes known in advance.** We classify each day as calm, normal or turbulent using the market's average volatility over the past month, with cut-offs taken from the training period. By those standards the test period had 253 calm days, 61 normal days and no turbulent days. The LSTM gains 1.8 percent over log-HAR on calm days and 0.7 percent on normal days.

**Days sorted by what actually happened.** On the third of days with the highest actual volatility, the LSTM scores 0.292, worse than log-HAR at 0.281, and plain HAR-RV is best at 0.242. This is an honest weakness: on spike days the LSTM reacts a little slower. HAR-RV wins those days only because it over-predicts all the time, so it happens to be closer when volatility jumps. On low and medium volatility days the LSTM is clearly best: 0.191 against 0.204, and 0.097 against 0.104.

**Four sub-periods.** We cut the test period into four equal parts. The LSTM is best in three of them. In the first, January to April 2025, which contains the April tariff shock, log-HAR is marginally ahead, 0.2010 against 0.2018.

**Removing extreme days.** If we delete the 1 percent of days with the highest volatility, the LSTM's gain over log-HAR grows from 1.6 to 2.3 percent. So the result does not depend on a few outliers; if anything, outliers hide it."

## A16. What we corrected along the way

"Our first version of this project showed a bigger win for the neural networks: 19 percent for HAR-RV-LSTM and 6.6 percent for EGARCH-LSTM. When we audited the code we found three problems.

First, the volatility data was dirty. It included overnight gaps and bad prints, which hurt plain HAR the most. After cleaning, HAR-RV's error fell from 0.423 to 0.229.

Second, the liquidity measure had a scaling error, so the groups were close to random. ICICI Bank was classed as medium liquidity and Infosys as low. Only 42 percent of stocks stay in the same group under the corrected measure.

Third, EGARCH had been fitted on the full sample including the test period, which leaks future information. Once fixed, EGARCH-LSTM went from 6.6 percent better than HAR to 18.5 percent worse.

The headline survived: HAR-RV-LSTM is still the best model. But the lesson is that data cleaning and leak-free design changed our results more than the choice of model did."

## A17. Where this can be used

"Four practical uses.

Options. Nearly all these stocks have options on the NSE. The fair price of an at-the-money straddle is roughly 0.8 times volatility times the position size, so a better volatility forecast means better pricing and better hedging.

Risk and margins. Value-at-Risk and futures margins are a multiple of forecast volatility times the position.

Portfolio sizing. In volatility targeting, the size of each position is the risk budget divided by forecast volatility, so that every stock contributes the same risk.

And a practical rule from our liquidity result: for the 30 or so most liquid stocks, the simple log-HAR model is enough. Across a broad portfolio with less liquid names, the LSTM pays for itself."

## A18. Limitations and conclusion

"Four limitations. The test period is one fairly calm stretch with no crisis in it. The stock list is today's NIFTY 100, so companies that dropped out are missing. We forecast intraday volatility only, not the overnight gap. And the model sees only past volatility; adding India VIX, jump measures and market-wide volatility is the natural next step.

To conclude: across 95 Indian stocks and 15 months of unseen data, an LSTM fed with HAR features gives the most accurate volatility forecasts, 15.4 percent better than HAR-RV and 1.6 percent better than log-HAR, both statistically significant. Most of the gain over HAR-RV comes from using logs. The network's own contribution is modest, but it grows to 3 percent in less liquid stocks, which supports our first hypothesis. And models that use only daily prices are clearly worse."

---

# PART B — WALKTHROUGH OF THE STREAMLIT SITE

*[Run `streamlit run app.py` and open the page.]*

## B1. The top of the page

"To make these results understandable without any finance background, we built this simulator. It uses only the test-period forecasts, so everything you see is out-of-sample.

At the top we explain the idea in plain words: every day a stock's price swings by some amount, and that swing is its risk. If you know tomorrow's swing, you can price insurance, decide how much safety money to keep, and decide how much of each stock to hold.

*[Click 'The four prediction methods, in one line each'.]*

We renamed the models. HAR-RV is the Classic formula. Log-HAR is the Improved formula. Our HAR-RV-LSTM is the AI model. And EGARCH-LSTM is AI on daily prices only.

*[Point to 'Pick a stock'. RELIANCE is selected.]*

Here you pick any of the 95 stocks, ordered from most traded to least traded. Reliance is rank 1, with about 1,361 crore rupees traded a day. By default the charts compare the Classic formula with the AI model; this switch shows all four."

## B2. Tab 1 — How accurate is it?

*[Tab 1 is open.]*

"The black line is how much Reliance actually swung each day. The blue and green lines are what the Classic formula and the AI model predicted the evening before. The closer a coloured line hugs the black line, the better.

Below are three numbers. Reliance swings about 1.05 percent a day. The AI model's prediction is off by 0.26 points on a typical day, against 0.28 for the Classic formula, and the AI was the closer of the two on 65 percent of days.

The table underneath shows the average miss for all four methods and whether each one tends to predict too high or too low. This tab is the plain-language version of our accuracy results and our bias analysis."

*[Change the stock to ADANIGREEN to show a less liquid stock, then switch back to RELIANCE.]*

## B3. Tab 2 — Hedging: pricing protection

*[Click tab 2.]*

"This tab turns accuracy into money. Suppose you hold the stock and buy one day of insurance against a big move, which is an option. The fair price is about 0.8 times the expected swing times the amount insured. If the prediction is too high you overpay; if it is too low the seller loses.

The slider sets the amount insured, here 1 crore rupees. The chart adds up, day by day, how many rupees each method mispriced the insurance by. A lower line is better.

For Reliance, over the 15 months, pricing with the AI model instead of the Classic formula would have avoided about 6.43 lakh rupees of mispricing on a 1 crore position. The table splits the error into overcharging and undercharging.

We note on the page that this is a simplified one-day option for teaching; real stock options on the NSE are monthly."

## B4. Tab 3 — Safety money (margin)

*[Click tab 3.]*

"This tab is about margin, the cash you must set aside against a bad day. A sensible rule is a multiple of tomorrow's predicted swing. You choose how cautious to be, here 'Cautious', and the size of the position.

The grey bars are the actual loss each day. The lines are the safety money each method would have kept. Whenever a grey bar pokes above a line, the cushion was not enough that day.

The numbers show the cash locked up by each method and how many days the loss beat the cushion. This tab shows an honest trade-off. The Classic formula predicts too high on average, so it locks up more cash and is breached slightly less often. The AI model frees up cash but sits closer to the edge."

## B5. Tab 4 — Building a portfolio

*[Click tab 4. Default stocks: RELIANCE, HDFCBANK, INFY, TRENT, ADANIGREEN.]*

"This tab shows portfolio construction and adjustment. If you put equal money in every stock, the jumpy stocks end up carrying most of the risk. A smarter rule is to put less money in stocks predicted to be jumpy and more in calm ones, and to adjust every day.

On the left, the bars show what share of the portfolio's total risk each stock carries. Grey is equal money, blue is risk-balanced using the Classic formula, green is risk-balanced using the AI model. The dashed line is the ideal, an equal share.

With five stocks the ideal is 20 percent each. With equal money, Adani Green alone carries 31 percent of the risk. With the AI model's daily adjustments, the largest share falls to 21 percent.

On the right you can see the adjustment happening: how the AI model changes the share of money in each stock over time. When a stock is predicted to become jumpier, its slice shrinks.

You can change the stocks, anywhere from 2 to 10, and the total money invested."

## B6. Tab 5 — Where does AI help most?

*[Click tab 5.]*

"This tab is our main research finding in plain words. We sorted the 95 stocks into three groups: heavily traded, moderately traded and thinly traded.

The bar chart shows how much the AI model reduces the prediction error in each group. Against the Classic formula, it is about 13 to 17 percent in every group. Against the Improved formula, which is a much tougher opponent, the AI adds 0.2 percent for heavily traded stocks, 1.4 percent for moderately traded and 3.0 percent for thinly traded ones.

In the scatter on the right, each dot is one stock, from most traded on the left to least traded on the right. Dots above the line are stocks where the AI wins, and you can hover to see the name.

So the AI earns its keep on the hard-to-predict stocks, and for the giants a good formula is enough."

## B7. The scorecard

*[Scroll down to 'Scorecard: all 95 stocks together'.]*

"Finally, the scorecard averages everything over all 95 stocks.

The AI model is the best method on 66 of the 95 stocks; the Improved formula on 29; the other two on none.
The average miss in the daily swing is 0.383 points for the AI model, against 0.393 for the Improved formula, 0.461 for the Classic formula and 0.544 for AI on daily prices.
Mispricing of protection on 1 crore a day is about 30,590 rupees for the AI model, against 36,749 for the Classic formula.
Safety money on a 1 crore position is 3.74 lakh for the AI model against 4.04 lakh for the Classic formula.
And position risk stays within 5 percent of its target with the AI model, against 9 percent with the Classic formula.

The 'What this means' list summarizes it, including our honest limit: most of the improvement over the Classic formula comes from one mathematical fix, and the AI adds a smaller extra gain on top, mainly in thinly traded stocks. The technical details, with the exact QLIKE numbers, are in the expander at the bottom."

---

## Quick reference: every number used

| Item | Value |
|---|---|
| Raw bars / session bars | 18.8 million / 17.7 million |
| Zero-volume bars ignored / spikes removed | 41,063 / 24 |
| Short days dropped / zero-RV days dropped | 1,455 / 188 |
| Clean stock-days / stocks | 234,226 / 95 |
| Train / validation / test stock-days | 152,935 / 46,599 / 29,774 (314 test days) |
| Liquidity groups | 32 high, 31 medium, 32 low; Spearman with turnover −0.91 |
| Mean volatility by group | 25.1%, 28.5%, 33.1%; AR(1) 0.70, 0.65, 0.65 |
| HAR coefficients (d, w, m) | HAR-RV 0.23, 0.37, 0.13; log-HAR 0.35, 0.25, 0.21 |
| EGARCH | γ < 0 for 92% of stocks; β mean 0.92; ν ≈ 4.8 |
| Network | 22-day input, 2 LSTM layers × 64, embedding 8, dropout 0.2, Adam 0.001, batch 256, patience 8, seeds 11/22/33 |
| Validation QLIKE | HAR-RV 0.2606, log-HAR 0.2298, LSTM 0.2258, EGARCH-LSTM 0.2703 |
| Test QLIKE | HAR-RV 0.2286, log-HAR 0.1965, LSTM 0.1935, EGARCH-LSTM 0.2709, combined 0.1931 |
| Test MAE ×10⁻⁴ | 1.656, 1.432, 1.401, 2.034 |
| Gains | LSTM −15.4% vs HAR-RV, −1.6% vs log-HAR; log-HAR −14.0% vs HAR-RV; EGARCH-LSTM +18.5% |
| DM t-statistics (QLIKE) | vs HAR-RV 6.18; vs log-HAR 2.80 (Holm p 0.010); vs EGARCH-LSTM 9.89; combined 0.24 |
| Bootstrap CI, log-HAR minus LSTM | [0.0011, 0.0054] |
| Seeds, test QLIKE | 0.1926 to 0.1959 |
| Stocks won | 95/95 vs HAR-RV; 67/95 vs log-HAR |
| Forecast ÷ actual | HAR-RV 1.21, log-HAR 1.08, LSTM 1.06, EGARCH-LSTM 1.39 |
| MZ R² | 0.402, 0.462, 0.470, 0.348 |
| Liquidity, LSTM QLIKE | 0.1915, 0.1898, 0.1989 |
| Liquidity, gain vs log-HAR | 0.3% (t 0.38), 1.4% (t 2.43), 3.0% (t 3.96); Spearman 0.29 |
| Regimes | Calm 253 days: 0.1936 vs 0.1971; Normal 61 days: 0.1929 vs 0.1944 |
| High-RV days | HAR-RV 0.242, log-HAR 0.281, LSTM 0.292 |
| Sub-periods (LSTM vs log-HAR) | 0.2018 vs 0.2010; 0.1776 vs 0.1819; 0.1811 vs 0.1891; 0.2136 vs 0.2146 |
| Top 1% removed | LSTM 0.1662 vs log-HAR 0.1701 (2.3%) |
| First version | HAR-RV 0.423, LSTM 0.342, EGARCH-LSTM 0.395; 42% of stocks keep their group |
| Streamlit, Reliance | swing 1.05%; miss 0.26 vs 0.28; AI closer on 65% of days; ₹6.43 lakh mispricing avoided |
| Streamlit, portfolio | Adani Green 31% of risk → 21% (ideal 20%) |
| Streamlit, scorecard | best on 0 / 29 / 66 / 0; miss 0.461 / 0.393 / 0.383 / 0.544; mispricing ₹36,749 / ₹31,332 / ₹30,590 / ₹43,419; safety money ₹4.04 / 3.78 / 3.74 / 4.28 lakh; risk off target 9% / 5% / 5% / 12% |
