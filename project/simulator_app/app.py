"""Volatility Forecast Simulator.

Data: simulator_data.csv, one row per stock and test day (1 Jan 2025 - 9 Apr 2026).
  rv, har, lhar, lstm, eg : daily volatility in basis points (actual, and four model forecasts)
  ret                     : intraday (open-to-close) return in basis points
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Can an ML model predict market risk?", layout="wide")

MODELS = {
    "har": "Classic formula",
    "lhar": "Improved formula",
    "lstm": "ML model",
    "eg": "ML on daily prices only",
}
TECH = {"har": "HAR-RV", "lhar": "log-HAR", "lstm": "HAR-RV-LSTM", "eg": "EGARCH-LSTM"}
COLORS = {"har": "#2a78d6", "lhar": "#eb6834", "lstm": "#1baf7a", "eg": "#b98000", "rv": "#101418"}
GROUPS = {"H": "Heavily traded", "M": "Moderately traded", "L": "Thinly traded"}
GROUP_COLORS = {"H": "#2a78d6", "M": "#eb6834", "L": "#1baf7a"}
K = np.sqrt(2 / np.pi)


# ----------------------------------------------------------------------------- helpers
@st.cache_data
def load_data():
    try:
        folder = Path(__file__).parent
    except NameError:
        folder = Path.cwd()
    df = pd.read_csv(folder / "simulator_data.csv", parse_dates=["Date"])
    for c in ["rv", "har", "lhar", "lstm", "eg", "ret"]:
        df[c] = df[c] / 100
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def inr(x):
    sign = "−" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} lakh"
    return f"{sign}₹{a:,.0f}"


def label(k):
    return f"{MODELS[k]} ({TECH[k]})"


def qlike(rv, f):
    r = (rv ** 2) / (f ** 2)
    return r - np.log(r) - 1


def chart(fig, ytitle, height=400, zero=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=0), yaxis_title=ytitle, plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#e3e6ea", rangemode="tozero" if zero else "normal")
    fig.update_xaxes(showgrid=False)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(t):
    try:
        st.dataframe(t, width="stretch")
    except TypeError:
        st.dataframe(t, use_container_width=True)


def better(new, old):
    """Percent reduction from old to new (positive = new is better)."""
    return 100 * (1 - new / old)


# ----------------------------------------------------------------------------- data
df = load_data()
info = df.groupby("Symbol").agg(Group=("Group", "first"), Rank=("Liquidity_Rank", "first"),
                                Turnover=("Turnover_cr", "first")).sort_values("Rank")
symbols = info.index.tolist()

# ----------------------------------------------------------------------------- intro
st.title("Can an ML model predict how risky a stock will be tomorrow?")
st.write(
    "Every day a stock's price swings by some amount. That swing is its **risk** (traders call it "
    "*volatility*). If you know tomorrow's swing in advance, you can price insurance on the stock, decide "
    "how much safety money to keep, and decide how much of each stock to hold. We trained four prediction "
    "methods on 7 years of data for 95 large Indian stocks, then tested them on 15 months they had never "
    "seen (Jan 2025 to Apr 2026). This page shows what happened."
)
with st.expander("The four prediction methods, in one line each"):
    st.markdown(
        "- **Classic formula (HAR-RV):** averages the stock's swings over the last day, week and month. "
        "The standard method in finance.\n"
        "- **Improved formula (log-HAR):** the same idea with one mathematical fix that stops rare extreme "
        "days from distorting it.\n"
        "- **ML model (HAR-RV-LSTM):** a neural network that reads the last 22 days of swings and learns "
        "patterns across all 95 stocks. **This is our main model.**\n"
        "- **ML model on daily prices only (EGARCH-LSTM):** a neural network that only sees end-of-day prices, "
        "not the swings inside the day."
    )

c1, c2 = st.columns([1.2, 2])
with c1:
    sym = st.selectbox("Pick a stock", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0,
                       format_func=lambda x: f"{x}  ·  {GROUPS[info.loc[x, 'Group']].lower()}")
with c2:
    show_all = st.toggle("Show all four methods (default: Classic formula vs ML model)", value=False)
shown = list(MODELS) if show_all else ["har", "lstm"]
st.caption(f"{sym} is **{GROUPS[info.loc[sym, 'Group']].lower()}**: about ₹{info.loc[sym, 'Turnover']:,} crore "
           f"changes hands a day (rank {info.loc[sym, 'Rank']} of 95, where 1 is the most traded).")

s = df[df["Symbol"] == sym].set_index("Date")
tabs = st.tabs(["1 · How accurate is it?", "2 · Hedging: pricing protection", "3 · Safety money (margin)",
                "4 · Building a portfolio", "5 · Where does Machine Learning help most?"])

# ----------------------------------------------------------------------------- 1 accuracy
with tabs[0]:
    st.subheader(f"Predicted vs actual daily swing for {sym}")
    st.write("The black line is how much the stock actually swung each day. A coloured line is what a method "
             "predicted **the evening before**. The closer a coloured line hugs the black line, the better.")
    smooth = st.radio("Black line", ["Weekly average (easier to read)", "Every single day"], horizontal=True)
    fig = go.Figure()
    actual = s["rv"].rolling(5, min_periods=1).mean() if smooth.startswith("Weekly") else s["rv"]
    fig.add_scatter(x=s.index, y=actual, name="Actual swing", line=dict(color=COLORS["rv"], width=1.6))
    for k in shown:
        fig.add_scatter(x=s.index, y=s[k], name=MODELS[k], line=dict(color=COLORS[k], width=1.8))
    chart(fig, "Daily swing (%)")

    miss = {k: (s[k] - s["rv"]).abs().mean() for k in MODELS}
    closer = ((s["lstm"] - s["rv"]).abs() < (s["har"] - s["rv"]).abs()).mean()
    a, b, c = st.columns(3)
    a.metric("Average actual swing", f"{s['rv'].mean():.2f}% a day")
    b.metric("ML model's average miss", f"{miss['lstm']:.2f} points",
             f"{better(miss['lstm'], miss['har']):.0f}% smaller than Classic formula")
    c.metric("Days the ML model was closer than Classic", f"{100 * closer:.0f}%")
    st.success(f"**In plain words:** {sym} swings about {s['rv'].mean():.2f}% a day. The Classic formula's "
               f"prediction is off by {miss['har']:.2f} points on a typical day; the ML model is off by "
               f"{miss['lstm']:.2f}. The ML model was the closer of the two on {100 * closer:.0f}% of days.")
    table(pd.DataFrame({
        "Average miss (points)": [f"{miss[k]:.3f}" for k in MODELS],
        "Tends to predict": [f"{100 * (s[k].mean() / s['rv'].mean() - 1):+.0f}% vs actual" for k in MODELS],
    }, index=[label(k) for k in MODELS]))

# ----------------------------------------------------------------------------- 2 hedging
with tabs[1]:
    st.subheader("Hedging: what should one day of protection cost?")
    st.write("Suppose you hold this stock and buy **one-day insurance** against a big move (an option "
             "contract). The fair price of that insurance depends on tomorrow's swing: about "
             "**0.8 × expected swing × amount insured**. Predict too high and you overpay; predict too low "
             "and the seller loses money. Either way, a wrong prediction costs someone.")
    amount = st.select_slider("Amount insured each day", [1e6, 1e7, 1e8], value=1e7, format_func=inr)
    gap = {k: K * (s[k] - s["rv"]) / 100 * amount for k in MODELS}
    fig = go.Figure()
    for k in shown:
        fig.add_scatter(x=s.index, y=gap[k].abs().cumsum(), name=MODELS[k], line=dict(color=COLORS[k], width=1.8))
    chart(fig, "Total money mispriced so far (₹)")
    err = {k: gap[k].abs().mean() for k in MODELS}
    fair = K * s["rv"].mean() / 100 * amount
    a, b, c = st.columns(3)
    a.metric("Fair price of one day's protection", inr(fair))
    b.metric("Classic formula: wrong by", f"{inr(err['har'])} a day")
    c.metric("ML model: wrong by", f"{inr(err['lstm'])} a day",
             f"{better(err['lstm'], err['har']):.0f}% less mispricing")
    st.success(f"**In plain words:** over the 15 months, pricing {sym}'s daily protection with the ML model "
               f"instead of the Classic formula would have avoided "
               f"{inr(gap['har'].abs().sum() - gap['lstm'].abs().sum())} of mispricing on a "
               f"{inr(amount)} position.")
    table(pd.DataFrame({
        "Wrong by, per day": [inr(err[k]) for k in MODELS],
        "Total overcharged": [inr(gap[k].clip(lower=0).sum()) for k in MODELS],
        "Total undercharged": [inr((-gap[k]).clip(lower=0).sum()) for k in MODELS],
    }, index=[label(k) for k in MODELS]))
    st.caption("Simplified for teaching: a one-day, at-the-money option with no trading costs. Real NSE stock "
               "options are monthly.")

# ----------------------------------------------------------------------------- 3 margin
with tabs[2]:
    st.subheader("Safety money: how much cash to set aside against a bad day")
    st.write("Brokers and exchanges make you keep **safety money (margin)** against a position. A sensible "
             "rule: safety money = a multiple of tomorrow's predicted swing. Keep too much and your cash "
             "sits idle; keep too little and a bad day wipes it out.")
    a, b = st.columns(2)
    level = a.radio("How cautious?", ["Normal", "Cautious", "Very cautious"], index=1, horizontal=True)
    mult = {"Normal": 1.645, "Cautious": 2.326, "Very cautious": 3.0}[level]
    pos = b.select_slider("Size of the position", [1e7, 1e8], value=1e7, format_func=inr)
    loss = (-s["ret"]).clip(lower=0) / 100 * pos
    held = {k: mult * s[k] / 100 * pos for k in MODELS}
    fig = go.Figure()
    fig.add_bar(x=s.index, y=loss, name="Actual loss that day", marker_color="#9aa1ab")
    for k in shown:
        fig.add_scatter(x=s.index, y=held[k], name=f"Safety money: {MODELS[k]}", line=dict(color=COLORS[k], width=1.8))
    chart(fig, "₹")
    bad = {k: int((loss > held[k]).sum()) for k in MODELS}
    a, b, c = st.columns(3)
    a.metric("Cash locked up, Classic formula", inr(held["har"].mean()))
    b.metric("Cash locked up, ML model", inr(held["lstm"].mean()),
             f"{inr(held['har'].mean() - held['lstm'].mean())} freed", delta_color="normal")
    c.metric("Days the loss beat the safety money", f"Classic {bad['har']} · ML {bad['lstm']}",
             f"out of {len(s)} days", delta_color="off")
    st.info("**In plain words:** grey bars are real losses; a line is the safety money a method would have "
            "kept. A bar poking above the line is a day the cushion was not enough. The honest trade-off: "
            "the Classic formula predicts too high on average, so it locks up more cash and is breached a "
            "little less often. The ML model frees cash but sits closer to the edge.")
    table(pd.DataFrame({
        "Average cash locked up": [inr(held[k].mean()) for k in MODELS],
        "Days cushion was not enough": [f"{bad[k]} of {len(s)}" for k in MODELS],
    }, index=[label(k) for k in MODELS]))

# ----------------------------------------------------------------------------- 4 portfolio
with tabs[3]:
    st.subheader("Building and adjusting a portfolio so no single stock dominates the risk")
    st.write("If you put **equal money** in every stock, the jumpy stocks carry most of the risk. A smarter "
             "rule: put **less money in stocks predicted to be jumpy and more in calm ones**, and adjust "
             "every day as the predictions change. Then each stock carries an equal share of the risk.")
    default = [x for x in ["RELIANCE", "HDFCBANK", "INFY", "TRENT", "ADANIGREEN"] if x in symbols]
    a, b = st.columns([2, 1])
    picks = a.multiselect("Stocks in the portfolio (pick 2 to 10)", symbols, default=default, max_selections=10)
    capital = b.select_slider("Total money invested", [1e6, 1e7, 1e8], value=1e7, format_func=inr)
    if len(picks) < 2:
        st.warning("Pick at least two stocks.")
    else:
        sub = df[df["Symbol"].isin(picks)]
        ret = sub.pivot(index="Date", columns="Symbol", values="ret").dropna()[picks]

        def weights(k):
            f = sub.pivot(index="Date", columns="Symbol", values=k).loc[ret.index, picks]
            inv = 1 / f
            return inv.div(inv.sum(axis=1), axis=0)

        plans = {"Equal money in each": pd.DataFrame(1 / len(picks), index=ret.index, columns=picks),
                 "Risk-balanced, Classic formula": weights("har"),
                 "Risk-balanced, ML model": weights("lstm")}
        rows, shares = [], {}
        for name, w in plans.items():
            pnl = w * ret / 100 * capital
            risk = np.sqrt((pnl ** 2).mean())
            share = risk / risk.sum()
            shares[name] = share
            total = pnl.sum(axis=1)
            rows.append({"Plan": name, "Typical daily swing": inr(np.sqrt((total ** 2).mean())),
                         "Worst single day": inr(total.min()),
                         "Biggest risk share of one stock": f"{100 * share.max():.0f}%",
                         "Ideal share": f"{100 / len(picks):.0f}%"})
        left, right = st.columns(2)
        with left:
            st.markdown("**Share of total risk carried by each stock**")
            fig = go.Figure()
            for name, col in zip(plans, ["#9aa1ab", COLORS["har"], COLORS["lstm"]]):
                fig.add_bar(x=picks, y=100 * shares[name], name=name, marker_color=col)
            fig.add_hline(y=100 / len(picks), line_dash="dash", line_color="#101418")
            fig.update_layout(barmode="group")
            chart(fig, "Share of portfolio risk (%)", height=380)
            st.caption("Dashed line = perfectly equal shares.")
        with right:
            st.markdown("**How the ML model adjusts the money in each stock over time**")
            w = plans["Risk-balanced, ML model"].rolling(5, min_periods=1).mean() * 100
            fig = go.Figure()
            for x in picks:
                fig.add_scatter(x=w.index, y=w[x], name=x, stackgroup="one", line=dict(width=0.5))
            chart(fig, "Share of money (%)", height=380)
            st.caption("When a stock is predicted to get jumpier, its slice shrinks.")
        table(pd.DataFrame(rows).set_index("Plan"))
        eq, ai = shares["Equal money in each"], shares["Risk-balanced, ML model"]
        st.success(f"**In plain words:** with equal money, **{eq.idxmax()}** alone carries "
                   f"{100 * eq.max():.0f}% of the portfolio's risk (ideal: {100 / len(picks):.0f}%). "
                   f"With the ML model's daily adjustments, the biggest share falls to {100 * ai.max():.0f}%.")
        st.caption("Uses each stock's own daily swings during market hours; overnight moves and links between "
                   "stocks are left out to keep the idea simple.")

# ----------------------------------------------------------------------------- 5 liquidity
with tabs[4]:
    st.subheader("Where does the Machine Learning help most?")
    st.write("Some stocks trade in huge volumes all day (Reliance, SBI, Infosys). Others trade thinly, and "
             "their prices are jumpier and harder to predict. We sorted the 95 stocks into three equal groups "
             "by how heavily they trade and measured how much the ML model reduces the prediction error.")

    @st.cache_data
    def per_stock_scores():
        out = pd.DataFrame({k: df.groupby("Symbol").apply(lambda x: qlike(x["rv"], x[k]).mean())
                            for k in MODELS}).join(info)
        out["gain"] = better(out["lstm"], out["lhar"])
        return out

    ps = per_stock_scores()
    grp = ps.groupby("Group")[list(MODELS)].mean().loc[list(GROUPS)]
    g_classic = better(grp["lstm"], grp["har"])
    g_improved = better(grp["lstm"], grp["lhar"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Error reduction from the ML model, by group**")
        fig = go.Figure()
        fig.add_bar(x=list(GROUPS.values()), y=g_classic.values, name="vs Classic formula",
                    marker_color=COLORS["har"], text=[f"{v:.1f}%" for v in g_classic], textposition="outside")
        fig.add_bar(x=list(GROUPS.values()), y=g_improved.values, name="vs Improved formula",
                    marker_color=COLORS["lhar"], text=[f"{v:.1f}%" for v in g_improved], textposition="outside")
        fig.update_layout(barmode="group")
        chart(fig, "Error reduction (%)", height=380)
    with right:
        st.markdown("**Each dot is one stock (ML model vs Improved formula)**")
        fig = go.Figure()
        for g, name in GROUPS.items():
            p = ps[ps["Group"] == g]
            fig.add_scatter(x=p["Rank"], y=p["gain"], mode="markers", name=name, text=p.index,
                            marker=dict(color=GROUP_COLORS[g], size=9, line=dict(color="white", width=1)),
                            hovertemplate="%{text}: %{y:.1f}%<extra></extra>")
        fig.add_hline(y=0, line_color="#5b6470")
        fig.update_layout(xaxis_title="← most traded        least traded →")
        fig.update_layout(hovermode="closest")
        chart(fig, "Error reduction (%)", height=380, zero=False)
    st.success(f"**In plain words:** against the Classic formula the ML model cuts the error by about "
               f"{g_classic.min():.0f} to {g_classic.max():.0f}% in every group. Against the Improved formula, "
               f"which is a much tougher opponent, the ML adds {g_improved['H']:.1f}% for heavily traded "
               f"stocks, {g_improved['M']:.1f}% for moderately traded and {g_improved['L']:.1f}% for thinly "
               f"traded ones. So the ML earns its keep on the hard-to-predict stocks; for the giants a good "
               f"formula is enough.")
    st.caption("Error here is QLIKE, the standard score for risk forecasts (0 = perfect). Dots above the line "
               "are stocks where the ML model wins.")

# ----------------------------------------------------------------------------- scorecard
st.header("Scorecard: all 95 stocks together")


@st.cache_data
def scorecard():
    ps = per_stock_scores()
    wins = ps[list(MODELS)].idxmin(axis=1).value_counts()
    g = df.groupby("Symbol")
    rows = []
    for k in MODELS:
        rows.append({
            "Method": label(k),
            "Best method on": f"{int(wins.get(k, 0))} of 95 stocks",
            "Average miss in daily swing": f"{g.apply(lambda x: (x[k] - x['rv']).abs().mean()).mean():.3f} points",
            "Protection mispriced, per ₹1 crore a day": inr(
                g.apply(lambda x: (K * (x[k] - x["rv"]).abs() / 100 * 1e7).mean()).mean()),
            "Safety money per ₹1 crore (cautious)": inr(
                g.apply(lambda x: (2.326 * x[k] / 100 * 1e7).mean()).mean()),
            "Position risk off target by": f"{100 * g.apply(lambda x: abs(np.sqrt(((x['ret'] / x[k]) ** 2).mean()) - 1)).mean():.0f}%",
        })
    return pd.DataFrame(rows).set_index("Method")


table(scorecard())
st.markdown(
    "**What this means**\n"
    "- **Accuracy:** the ML model is the most accurate method on 66 of the 95 stocks.\n"
    "- **Hedging:** its predictions misprice protection the least.\n"
    "- **Safety money:** it locks up the least cash, at the cost of slightly more close calls.\n"
    "- **Portfolios:** sizing positions with it keeps each stock's risk closest to its target.\n"
    "- **Honest limit:** most of the improvement over the Classic formula comes from one mathematical fix "
    "(the Improved formula). The ML adds a smaller extra gain on top, mainly in thinly traded stocks."
)
with st.expander("Technical details"):
    st.markdown(
        "Risk = realized volatility from 5-minute NSE prices during market hours. Test period: 1 Jan 2025 "
        "to 9 Apr 2026, 29,774 stock-days, never used in training. Pooled test QLIKE: HAR-RV 0.2286, "
        "log-HAR 0.1965, HAR-RV-LSTM 0.1935, EGARCH-LSTM 0.2709. Trading groups use the Amihud illiquidity "
        "measure from 2016 to 2022. Source: Guntur & Panchal (2026), AI Lab project. Illustrative, not "
        "investment advice."
    )
