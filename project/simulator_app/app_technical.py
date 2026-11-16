"""Volatility Forecast Simulator (Streamlit).

Run:  streamlit run app.py

Data: simulator_data.csv, one row per stock and test day (1 Jan 2025 - 9 Apr 2026).
  rv, har, lhar, lstm, eg : daily volatility in basis points (sqrt of realized variance x 10,000)
                            for the realized value and the four model forecasts
  ret                     : intraday (open-to-close) return in basis points
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Volatility Forecast Simulator", layout="wide")

MODELS = {"har": "HAR-RV", "lhar": "log-HAR", "lstm": "HAR-RV-LSTM", "eg": "EGARCH-LSTM"}
COLORS = {"har": "#2a78d6", "lhar": "#eb6834", "lstm": "#1baf7a", "eg": "#b98000", "rv": "#101418"}
GROUPS = {"H": "High liquidity", "M": "Medium liquidity", "L": "Low liquidity"}
GROUP_COLORS = {"H": "#2a78d6", "M": "#eb6834", "L": "#1baf7a"}
ANN = np.sqrt(252) / 100          # daily bp -> annualized %
K = np.sqrt(2 / np.pi)            # ATM straddle premium ~ 0.8 x volatility x notional


# ----------------------------------------------------------------------------- helpers
@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / "simulator_data.csv", parse_dates=["Date"])
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def inr(x):
    sign = "−" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} lakh"
    return f"{sign}₹{a:,.0f}"


def qlike(rv, f):
    r = (rv ** 2) / (f ** 2)      # ratio of variances
    return r - np.log(r) - 1


def stock_metrics(s, k):
    """QLIKE, average miss (annualized vol points) and bias for one stock and model."""
    return {
        "q": qlike(s["rv"], s[k]).mean(),
        "miss": (s["rv"] - s[k]).abs().mean() * ANN,
        "bias": s[k].mean() / s["rv"].mean(),
    }


def straddle(s, k, notional):
    """Daily gap between the quote at forecast vol and the fair price at realized vol."""
    d = K * (s[k] - s["rv"]) / 1e4 * notional
    return {"path": d.abs().cumsum(), "err": d.abs().mean(), "total": d.abs().sum(),
            "over": d.clip(lower=0).sum(), "under": (-d).clip(lower=0).sum()}


def margin(s, k, mult, notional):
    held = mult * s[k] / 1e4 * notional
    breaches = int((-s["ret"] > mult * s[k]).sum())
    return {"held": held, "breaches": breaches, "rate": breaches / len(s), "avg": held.mean()}


def vol_target(s, k, budget):
    pnl = budget * s["ret"] / s[k]
    return {"pnl": pnl, "ratio": np.sqrt((pnl ** 2).mean()) / budget}


def base_layout(fig, ytitle, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title=ytitle,
                      plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#e3e6ea", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return fig


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                      # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_table(table):
    try:
        st.dataframe(table, width="stretch")
    except TypeError:
        st.dataframe(table, use_container_width=True)


def headline(label, value, note):
    st.metric(label, value)
    st.caption(note)


def side_stats(rows):
    """rows: list of (model name, headline, detail)."""
    for name, head, detail in rows:
        st.markdown(f"**{name}**  \n{head}  \n<span style='color:#5b6470;font-size:0.85em'>{detail}</span>",
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
df = load_data()
info = df.groupby("Symbol").agg(Group=("Group", "first"), Rank=("Liquidity_Rank", "first"),
                                Turnover=("Turnover_cr", "first")).sort_values("Rank")
symbols = info.index.tolist()

per_stock = pd.DataFrame({k: df.groupby("Symbol").apply(lambda s: qlike(s["rv"], s[k]).mean())
                          for k in MODELS}).join(info)
per_stock["gain"] = 100 * (1 - per_stock["lstm"] / per_stock["lhar"])

# ----------------------------------------------------------------------------- header
st.title("What a better volatility forecast is worth on a trading desk")
st.write("95 NIFTY 100 stocks, out-of-sample days from 1 Jan 2025 to 9 Apr 2026. Every model forecast "
         "tomorrow's volatility using only past data. Pick a stock and see what each forecast would have "
         "meant for pricing options, setting margins and sizing positions.")

c1, c2 = st.columns([1, 2])
with c1:
    default = symbols.index("RELIANCE") if "RELIANCE" in symbols else 0
    sym = st.selectbox("Stock (ordered from most to least liquid)", symbols, index=default,
                       format_func=lambda x: f"{x}  ·  {GROUPS[info.loc[x, 'Group']]}")
with c2:
    chosen = st.multiselect("Models shown in the charts", list(MODELS), default=["har", "lhar", "lstm"],
                            format_func=MODELS.get)
st.caption(f"{sym}: {GROUPS[info.loc[sym, 'Group']]} · liquidity rank {info.loc[sym, 'Rank']} of 95 · "
           f"average turnover ₹{info.loc[sym, 'Turnover']:,} cr/day")

s = df[df["Symbol"] == sym].set_index("Date")
tabs = st.tabs(["1 · Forecast vs reality", "2 · Options desk", "3 · Risk & futures margin",
                "4 · Portfolio sizing", "5 · Does liquidity matter?"])

# ----------------------------------------------------------------------------- 1 forecast
with tabs[0]:
    st.subheader("Tomorrow's volatility: forecast and what happened")
    left, right = st.columns([2.2, 1])
    with left:
        window = st.selectbox("Smooth realized line", [1, 5, 22], index=1,
                              format_func=lambda w: "Daily" if w == 1 else f"{w}-day average")
        fig = go.Figure()
        fig.add_scatter(x=s.index, y=(s["rv"] * ANN).rolling(window, min_periods=1).mean(),
                        name="Realized", line=dict(color=COLORS["rv"], width=1.6))
        for k in chosen:
            fig.add_scatter(x=s.index, y=s[k] * ANN, name=MODELS[k], line=dict(color=COLORS[k], width=1.8))
        show_chart(base_layout(fig, "Annualized volatility (%)"))
    with right:
        ms = {k: stock_metrics(s, k) for k in MODELS}
        best = min(ms, key=lambda k: ms[k]["q"])
        st.metric(f"Best model for {sym}", MODELS[best], f"QLIKE {ms[best]['q']:.3f}")
        side_stats([(MODELS[k], f"QLIKE {m['q']:.3f}",
                     f"average miss {m['miss']:.1f} vol pts · forecasts {100 * (m['bias'] - 1):+.1f}% vs realized")
                    for k, m in ms.items()])
    st.info("**How to read it.** A good forecast tracks the black line without lagging behind spikes or "
            "sitting above it in calm weeks. QLIKE is the standard loss for volatility forecasts (0 = perfect).")

# ----------------------------------------------------------------------------- 2 options
with tabs[1]:
    st.subheader("Quoting a 1-day ATM straddle with each model's volatility")
    left, right = st.columns([2.2, 1])
    with left:
        notional = st.selectbox("Notional per day", [1e6, 1e7, 1e8], index=1, format_func=inr, key="opt_n")
        fig = go.Figure()
        for k in chosen:
            fig.add_scatter(x=s.index, y=straddle(s, k, notional)["path"], name=MODELS[k],
                            line=dict(color=COLORS[k], width=1.8))
        show_chart(base_layout(fig, "Money mispriced so far (₹)"))
    with right:
        rs = {k: straddle(s, k, notional) for k in MODELS}
        headline("Average pricing error per day (HAR-RV-LSTM)", inr(rs["lstm"]["err"]),
                  f"HAR-RV {inr(rs['har']['err'])}")
        side_stats([(MODELS[k], f"{inr(r['total'])} mispriced · {inr(r['err'])}/day",
                     f"{inr(r['over'])} overcharged (lost trades) · {inr(r['under'])} undercharged (given away)")
                    for k, r in rs.items()])
    st.info("**Why it matters.** A straddle pays the size of the move, so its fair price is about "
            "0.8 × volatility × notional. A forecast that is too low gives premium away; one that is too high "
            "loses trades. A frictionless 1-day option is a teaching simplification: real NSE stock options "
            "are monthly and trade around implied volatility.")

# ----------------------------------------------------------------------------- 3 margin
with tabs[2]:
    st.subheader("How much margin to hold on a futures position")
    left, right = st.columns([2.2, 1])
    with left:
        a, b, c = st.columns(3)
        mult = a.selectbox("Multiplier", [1.645, 2.326, 3.0], index=1,
                           format_func=lambda m: {1.645: "1.65 (95%)", 2.326: "2.33 (99%)", 3.0: "3.00"}[m])
        pos = b.selectbox("Position", [1e7, 1e8], format_func=inr, key="mar_n")
        mk = c.selectbox("Model in chart", list(MODELS), index=2, format_func=MODELS.get)
        m = margin(s, mk, mult, pos)
        fig = go.Figure()
        fig.add_bar(x=s.index, y=(-s["ret"]).clip(lower=0) / 1e4 * pos, name="Daily loss",
                    marker_color="#9aa1ab")
        fig.add_scatter(x=s.index, y=m["held"], name=f"{MODELS[mk]} margin",
                        line=dict(color=COLORS[mk], width=1.8))
        show_chart(base_layout(fig, "₹"))
    with right:
        target = {1.645: 5, 2.326: 1, 3.0: 0.13}[mult]
        headline(f"Breaches, {MODELS[mk]}", f"{m['breaches']} of {len(s)} days",
                  f"{100 * m['rate']:.2f}% vs target ≈ {target}%")
        side_stats([(MODELS[k], f"{100 * r['rate']:.2f}% breaches", f"average margin held {inr(r['avg'])}")
                    for k, r in ((k, margin(s, k, mult, pos)) for k in MODELS)])
    st.info("**Trade-off.** Margin = multiplier × forecast volatility × position; a breach is a day when the "
            "intraday loss on a long position exceeded it. All models breach more than a normal distribution "
            "implies, because daily moves have fat tails. A forecast that runs high ties up more capital; one "
            "that runs low breaches more often.")

# ----------------------------------------------------------------------------- 4 sizing
with tabs[3]:
    st.subheader("Volatility targeting: keep each position's daily risk at a budget")
    left, right = st.columns([2.2, 1])
    with left:
        budget = st.selectbox("Daily risk budget", [1e5, 5e5], format_func=inr)
        fig = go.Figure()
        for k in chosen:
            pnl = vol_target(s, k, budget)["pnl"]
            fig.add_scatter(x=s.index, y=np.sqrt((pnl ** 2).rolling(20).mean()), name=MODELS[k],
                            line=dict(color=COLORS[k], width=1.8))
        fig.add_hline(y=budget, line_dash="dash", line_color="#101418")
        show_chart(base_layout(fig, "Realized daily risk, past 20 days (₹)"))
    with right:
        rs = {k: vol_target(s, k, budget) for k in MODELS}
        headline("Realized risk vs budget (HAR-RV-LSTM)", f"{100 * rs['lstm']['ratio']:.0f}% of budget",
                  "100% = exactly on budget")
        side_stats([(MODELS[k], f"{100 * r['ratio']:.0f}% of budget",
                     f"realized {inr(r['ratio'] * budget)} a day vs {inr(budget)}") for k, r in rs.items()])
    st.info("**Why it matters.** Position size = risk budget ÷ forecast volatility. If the forecast is too "
            "high, risk runs below budget and capital sits idle; if too low, risk overshoots. A stable line "
            "near the dashed budget is the goal.")

# ----------------------------------------------------------------------------- 5 liquidity
with tabs[4]:
    st.subheader("Where the deep-learning model earns its keep")
    left, right = st.columns([2.2, 1])
    with left:
        fig = go.Figure()
        for g, name in GROUPS.items():
            p = per_stock[per_stock["Group"] == g]
            fig.add_scatter(x=p["Rank"], y=p["gain"], mode="markers", name=name, text=p.index,
                            marker=dict(color=GROUP_COLORS[g], size=9, line=dict(color="white", width=1)),
                            hovertemplate="%{text}: %{y:.1f}%<extra></extra>")
        cur = per_stock.loc[sym]
        fig.add_scatter(x=[cur["Rank"]], y=[cur["gain"]], mode="markers+text", text=[sym],
                        textposition="top center", showlegend=False,
                        textfont=dict(color="#101418", size=13),
                        marker=dict(color="#101418", size=13, symbol="circle-open", line=dict(width=2)))
        fig.add_hline(y=0, line_color="#5b6470")
        fig = base_layout(fig, "QLIKE gain of HAR-RV-LSTM over log-HAR (%)")
        fig.update_layout(hovermode="closest", xaxis_title="Liquidity rank (1 = most liquid)")
        fig.update_yaxes(rangemode="normal")
        show_chart(fig)
    with right:
        grp = per_stock.groupby("Group")[list(MODELS)].mean().loc[list(GROUPS)]
        gain_lh = 100 * (1 - grp["lstm"] / grp["lhar"])
        gain_h = 100 * (1 - grp["lstm"] / grp["har"])
        headline("Gain over log-HAR, least liquid third", f"{gain_lh['L']:.1f}%",
                  f"vs {gain_lh['H']:.1f}% in the most liquid third")
        show_table(pd.DataFrame({"vs log-HAR": [f"{v:.1f}%" for v in gain_lh],
                                   "vs HAR-RV": [f"{v:.1f}%" for v in gain_h]},
                                  index=list(GROUPS.values())))
    st.info("**Reading.** In the most liquid stocks (Reliance, SBI, Infosys…) the simple log-HAR model is "
            "already as good. As liquidity falls, realized volatility gets noisier and the LSTM's gain grows.")

# ----------------------------------------------------------------------------- all stocks
st.header("All 95 stocks, test period")
st.caption("Averages over the 95 stocks. Straddle on ₹1 crore a day; margin at 2.33× forecast on a ₹1 crore "
           "long future; risk off budget = average gap between realized and budgeted daily risk.")


@st.cache_data
def summary_table():
    best = per_stock[list(MODELS)].idxmin(axis=1).value_counts()
    rows = []
    for k, name in MODELS.items():
        g = df.groupby("Symbol")
        rows.append({
            "Model": name,
            "QLIKE": round(per_stock[k].mean(), 4),
            "Best on (stocks)": int(best.get(k, 0)),
            "Straddle pricing error / day": inr(g.apply(lambda x: straddle(x, k, 1e7)["err"]).mean()),
            "99% margin breaches": f"{100 * g.apply(lambda x: margin(x, k, 2.326, 1e7)['rate']).mean():.2f}%",
            "Margin held": inr(g.apply(lambda x: margin(x, k, 2.326, 1e7)["avg"]).mean()),
            "Risk off budget": f"{100 * g.apply(lambda x: abs(vol_target(x, k, 1e5)['ratio'] - 1)).mean():.0f}%",
        })
    return pd.DataFrame(rows).set_index("Model")


show_table(summary_table())
st.caption("Source: Guntur & Panchal (2026), AI Lab project. 5-minute NSE data; forecasts are the "
           "out-of-sample test set (29,774 stock-days). Intraday (open-to-close) volatility and returns; "
           "overnight gaps excluded. Illustrative, not investment advice.")
