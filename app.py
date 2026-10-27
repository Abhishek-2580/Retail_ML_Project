"""RetailPulse dashboard.

Run:  streamlit run app.py
Needs a `data/` folder next to this file (created by pipeline.py) and: pip install plotly
"""
import calendar
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

INDIGO, TEAL, AMBER, ROSE, SLATE = "#4F46E5", "#14B8A6", "#F59E0B", "#F43F5E", "#64748B"
COLORWAY = [INDIGO, TEAL, AMBER, ROSE, "#0EA5E9", "#8B5CF6", SLATE]
SEG_COLORS = {"High-value regulars": INDIGO, "Recently active": TEAL, "Low-spend, cooling": AMBER}
TIER_COLORS = {"Platinum": INDIGO, "Gold": AMBER, "Silver": "#94A3B8"}

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
[data-testid="stSidebar"] {background: #0F172A;}
[data-testid="stSidebar"] * {color: #E2E8F0 !important;}
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,.06);}
div[data-testid="stMetricLabel"] p {font-size: .74rem; color: #64748B; text-transform: uppercase;
    letter-spacing: .05em; font-weight: 600;}
div[data-testid="stMetricValue"] {font-size: 1.65rem; font-weight: 700; color: #0F172A;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 100%); border-radius: 14px;
    padding: 20px 26px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0; font-size: 1.7rem; font-weight: 700; padding: 0;}
.hero p {color: #E0E7FF; margin: 4px 0 0 0; font-size: .95rem;}
.callout {border-left: 4px solid #4F46E5; background: #EEF2FF; padding: 12px 16px;
    border-radius: 8px; color: #1E1B4B; font-size: .92rem; margin: 10px 0 4px 0;}
.callout.warn {border-left-color: #F59E0B; background: #FFFBEB; color: #78350F;}
span[data-baseweb="tag"] {background-color: #4F46E5 !important;}
div[role="slider"] {background-color: #4F46E5 !important;}
button[kind="primary"], .stDownloadButton button {border-radius: 8px;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
@st.cache_data
def load(name, dates=()):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        st.error(f"Missing data/{name}. Run `python pipeline.py` and keep the data folder next to app.py.")
        st.stop()
    return pd.read_csv(path, parse_dates=list(dates))


def money(x):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:,.1f}{suf}"
    return f"{x:,.0f}"


def header(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def callout(text, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def style(fig, height=380, title=None):
    fig.update_layout(template="plotly_white", colorway=COLORWAY, height=height,
                      margin=dict(l=8, r=8, t=48, b=8),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
                      legend=dict(orientation="h", y=1.12, x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=title, font=dict(size=15, color="#0F172A")))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF0F4")
    return fig


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------- sidebar
st.sidebar.markdown("## 📊 RetailPulse")
st.sidebar.caption("Customer analytics & demand forecasting")
page = st.sidebar.radio("Navigate", ["Sales overview", "Demand forecast", "Customer segments",
                                     "Churn watchlist", "Inventory"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Sales data: Jan 2021 - Oct 2025\n\nInventory snapshot: 31 Oct 2025")

# ------------------------------------------------------------------- Sales
if page == "Sales overview":
    header("Sales overview", "Sales, profit and volume by channel, category and city")
    sm = load("sales_monthly.csv", ["month"])
    c1, c2, c3 = st.columns(3)
    cats, chans, cities = (sorted(sm[c].unique()) for c in ["category", "channel", "city"])
    cat = c1.multiselect("Category", cats, default=cats)
    ch = c2.multiselect("Channel", chans, default=chans)
    ci = c3.multiselect("City", cities, default=cities)
    f = sm[sm.category.isin(cat) & sm.channel.isin(ch) & sm.city.isin(ci)]
    if f.empty:
        st.warning("No data for this selection.")
        st.stop()

    sales, profit, units = f.sales.sum(), f.profit.sum(), f.units.sum()
    months = sorted(f["month"].unique())
    delta = None
    if len(months) >= 24:
        cur = f[f.month.isin(months[-12:])].sales.sum()
        prev = f[f.month.isin(months[-24:-12])].sales.sum()
        if prev:
            delta = f"{(cur / prev - 1) * 100:+.1f}% vs prior 12 months"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sales", money(sales), delta=delta)
    k2.metric("Profit", money(profit))
    k3.metric("Profit margin", f"{profit / sales:.1%}" if sales else "n/a")
    k4.metric("Units sold", money(units))

    t1, t2, t3 = st.tabs(["Trend", "Seasonality", "Breakdown"])
    with t1:
        m = f.groupby("month")[["sales", "profit"]].sum().reset_index()
        fig = go.Figure()
        fig.add_scatter(x=m.month, y=m.sales, name="Sales", mode="lines",
                        line=dict(color=INDIGO, width=2.5), fill="tozeroy",
                        fillcolor="rgba(79,70,229,0.12)")
        fig.add_scatter(x=m.month, y=m.profit, name="Profit", mode="lines",
                        line=dict(color=TEAL, width=2.5))
        fig.update_layout(hovermode="x unified")
        show(style(fig, 400, "Monthly sales and profit"))
        callout("Volume roughly halves every September and recovers in December. The pattern repeats "
                "in every year of the data, so it is seasonality, not a data error.")
    with t2:
        h = (f.assign(year=f.month.dt.year, mo=f.month.dt.month)
              .pivot_table(index="year", columns="mo", values="units", aggfunc="sum"))
        h.columns = [calendar.month_abbr[c] for c in h.columns]
        fig = px.imshow(h, aspect="auto", color_continuous_scale="Purples", text_auto=".2s")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(side="top")
        show(style(fig, 330, "Units sold by month and year"))
        callout("Each row is a year. The dark band in July-August and the light September-October "
                "block show the same drop every year.")
    with t3:
        a, b, c = st.columns([1.3, 1, 1])
        cd = f.groupby("category")[["sales", "profit"]].sum().sort_values("sales")
        fig = go.Figure()
        fig.add_bar(y=cd.index, x=cd.sales, name="Sales", orientation="h", marker_color=INDIGO)
        fig.add_bar(y=cd.index, x=cd.profit, name="Profit", orientation="h", marker_color=TEAL)
        fig.update_layout(barmode="group")
        with a:
            show(style(fig, 380, "Sales and profit by category"))
        chd = f.groupby("channel")["sales"].sum().reset_index()
        fig = px.pie(chd, names="channel", values="sales", hole=0.58, color_discrete_sequence=COLORWAY)
        fig.update_traces(textinfo="percent", sort=False)
        with b:
            show(style(fig, 380, "Sales by channel"))
        cyd = f.groupby("city")["sales"].sum().sort_values().reset_index()
        fig = px.bar(cyd, x="sales", y="city", orientation="h", color_discrete_sequence=[INDIGO])
        with c:
            show(style(fig, 380, "Sales by city"))

# ---------------------------------------------------------------- Forecast
elif page == "Demand forecast":
    header("Demand forecast", "Weekly demand for the top 5 products, next 4 weeks")
    hist = load("weekly_history.csv", ["date"])
    fw = load("forecast_weekly.csv", ["ds"])
    fs = load("forecast_summary.csv")
    bt = load("backtest_summary.csv")
    fs["gap"] = (fs["prophet"] - fs["naive_x_own_ratio"]).abs() / fs["blend"]
    fs["confidence"] = np.where(fs.gap < 0.10, "High", np.where(fs.gap < 0.25, "Medium", "Low"))

    t1, t2, t3 = st.tabs(["Product forecast", "Model comparison", "All products"])
    with t1:
        pid = st.selectbox("Product", sorted(fs.product_id.unique()), key="fc_product")
        r = fs[fs.product_id == pid].iloc[0]
        best = bt[bt.product_id == pid].sort_values("WAPE_%").iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected demand (4 weeks)", f"{r['blend']:.0f} units")
        k2.metric("Safe stock level", f"{r['upper']:.0f} units", help="Upper bound of the forecast range")
        k3.metric("Confidence", r["confidence"],
                  help="Share of disagreement between Prophet and the seasonal-ratio estimate: "
                       "under 10% High, under 25% Medium, otherwise Low.")
        k4.metric("Best backtest error", f"{best['WAPE_%']:.1f}% WAPE",
                  help=f"Best model in the backtest: {best['model']}")

        h = hist[hist.product_id == pid].sort_values("date").tail(52)
        f = fw[fw.product_id == pid].sort_values("ds")
        fig = go.Figure()
        fig.add_scatter(x=f.ds, y=f.yhat_upper, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=f.ds, y=f.yhat_lower, line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(20,184,166,0.18)", name="Forecast range", hoverinfo="skip")
        fig.add_scatter(x=h.date, y=h.total_quantity, name="Actual weekly units",
                        line=dict(color=INDIGO, width=2.5))
        fx = [h.date.iloc[-1]] + list(f.ds)
        fy = [h.total_quantity.iloc[-1]] + list(f.yhat)
        fig.add_scatter(x=fx, y=fy, name="Forecast", mode="lines+markers",
                        line=dict(color=TEAL, width=2.5, dash="dash"))
        x0 = h.date.iloc[-1]
        fig.add_shape(type="line", x0=x0, x1=x0, y0=0, y1=1, yref="paper",
                      line=dict(color="#94A3B8", dash="dot"))
        fig.add_annotation(x=x0, y=1, yref="paper", text="Forecast start", showarrow=False,
                           yanchor="bottom", font=dict(color="#64748B", size=11))
        fig.update_layout(hovermode="x unified")
        show(style(fig, 420, f"Product {pid}: last 52 weeks and next 4 weeks"))
        if r["confidence"] == "Low":
            callout("Low confidence: the two forecasting methods disagree noticeably for this product. "
                    "Use the forecast as a rough guide only.", warn=True)
        else:
            callout("Expected demand is the average of the Prophet forecast and a seasonal-ratio estimate "
                    "from the product's own history. Plan stock against the safe level, not the average.")
    with t2:
        pid2 = st.selectbox("Product", sorted(bt.product_id.unique()), key="bt_product")
        b = bt[bt.product_id == pid2].sort_values("WAPE_%", ascending=False)
        colors = [TEAL if v == b["WAPE_%"].min() else "#CBD5E1" for v in b["WAPE_%"]]
        fig = go.Figure(go.Bar(x=b["WAPE_%"], y=b["model"], orientation="h", marker_color=colors,
                               text=[f"{v:.1f}%" for v in b["WAPE_%"]], textposition="outside"))
        fig.update_xaxes(title="WAPE % (lower is better)")
        show(style(fig, 300, f"Backtest error, product {pid2} (13 four-week windows)"))
        avg = bt.groupby("model")["WAPE_%"].mean().sort_values().round(1)
        callout("Average WAPE across the 5 products: " +
                ", ".join(f"<b>{m}</b> {v}%" for m, v in avg.items()) + ".")
    with t3:
        show(style(go.Figure([
            go.Bar(x=fs.product_id.astype(str), y=fs.blend, name="Expected demand", marker_color=INDIGO),
            go.Bar(x=fs.product_id.astype(str), y=fs.upper, name="Safe stock level", marker_color=TEAL)])
            .update_layout(barmode="group"), 340, "Next 4 weeks by product (units)"))
        st.dataframe(fs[["product_id", "prophet", "naive_x_own_ratio", "blend", "upper", "confidence"]],
                     hide_index=True,
                     column_config={"product_id": "Product", "prophet": "Prophet",
                                    "naive_x_own_ratio": "Seasonal ratio", "blend": "Expected",
                                    "upper": "Safe level", "confidence": "Confidence"})

# --------------------------------------------------------------- Customers
elif page == "Customer segments":
    header("Customer segments", "RFM segmentation of registered customers (recency, frequency, spend)")
    cs = load("customer_segments.csv")
    prof = cs.groupby("segment").agg(customers=("customer_id", "size"), recency=("recency", "mean"),
                                     purchases=("frequency", "mean"), spend=("monetary", "mean")).reset_index()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registered customers", f"{len(cs):,}")
    k2.metric("Avg spend per customer", money(cs.monetary.mean()))
    k3.metric("Avg purchases", f"{cs.frequency.mean():.0f}")
    k4.metric("Avg days since last purchase", f"{cs.recency.mean():.0f}")

    a, b = st.columns([1, 1.4])
    fig = px.pie(prof, names="segment", values="customers", hole=0.6, color="segment",
                 color_discrete_map=SEG_COLORS)
    fig.update_traces(textinfo="percent+label", sort=False)
    fig.update_layout(showlegend=False)
    with a:
        show(style(fig, 380, "Customers per segment"))
    sample = cs.sample(min(len(cs), 1500), random_state=1)
    fig = px.scatter(sample, x="frequency", y="monetary", color="segment", opacity=0.6,
                     color_discrete_map=SEG_COLORS, labels={"frequency": "Purchases", "monetary": "Total spend"})
    with b:
        show(style(fig, 380, "Purchases vs spend"))

    st.markdown("##### Segment profile")
    st.dataframe(prof, hide_index=True, column_config={
        "segment": "Segment", "customers": st.column_config.NumberColumn("Customers", format="%d"),
        "recency": st.column_config.NumberColumn("Days since last purchase", format="%.1f"),
        "purchases": st.column_config.NumberColumn("Avg purchases", format="%.1f"),
        "spend": st.column_config.NumberColumn("Avg spend", format="%.0f")})

    mix = (pd.crosstab(cs.segment, cs.loyalty_segment, normalize="index") * 100).reset_index() \
            .melt(id_vars="segment", var_name="Loyalty tier", value_name="share")
    fig = px.bar(mix, x="share", y="segment", color="Loyalty tier", orientation="h", barmode="stack",
                 color_discrete_map=TIER_COLORS, labels={"share": "Share of segment (%)", "segment": ""})
    show(style(fig, 280, "Loyalty tier mix inside each segment"))
    callout("Segments are value tiers cut from a continuous spread (silhouette score 0.29), not natural "
            "groups. The loyalty tier mix is the same in every segment, so the loyalty tier does not "
            "identify the best customers.")
    pick = st.selectbox("Download a segment", sorted(cs.segment.unique()), key="seg_pick")
    st.download_button("Download CSV", cs[cs.segment == pick].to_csv(index=False), file_name=f"{pick}.csv")

# ------------------------------------------------------------------- Churn
elif page == "Churn watchlist":
    header("Churn watchlist", "Customers with no recent purchase, for win-back campaigns")
    cs = load("customer_segments.csv")
    days = st.slider("Flag customers with no purchase for at least (days)", 15, 120, 45, key="churn_days")
    w = cs[cs.recency >= days].sort_values("recency", ascending=False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers flagged", f"{len(w):,}")
    k2.metric("Share of customers", f"{len(w) / len(cs):.1%}")
    k3.metric("Lifetime spend of flagged", money(w.monetary.sum()))

    fig = px.histogram(cs, x="recency", nbins=40, color_discrete_sequence=[INDIGO],
                       labels={"recency": "Days since last purchase"})
    fig.add_shape(type="line", x0=days, x1=days, y0=0, y1=1, yref="paper", line=dict(color=ROSE, dash="dash"))
    fig.add_annotation(x=days, y=1, yref="paper", text=f"{days} days", showarrow=False,
                       yanchor="bottom", font=dict(color=ROSE))
    fig.update_layout(yaxis_title="Customers", bargap=0.05)
    show(style(fig, 320, "Days since last purchase"))
    st.dataframe(w[["customer_id", "recency", "frequency", "monetary", "segment", "loyalty_segment"]],
                 hide_index=True, column_config={
                     "customer_id": "Customer", "recency": "Days since last purchase",
                     "frequency": "Purchases", "monetary": st.column_config.NumberColumn("Spend", format="%.0f"),
                     "segment": "Segment", "loyalty_segment": "Loyalty tier"})
    st.download_button("Download win-back list", w.to_csv(index=False), file_name="winback_list.csv")
    callout("A churn model (logistic regression, random forest) was tested with a 60-day definition and "
            "reached an AUC of about 0.5 in cross-validation, no better than chance. This page is therefore "
            "a simple recency rule, not a prediction. Recency is measured at 31 Oct 2025, a low-season month.",
            warn=True)

# --------------------------------------------------------------- Inventory
else:
    header("Inventory", "Stock value, excess stock and reorder status")
    inv = load("inventory_overstock.csv")
    kw = st.slider("Weeks of demand worth keeping in stock", 4, 52, 26, key="keep_weeks")
    inv["excess_now"] = (inv.stock_on_hand - inv.avg_weekly_units * kw).clip(lower=0) * inv.cost_price
    total, excess = inv.stock_value.sum(), inv.excess_now.sum()
    below = int((inv.stock_on_hand < inv.reorder_point).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stock value (at cost)", money(total))
    k2.metric("Excess stock value", money(excess))
    k3.metric("Excess share", f"{excess / total:.0%}")
    k4.metric("Rows below reorder point", f"{below:,}")

    a, b = st.columns(2)
    cd = inv.groupby("category")["excess_now"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(cd, x="category", y="excess_now", color_discrete_sequence=[AMBER],
                 text=[money(v) for v in cd.excess_now], labels={"excess_now": "Excess value", "category": ""})
    fig.update_traces(textposition="outside")
    with a:
        show(style(fig, 360, "Excess stock value by category"))
    cover = (inv.stock_on_hand / inv.avg_weekly_units.replace(0, np.nan)).dropna()
    if len(cover):
        fig = px.histogram(cover.clip(upper=cover.quantile(0.99)), nbins=40, color_discrete_sequence=[INDIGO],
                           labels={"value": "Weeks of cover"})
        fig.update_layout(showlegend=False, yaxis_title="Store-product rows", xaxis_title="Weeks of cover")
        with b:
            show(style(fig, 360, "Weeks of cover (current stock / average weekly sales)"))

    st.markdown("##### 25 largest excess positions (store x product)")
    top = inv.sort_values("excess_now", ascending=False).head(25)
    st.dataframe(top[["store_id", "product_id", "category", "stock_on_hand", "avg_weekly_units", "excess_now"]],
                 hide_index=True, column_config={
                     "store_id": "Store", "product_id": "Product", "category": "Category",
                     "stock_on_hand": st.column_config.NumberColumn("Stock on hand", format="%d"),
                     "avg_weekly_units": st.column_config.NumberColumn("Avg weekly units", format="%.2f"),
                     "excess_now": st.column_config.NumberColumn("Excess value", format="%.0f")})
    callout("No store-product is below its reorder point, so this data shows overstock, not stockout risk. "
            "Based on one inventory snapshot (31 Oct 2025) and average weekly demand over the last 52 weeks.")
