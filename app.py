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
/* Main page spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: clamp(1rem, 3vw, 2.5rem);
    padding-right: clamp(1rem, 3vw, 2.5rem);
    max-width: 100%;
}

/* Hide Streamlit menu and footer */
#MainMenu, footer {
    visibility: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F172A;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(148, 163, 184, 0.14);
}

/* Responsive headings and text */
h1, h2, h3, h4 {
    overflow-wrap: break-word;
    word-break: normal;
    line-height: 1.4;
}
.stMarkdown p, [data-testid="stCaptionContainer"] p {
    line-height: 1.55;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 14px;
    min-width: 0;
    height: auto;
    box-sizing: border-box;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
}

/* Metric labels */
div[data-testid="stMetricLabel"] {
    min-width: 0;
}

div[data-testid="stMetricLabel"] p {
    font-size: clamp(0.65rem, 1vw, 0.78rem);
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-weight: 600;
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.4;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    min-width: 0;
    font-size: clamp(1rem, 2vw, 1.65rem);
    font-weight: 700;
    color: #0F172A;
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.35;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 14px;
    padding: clamp(16px, 3vw, 26px);
    margin-bottom: 20px;
    box-sizing: border-box;
}

.hero h1 {
    color: #FFFFFF;
    margin: 0;
    font-size: clamp(1.25rem, 3vw, 1.8rem);
    font-weight: 700;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.hero p {
    color: #E0E7FF;
    margin-top: 8px;
    font-size: clamp(0.85rem, 1.5vw, 1rem);
    line-height: 1.5;
}

/* Information callouts */
.callout {
    border-left: 4px solid #4F46E5;
    background: #EEF2FF;
    padding: 12px 16px;
    border-radius: 8px;
    color: #1E1B4B;
    font-size: 0.92rem;
    line-height: 1.6;
    overflow-wrap: anywhere;
    margin: 12px 0;
    box-sizing: border-box;
}

.callout.warn {
    border-left-color: #F59E0B;
    background: #FFFBEB;
    color: #78350F;
}

/* Selectbox tags and sliders */
span[data-baseweb="tag"] {
    background-color: #4F46E5;
}

div[role="slider"] {
    background-color: #4F46E5;
}

/* Buttons */
button[kind="primary"],
.stDownloadButton button {
    border-radius: 8px;
    white-space: normal;
    height: auto;
    min-height: 2.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    white-space: normal;
}

/* Prevent flex/grid children from forcing overflow */
[data-testid="stHorizontalBlock"] > div,
[data-testid="stColumn"] {
    min-width: 0;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    div[data-testid="stMetric"] {
        padding: 10px;
    }

    .hero {
        padding: 16px;
    }
}
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


def style(fig, height=400, title=None):
    """Apply a consistent, responsive layout to every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        height=height,
        autosize=True,
        margin=dict(l=48, r=24, t=88, b=88),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#F4F6F9"),
        title=dict(
            text=title or "",
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=17, color="#EEF1F8"),
            pad=dict(t=8, b=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#EBEEF1"),
            bgcolor="rgba(255,255,255,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, automargin=True, title_standoff=12)
    fig.update_yaxes(gridcolor="#E9EEF5", automargin=True, title_standoff=12, zeroline=False)
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
    cats, chans, cities = (sorted(sm[c].dropna().unique()) for c in ["category", "channel", "city"])
    cat = c1.multiselect("Category", cats, default=cats)
    ch = c2.multiselect("Channel", chans, default=chans)
    ci = c3.multiselect("City", cities, default=cities)
    f = sm[sm.category.isin(cat) & sm.channel.isin(ch) & sm.city.isin(ci)].copy()
    if f.empty:
        st.warning("No data for this selection. Choose at least one category, channel and city.")
        st.stop()

    sales, profit, units = f.sales.sum(), f.profit.sum(), f.units.sum()
    months = sorted(f["month"].dropna().unique())
    delta = None
    if len(months) >= 24:
        cur = f[f.month.isin(months[-12:])].sales.sum()
        prev = f[f.month.isin(months[-24:-12])].sales.sum()
        if prev:
            delta = f"{(cur / prev - 1) * 100:+.1f}% vs prior 12 months"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total sales", money(sales), delta=delta)
    k2.metric("Total profit", money(profit))
    k3.metric("Profit margin", f"{profit / sales:.1%}" if sales else "n/a")
    k4.metric("Units sold", money(units))

    t1, t2, t3 = st.tabs(["Trend", "Seasonality", "Breakdown"])
    with t1:
        m = f.groupby("month", as_index=False)[["sales", "profit"]].sum().sort_values("month")
        fig = go.Figure()
        fig.add_scatter(
            x=m.month, y=m.sales, name="Sales", mode="lines",
            line=dict(color=INDIGO, width=2.5), fill="tozeroy",
            fillcolor="rgba(79,70,229,0.10)",
        )
        fig.add_scatter(
            x=m.month, y=m.profit, name="Profit", mode="lines",
            line=dict(color=TEAL, width=2.5),
        )
        fig.update_layout(hovermode="x unified", legend_title_text="Measure")
        show(style(fig, 430, "Monthly sales and profit"))
        callout("Review the monthly pattern across years before interpreting seasonal changes. "
                "Use the filters above to compare categories, channels and cities.")

    with t2:
        h = (f.assign(year=f.month.dt.year, mo=f.month.dt.month)
             .pivot_table(index="year", columns="mo", values="units", aggfunc="sum"))
        h = h.reindex(columns=range(1, 13))
        h.columns = [calendar.month_abbr[c] for c in h.columns]
        fig = px.imshow(
            h, aspect="auto", color_continuous_scale="Purples", text_auto=".2s",
            labels=dict(x="Month", y="Year", color="Units"),
        )
        fig.update_layout(coloraxis_showscale=True, coloraxis_colorbar=dict(title="Units", thickness=12))
        fig.update_xaxes(side="top")
        show(style(fig, 390, "Units sold by month and year"))
        callout("Each row represents a year. Compare the same months across rows to identify recurring "
                "seasonal patterns.")

    with t3:
        a, b, c = st.columns([1.3, 1, 1])
        cd = f.groupby("category")[["sales", "profit"]].sum().sort_values("sales").reset_index()
        fig = go.Figure()
        fig.add_bar(y=cd.category, x=cd.sales, name="Sales", orientation="h", marker_color=INDIGO)
        fig.add_bar(y=cd.category, x=cd.profit, name="Profit", orientation="h", marker_color=TEAL)
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.12)
        with a:
            show(style(fig, 410, "Sales and profit by category"))

        chd = f.groupby("channel", as_index=False)["sales"].sum()
        fig = px.pie(chd, names="channel", values="sales", hole=0.58,
                     color_discrete_sequence=COLORWAY)
        fig.update_traces(textinfo="percent", sort=False, textposition="inside")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12, x=0))
        with b:
            show(style(fig, 410, "Sales by channel"))

        cyd = f.groupby("city", as_index=False)["sales"].sum().sort_values("sales")
        fig = px.bar(cyd, x="sales", y="city", orientation="h",
                     color_discrete_sequence=[INDIGO], labels={"sales": "Sales", "city": "City"})
        fig.update_traces(hovertemplate="City: %{y}<br>Sales: %{x:,.0f}<extra></extra>")
        with c:
            show(style(fig, 410, "Sales by city"))

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
