"""RetailPulse dashboard.

Run:  streamlit run app.py
Needs a `data/` folder next to this file, created by the export cell in the notebook.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RetailPulse", layout="wide")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_data
def load(name, dates=()):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        st.error(f"Missing data/{name}. Run the export cell in the notebook and "
                 "keep the data folder next to app.py.")
        st.stop()
    return pd.read_csv(path, parse_dates=list(dates))


page = st.sidebar.radio("Page", ["Sales overview", "Demand forecast", "Customer segments",
                                 "Churn watchlist", "Inventory"])
st.sidebar.caption("RetailPulse | sales data Jan 2021 - Oct 2025")
st.title(f"RetailPulse: {page}")

# ------------------------------------------------------------------ Sales
if page == "Sales overview":
    sm = load("sales_monthly.csv", ["month"])
    c1, c2, c3 = st.columns(3)
    cats, chans, cities = (sorted(sm[c].unique()) for c in ["category", "channel", "city"])
    cat = c1.multiselect("Category", cats, default=cats)
    ch = c2.multiselect("Channel", chans, default=chans)
    ci = c3.multiselect("City", cities, default=cities)
    f = sm[sm.category.isin(cat) & sm.channel.isin(ch) & sm.city.isin(ci)]

    sales, profit, units = f.sales.sum(), f.profit.sum(), f.units.sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sales", f"{sales:,.0f}")
    k2.metric("Profit", f"{profit:,.0f}")
    k3.metric("Margin", f"{profit / sales:.1%}" if sales else "n/a")
    k4.metric("Units", f"{units:,.0f}")

    st.subheader("Monthly sales and profit")
    st.line_chart(f.groupby("month")[["sales", "profit"]].sum())
    st.caption("Volume halves every September and rises again in December. "
               "It repeats in every year of the data, so it is seasonality, not an error.")
    a, b = st.columns(2)
    a.subheader("Profit by category")
    a.bar_chart(f.groupby("category")["profit"].sum())
    b.subheader("Sales by channel")
    b.bar_chart(f.groupby("channel")["sales"].sum())

# --------------------------------------------------------------- Forecast
elif page == "Demand forecast":
    hist = load("weekly_history.csv", ["date"])
    fw = load("forecast_weekly.csv", ["ds"])
    fs = load("forecast_summary.csv")
    bt = load("backtest_summary.csv")

    pid = st.selectbox("Product", sorted(fs.product_id.unique()))
    h = hist[hist.product_id == pid].sort_values("date").tail(52).set_index("date")["total_quantity"]
    f = fw[fw.product_id == pid].set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
    f.columns = ["forecast", "lower", "upper"]
    st.line_chart(pd.concat([h.rename("actual"), f], axis=1))

    r = fs[fs.product_id == pid].iloc[0]
    gap = abs(r["prophet"] - r["naive_x_own_ratio"]) / r["blend"]
    conf = "High" if gap < 0.10 else "Medium" if gap < 0.25 else "Low"
    k1, k2, k3 = st.columns(3)
    k1.metric("Expected demand, next 4 weeks", f"{r['blend']:.0f} units")
    k2.metric("Safe stock level (upper bound)", f"{r['upper']:.0f} units")
    k3.metric("Confidence", conf, help="Based on how far Prophet and the seasonal-ratio "
              "estimate disagree: under 10% High, under 25% Medium, else Low.")

    st.subheader("Backtest (13 four-week windows)")
    t = bt[bt.product_id == pid][["model", "WAPE_%"]].sort_values("WAPE_%").round(1)
    st.dataframe(t, hide_index=True)
    st.caption("WAPE = total absolute error / total demand. Lower is better. "
               "Expected demand is the average of Prophet and a seasonal-ratio estimate.")
    with st.expander("All products"):
        st.dataframe(fs.round(1), hide_index=True)

# --------------------------------------------------------------- Segments
elif page == "Customer segments":
    cs = load("customer_segments.csv")
    prof = cs.groupby("segment").agg(customers=("customer_id", "size"),
                                     recency_days=("recency", "mean"),
                                     purchases=("frequency", "mean"),
                                     spend=("monetary", "mean")).round(1)
    st.subheader("Segment profile")
    st.dataframe(prof)
    st.bar_chart(prof["customers"])
    st.subheader("Loyalty tier mix inside each segment")
    st.dataframe(pd.crosstab(cs.segment, cs.loyalty_segment, normalize="index").round(2))
    st.caption("Segments are value tiers cut from a continuous spread (silhouette score 0.29), "
               "not natural groups. The loyalty tier mix is the same in every segment, "
               "so the loyalty tier does not identify the best customers.")
    pick = st.selectbox("Download a segment", sorted(cs.segment.unique()))
    st.download_button("Download CSV", cs[cs.segment == pick].to_csv(index=False),
                       file_name=f"{pick}.csv")

# ----------------------------------------------------------------- Churn
elif page == "Churn watchlist":
    cs = load("customer_segments.csv")
    days = st.slider("No purchase for at least (days)", 15, 120, 45)
    w = cs[cs.recency >= days].sort_values("recency", ascending=False)
    k1, k2 = st.columns(2)
    k1.metric("Customers flagged", f"{len(w):,}")
    k2.metric("Share of customers", f"{len(w) / len(cs):.1%}")
    st.dataframe(w[["customer_id", "recency", "frequency", "monetary", "segment",
                    "loyalty_segment"]], hide_index=True)
    st.download_button("Download win-back list", w.to_csv(index=False),
                       file_name="winback_list.csv")
    st.info("A churn model (logistic regression, random forest) was tested with a 60-day "
            "definition and reached AUC of about 0.5 in cross-validation, i.e. no better than "
            "chance. This page is therefore a simple recency rule, not a prediction. "
            "Recency is measured at 31 Oct 2025, a low-season month.")

# ------------------------------------------------------------- Inventory
else:
    inv = load("inventory_overstock.csv")
    kw = st.slider("Weeks of demand worth keeping in stock", 4, 52, 26)
    inv["excess_value_now"] = ((inv.stock_on_hand - inv.avg_weekly_units * kw)
                               .clip(lower=0) * inv.cost_price)
    total, excess = inv.stock_value.sum(), inv.excess_value_now.sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stock value (at cost)", f"{total:,.0f}")
    k2.metric("Excess stock value", f"{excess:,.0f}")
    k3.metric("Excess share", f"{excess / total:.0%}")
    k4.metric("Rows below reorder point", f"{int((inv.stock_on_hand < inv.reorder_point).sum()):,}")

    st.subheader("Excess value by category")
    st.bar_chart(inv.groupby("category")["excess_value_now"].sum().sort_values(ascending=False))
    st.subheader("25 largest excess positions (store x product)")
    st.dataframe(inv.sort_values("excess_value_now", ascending=False).head(25)
                 [["store_id", "product_id", "category", "stock_on_hand",
                   "avg_weekly_units", "excess_value_now"]].round(1), hide_index=True)
    st.caption("No store-product is below its reorder point, so this data shows overstock, "
               "not stockout risk. Based on one inventory snapshot (31 Oct 2025) and "
               "average weekly demand over the last 52 weeks.")
