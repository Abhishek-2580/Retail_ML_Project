"""RetailPulse pipeline: raw CSVs -> clean/merge -> forecast, inventory, segments -> data/*.csv

Usage:
    python pipeline.py                       # raw CSVs in the current folder, output in ./data
    python pipeline.py --raw-dir path/to/raw --out data
    python pipeline.py --skip-backtest       # faster: keeps the existing backtest_summary.csv
    python pipeline.py --mlflow              # also log runs to MLflow (sqlite:///mlflow.db)

Raw files expected: bm_customers.csv, bm_sales.csv, bm_skus.csv, bm_stores.csv, bm_inventory.csv
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

log = logging.getLogger("retailpulse")
TOP_N, HORIZON, N_WINDOWS = 5, 4, 13


def wape(y, yhat):
    return np.abs(y - yhat).sum() / y.sum() * 100


# ------------------------------------------------------------------ ETL
def load_and_clean(raw_dir):
    rd = lambda f: pd.read_csv(os.path.join(raw_dir, f))
    cdf, sdf, pdf, stdf, idf = (rd("bm_customers.csv"), rd("bm_sales.csv"), rd("bm_skus.csv"),
                                rd("bm_stores.csv"), rd("bm_inventory.csv"))
    cdf = cdf.rename(columns={"cust_id": "customer_id", "city": "customer_city"})
    sdf = sdf.rename(columns={"sku_id": "product_id", "unit_price": "sales_price",
                              "discount_pct": "discount_percentage"})
    pdf = pdf.rename(columns={"sku_id": "product_id", "sku_name": "product_name",
                              "unit_price": "list_price"})
    idf = idf.rename(columns={"sku_id": "product_id"})

    cdf["registration_date"] = pd.to_datetime(cdf["registration_date"])
    sdf["date"] = pd.to_datetime(sdf["date"])
    stdf["opening_date"] = pd.to_datetime(stdf["opening_date"])
    idf["last_restock_date"] = pd.to_datetime(idf["last_restock_date"])
    idf["snapshot_date"] = pd.to_datetime(idf["snapshot_date"])

    n0 = len(sdf)
    sdf = sdf.drop_duplicates()
    log.info("sales rows: %d (%d exact duplicates dropped)", len(sdf), n0 - len(sdf))
    log.info("sales without customer_id (guests): %d", sdf["customer_id"].isna().sum())
    return cdf, sdf, pdf, stdf, idf


def build_sales_table(sdf, pdf, cdf, stdf):
    df = (sdf.merge(pdf, on="product_id", how="left")
             .merge(cdf, on="customer_id", how="left")
             .merge(stdf, on="store_id", how="left"))
    assert len(df) == len(sdf), "a join duplicated sales rows (duplicate keys in a lookup table)"
    df["profit"] = df["total_value"] - df["quantity"] * df["cost_price"]
    return df


def sales_monthly(df):
    return (df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
              .groupby(["month", "category", "channel", "city"])
              .agg(units=("quantity", "sum"), sales=("total_value", "sum"),
                   profit=("profit", "sum")).reset_index())


# ------------------------------------------------------------- Forecast
def weekly_demand(sdf):
    daily = (sdf.groupby(["date", "product_id"])["quantity"].sum().rename("total_quantity"))
    idx = pd.MultiIndex.from_product(
        [pd.date_range(sdf["date"].min(), sdf["date"].max(), freq="D"), sdf["product_id"].unique()],
        names=["date", "product_id"])
    daily = daily.reindex(idx, fill_value=0).reset_index()
    weekly = (daily.set_index("date").groupby("product_id")["total_quantity"]
                   .resample("W").sum().reset_index())
    # keep full weeks only (weeks end on Sunday)
    start, end = sdf["date"].min(), sdf["date"].max()
    weekly = weekly[(weekly["date"] - pd.Timedelta(days=6) >= start) & (weekly["date"] <= end)]
    top = (weekly.groupby("product_id")["total_quantity"].sum()
                 .sort_values(ascending=False).head(TOP_N).index.tolist())
    return weekly, top


def _prophet(train):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    d = train.reset_index().rename(columns={"date": "ds", "total_quantity": "y"})
    return Prophet(weekly_seasonality=False, yearly_seasonality=True,
                   daily_seasonality=False).fit(d)


def backtest(weekly, top):
    rows = []
    for pid in top:
        w = weekly[weekly["product_id"] == pid].set_index("date")["total_quantity"]
        for k in range(N_WINDOWS, 0, -1):
            end = len(w) - (k - 1) * HORIZON
            tr, te = w.iloc[:end - HORIZON], w.iloc[end - HORIZON:end]
            preds = {
                "last_week_repeat": np.full(HORIZON, tr.iloc[-1]),
                "same_weeks_last_year": w.shift(52).iloc[end - HORIZON:end].values,
                "prophet": _prophet(tr).predict(pd.DataFrame({"ds": te.index}))["yhat"]
                                       .clip(lower=0).values}
            for name, p in preds.items():
                rows.append({"product_id": pid, "model": name,
                             "abs_err": np.abs(te.values - p).sum(), "actual": te.values.sum()})
    bt = pd.DataFrame(rows)
    out = (bt.groupby(["product_id", "model"])
             .apply(lambda g: g["abs_err"].sum() / g["actual"].sum() * 100, include_groups=False)
             .rename("WAPE_%").reset_index().sort_values(["product_id", "WAPE_%"]))
    return out


def own_seasonal_ratio(w):
    """Average of (next 4 weeks / last 4 weeks) in the same weeks of the previous 4 years."""
    last4 = w.index[-4:]
    next4 = pd.date_range(last4[-1] + pd.Timedelta(weeks=1), periods=4, freq="W")
    ratios = []
    for y in (1, 2, 3, 4):
        shift = pd.Timedelta(weeks=52 * y)
        prev, nxt = w.reindex(last4 - shift).sum(), w.reindex(next4 - shift).sum()
        if prev > 0:
            ratios.append(nxt / prev)
    return float(np.mean(ratios)) if ratios else 1.0


def forecast_next(weekly, top):
    fut, summ = [], []
    for pid in top:
        w = weekly[weekly["product_id"] == pid].set_index("date")["total_quantity"]
        m = _prophet(w)
        fc = m.predict(m.make_future_dataframe(periods=HORIZON, freq="W")).tail(HORIZON)
        fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc[["yhat", "yhat_lower", "yhat_upper"]] = fc[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)
        fc["product_id"] = pid
        fut.append(fc)
        naive = w.iloc[-4:].sum() * own_seasonal_ratio(w)
        summ.append({"product_id": pid, "prophet": fc["yhat"].sum(), "naive_x_own_ratio": naive,
                     "upper": fc["yhat_upper"].sum()})
    summ = pd.DataFrame(summ)
    summ["blend"] = summ[["prophet", "naive_x_own_ratio"]].mean(axis=1)
    summ = summ[["product_id", "prophet", "naive_x_own_ratio", "blend", "upper"]].round(1)
    return pd.concat(fut), summ


# ------------------------------------------------------------ Inventory
def inventory_overstock(idf, sdf, pdf, keep_weeks=26, window=52):
    recent = sdf[sdf["date"] > sdf["date"].max() - pd.Timedelta(weeks=window)]
    wk = (recent.groupby(["store_id", "product_id"])["quantity"].sum() / window) \
            .rename("avg_weekly_units").reset_index()
    chk = (idf.merge(wk, on=["store_id", "product_id"], how="left")
              .fillna({"avg_weekly_units": 0})
              .merge(pdf[["product_id", "category", "cost_price"]], on="product_id", how="left"))
    chk["stock_value"] = chk["stock_on_hand"] * chk["cost_price"]
    chk["excess_units"] = (chk["stock_on_hand"] - chk["avg_weekly_units"] * keep_weeks).clip(lower=0)
    chk["excess_value"] = chk["excess_units"] * chk["cost_price"]
    log.info("below reorder point: %d of %d | excess %.0f%% of stock value",
             (chk["stock_on_hand"] < chk["reorder_point"]).sum(), len(chk),
             chk["excess_value"].sum() / chk["stock_value"].sum() * 100)
    return chk


# --------------------------------------------------------- Segmentation
def segment_customers(sdf, cdf, k=3):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    reg = sdf[sdf["customer_id"].notna()]
    snap = reg["date"].max() + pd.Timedelta(days=1)
    rfm = reg.groupby("customer_id").agg(
        recency=("date", lambda x: (snap - x.max()).days),
        frequency=("date", "size"), monetary=("total_value", "sum")).reset_index()
    X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rfm["cluster"] = km.labels_
    sil = silhouette_score(X, km.labels_)
    # name clusters by their profile (does not depend on label numbering)
    prof = rfm.groupby("cluster")[["recency", "monetary"]].mean()
    high = prof["monetary"].idxmax()
    rest = prof.drop(index=high)
    recent = rest["recency"].idxmin()
    low = [c for c in rest.index if c != recent][0]
    rfm["segment"] = rfm["cluster"].map({high: "High-value regulars",
                                         recent: "Recently active", low: "Low-spend, cooling"})
    rfm_c = rfm.merge(cdf[["customer_id", "loyalty_segment", "age", "gender"]], on="customer_id")
    log.info("silhouette %.3f | segment sizes: %s", sil, rfm["segment"].value_counts().to_dict())
    return rfm_c, sil


# --------------------------------------------------------------- MLflow
def log_mlflow(summary, sil, rfm_c):
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RetailPulse")
    if summary is not None:
        for model, g in summary.groupby("model"):
            with mlflow.start_run(run_name=f"forecast_{model}"):
                mlflow.log_params({"level": "weekly", "horizon_weeks": HORIZON,
                                   "backtest_windows": N_WINDOWS, "n_products": TOP_N})
                mlflow.log_metric("avg_wape", g["WAPE_%"].mean())
    with mlflow.start_run(run_name="segmentation_kmeans_k3"):
        mlflow.log_params({"k": 3, "features": "log RFM, standardized"})
        mlflow.log_metric("silhouette", sil)
        for seg, n in rfm_c["segment"].value_counts().items():
            mlflow.log_metric("size_" + seg.replace(" ", "_").replace(",", ""), int(n))


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=".")
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(a.out, exist_ok=True)
    save = lambda df, name: df.to_csv(os.path.join(a.out, name), index=False)
    t0 = time.time()

    cdf, sdf, pdf, stdf, idf = load_and_clean(a.raw_dir)
    df = build_sales_table(sdf, pdf, cdf, stdf)
    save(sales_monthly(df), "sales_monthly.csv")
    log.info("ETL done, profit margin %.1f%%", df["profit"].sum() / df["total_value"].sum() * 100)

    weekly, top = weekly_demand(sdf)
    save(weekly[weekly["product_id"].isin(top)], "weekly_history.csv")
    summary = None
    if a.skip_backtest:
        log.info("backtest skipped")
    else:
        log.info("running backtest (%d windows x %d products) ...", N_WINDOWS, len(top))
        summary = backtest(weekly, top)
        save(summary, "backtest_summary.csv")
    fut, fsum = forecast_next(weekly, top)
    save(fut, "forecast_weekly.csv")
    save(fsum, "forecast_summary.csv")

    save(inventory_overstock(idf, sdf, pdf), "inventory_overstock.csv")
    rfm_c, sil = segment_customers(sdf, cdf)
    save(rfm_c, "customer_segments.csv")

    if a.mlflow:
        log_mlflow(summary, sil, rfm_c)
    log.info("done in %.0fs, files in %s: %s", time.time() - t0, a.out, sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
