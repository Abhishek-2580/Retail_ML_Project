<<<<<<< HEAD
# RetailPulse: Customer Analytics & Demand Forecasting

Analysis of 641,798 retail sales (Jan 2021 - Oct 2025, 5,000 customers, 200 products, 50 stores
in Dubai, Abu Dhabi and Sharjah) with weekly demand forecasting, inventory review, customer
segmentation, a churn study, an interactive dashboard and experiment tracking.

## Files
| File | Purpose |
|---|---|
| `RETAIL_final_v2.ipynb` | Full analysis with results and conclusions per section |
| `pipeline.py` | Repeatable ETL: raw CSVs -> clean, merge, forecast, inventory, segments -> `data/*.csv` |
| `app.py` | Streamlit dashboard (5 pages) reading `data/` |
| `Dockerfile`, `requirements*.txt` | Container for the dashboard |

## Run it
```
pip install -r requirements-pipeline.txt
python pipeline.py --raw-dir <folder with bm_*.csv> --out data      # add --skip-backtest for a quick run, --mlflow to log runs
pip install -r requirements.txt
streamlit run app.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db          # view logged runs
```
Docker (run the pipeline first so `data/` exists):
```
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```

## Findings
- **Seasonality:** volume halves every September and rises in December, in every year of the data.
- **Forecasting:** daily product demand is ~2-3 units and too noisy (WAPE 70-135%), so forecasting is weekly.
  Over 13 four-week backtest windows Prophet had the lowest error on all 5 top products
  (average WAPE 33.7% vs 37.7% for the best baseline). Product 1111 is low confidence.
- **Inventory:** no stockout risk in this data (0 of 8,735 rows below reorder point); 88% of stock value
  (38.9M at cost) exceeds 26 weeks of demand, mostly Electronics.
- **Segmentation:** customers are very similar (silhouette 0.29), so segments are value tiers, not natural groups.
  Loyalty tier does not match behaviour (Platinum spends the same as Silver).
- **Churn:** with a 60-day definition (9% of customers) logistic regression and random forest reach AUC about 0.5,
  i.e. no usable signal. The dashboard uses a simple recency rule instead.

## Limitations
- Inventory is a single snapshot (31 Oct 2025) and looks generated independently of sales.
- About 25% of sales have no customer ID and are excluded from customer analysis.
- Forecasts cover the top 5 products only; the November forecast was cross-checked against seasonal history.

## Future work
Drift monitoring, CI/CD, Kubernetes deployment, LSTM or gradient-boosting forecasts with promotion data,
and richer customer data (returns, complaints, app activity) for churn.
=======
# Retail_ML_Project
>>>>>>> 5db99ff041beabe9725eccd93df5c85def8ae590
