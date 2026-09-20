"""Shared loaders that reproduce the Task 1 preprocessing exactly (same features, same encodings)."""
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42

def load_superstore():
    df = pd.read_csv("data/superstore.csv", encoding="latin-1")
    df["Target"] = (df["Profit"] > 0).astype(int)          # 1 = Profitable, 0 = Loss-making (as in Task 1)
    df["Year"] = df["Order Date"].str.extract(r"(\d{4})$")[0].astype(int)   # Order Date mixes d-m-Y and m/d/Y; the year is always last
    num = ["Sales", "Quantity", "Discount", "Shipping Cost"]
    cat = ["Segment", "Market", "Region", "Category", "Sub-Category", "Order Priority", "Ship Mode"]
    X = df[num + cat].copy()
    for c in cat:                                            # LabelEncoder on the whole column, as in Task 1
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    meta = df[["Order ID", "Customer ID", "Market", "Segment", "Region", "Year"]].copy()
    return X, df["Target"], meta

def load_retail_raw():
    df = pd.read_csv("data/online_retail_full.csv", encoding="latin-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

def build_rfm(df, snapshot=None, with_country=True):
    snapshot = snapshot or (df["InvoiceDate"].max() + pd.Timedelta(days=1))
    g = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        AvgBasket=("Revenue", "mean"),
        DistinctItems=("StockCode", "nunique"),
        NLines=("Revenue", "size"),
    )
    if with_country:
        g["Country"] = df.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iloc[0])
    return g.reset_index()

def load_rfm_task1():
    raw = load_retail_raw()
    rfm = build_rfm(raw)
    thr = rfm["Monetary"].quantile(0.75)
    rfm["HighValue"] = (rfm["Monetary"] >= thr).astype(int)
    return raw, rfm

FEATS_KNN = ["Recency", "Frequency", "AvgBasket", "DistinctItems"]
