"""Q1 - Was the Task 1 evaluation unbiased?  Re-evaluates both models under stricter protocols.
Outputs results/eval_*.json.  Positive class = 1 = Profitable (Model 1) / High-Value (Model 2), as in Task 1."""
import json, warnings, numpy as np, pandas as pd
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold,
                                     cross_validate, GridSearchCV)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, recall_score, balanced_accuracy_score, make_scorer)
from common import *
warnings.filterwarnings("ignore")

def metrics(y_true, pred, proba):
    return dict(acc=accuracy_score(y_true, pred), f1=f1_score(y_true, pred), auc=roc_auc_score(y_true, proba),
                bal_acc=balanced_accuracy_score(y_true, pred), neg_recall=recall_score(y_true, pred, pos_label=0))

SCORING = dict(acc="accuracy", f1="f1", auc="roc_auc", bal_acc="balanced_accuracy",
               neg_recall=make_scorer(recall_score, pos_label=0))

def summarise(cv_res):
    return {k[5:]: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in cv_res.items() if k.startswith("test_")}

def dt(): return DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED, class_weight="balanced")
def knn(): return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9, weights="distance"))

out = {}
# =============================== MODEL 1: Decision tree on Superstore ==================================
X, y, meta = load_superstore()
m1 = {}
# (a) Task-1 protocol repeated over 30 random 75/25 splits -> how much does one split move the number?
rows = []
for s in range(30):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s, stratify=y)
    c = dt().fit(Xtr, ytr); rows.append(metrics(yte, c.predict(Xte), c.predict_proba(Xte)[:, 1]))
r = pd.DataFrame(rows)
m1["holdout_30seeds"] = {k: dict(mean=float(r[k].mean()), sd=float(r[k].std()), min=float(r[k].min()), max=float(r[k].max())) for k in r}
# seed 42 exactly as reported in Task 1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
c = dt().fit(Xtr, ytr); m1["holdout_seed42"] = metrics(yte, c.predict(Xte), c.predict_proba(Xte)[:, 1])
# (b) repeated stratified 5-fold CV (5 x 3)
m1["repeated_cv_5x3"] = summarise(cross_validate(dt(), X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED), scoring=SCORING, n_jobs=-1))
# (c) grouped CV: keep every order (or customer) entirely inside one fold
for name, grp in [("group_by_order", meta["Order ID"]), ("group_by_customer", meta["Customer ID"])]:
    m1[name] = summarise(cross_validate(dt(), X, y, groups=grp, cv=GroupKFold(n_splits=5), scoring=SCORING, n_jobs=-1))
m1["n_orders"] = int(meta["Order ID"].nunique()); m1["n_customers"] = int(meta["Customer ID"].nunique())
# share of test rows whose order / customer also appears in the training set under the Task 1 split
tr_idx, te_idx = Xtr.index, Xte.index
m1["test_rows_with_order_in_train"] = float(meta.loc[te_idx, "Order ID"].isin(meta.loc[tr_idx, "Order ID"]).mean())
m1["test_rows_with_customer_in_train"] = float(meta.loc[te_idx, "Customer ID"].isin(meta.loc[tr_idx, "Customer ID"]).mean())
# (d) temporal validation: train on the past, test on the future
tmp = {}
for train_years, test_year in [((2011, 2012), 2013), ((2011, 2012, 2013), 2014)]:
    tr = meta["Year"].isin(train_years); te = meta["Year"] == test_year
    c = dt().fit(X[tr], y[tr]); tmp[f"train{train_years[0]}-{train_years[-1]}_test{test_year}"] = metrics(y[te], c.predict(X[te]), c.predict_proba(X[te])[:, 1]) | dict(base_rate_test=float(y[te].mean()), n_test=int(te.sum()))
m1["temporal"] = tmp
# (e) nested CV: hyper-parameters tuned inside each training fold only
grid = {"max_depth": [3, 4, 6, 8, 12], "min_samples_leaf": [10, 50, 200]}
outer = StratifiedKFold(5, shuffle=True, random_state=SEED); res = []; picked = []
for tr, te in outer.split(X, y):
    gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED, class_weight="balanced"), grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
    res.append(metrics(y.iloc[te], gs.predict(X.iloc[te]), gs.predict_proba(X.iloc[te])[:, 1])); picked.append(gs.best_params_)
r = pd.DataFrame(res); m1["nested_cv"] = {k: (float(r[k].mean()), float(r[k].std(ddof=1))) for k in r}; m1["nested_cv_chosen_params"] = picked
out["model1"] = m1; print("M1 done", flush=True)

# =============================== MODEL 2: kNN on Online Retail =========================================
raw, rfm = load_rfm_task1()
Xk, yk = rfm[FEATS_KNN], rfm["HighValue"]
m2 = {}
# Task-1 protocol exactly: scaler fitted on ALL customers, then a 75/25 split
Xs_all = StandardScaler().fit_transform(Xk)
def t1(seed):
    a, b, c_, d = train_test_split(Xs_all, yk, test_size=0.25, random_state=seed, stratify=yk)
    k = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(a, c_); return metrics(d, k.predict(b), k.predict_proba(b)[:, 1])
m2["holdout_seed42"] = t1(42)
r = pd.DataFrame([t1(s) for s in range(30)]); m2["holdout_30seeds_scaler_before_split"] = {k: dict(mean=float(r[k].mean()), sd=float(r[k].std()), min=float(r[k].min()), max=float(r[k].max())) for k in r}
def t1_clean(seed):   # same, but scaler fitted on training data only
    a, b, c_, d = train_test_split(Xk, yk, test_size=0.25, random_state=seed, stratify=yk)
    k = knn().fit(a, c_); return metrics(d, k.predict(b), k.predict_proba(b)[:, 1])
r = pd.DataFrame([t1_clean(s) for s in range(30)]); m2["holdout_30seeds_scaler_in_pipeline"] = {k: dict(mean=float(r[k].mean()), sd=float(r[k].std()), min=float(r[k].min()), max=float(r[k].max())) for k in r}
m2["repeated_cv_5x10"] = summarise(cross_validate(knn(), Xk, yk, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED), scoring=SCORING, n_jobs=-1))
# Is the label re-expressing its own features?  (Monetary = AvgBasket x number of lines)
m2["corr_log_monetary_vs_log_avgbasket_x_lines"] = float(np.corrcoef(np.log(rfm["Monetary"]), np.log(rfm["AvgBasket"] * rfm["NLines"]))[0, 1])
m2["corr_spearman_distinctitems_vs_monetary"] = float(rfm[["DistinctItems", "Monetary"]].corr(method="spearman").iloc[0, 1])
single = {}
for f in FEATS_KNN:
    a = roc_auc_score(yk, rfm[f]); single[f] = float(max(a, 1 - a))
m2["single_feature_auc"] = single
# nested CV over k and weighting
grid = {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15, 25, 45], "kneighborsclassifier__weights": ["uniform", "distance"]}
res = []; picked = []
for rep in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xk, yk):
        gs = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), grid, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=rep), n_jobs=-1).fit(Xk.iloc[tr], yk.iloc[tr])
        res.append(metrics(yk.iloc[te], gs.predict(Xk.iloc[te]), gs.predict_proba(Xk.iloc[te])[:, 1])); picked.append({k.split("__")[1]: v for k, v in gs.best_params_.items()})
r = pd.DataFrame(res); m2["nested_cv"] = {k: (float(r[k].mean()), float(r[k].std(ddof=1))) for k in r}
m2["nested_cv_chosen_params"] = pd.Series([str(p) for p in picked]).value_counts().to_dict()

# ---- Predictive redesign: features from the past, label from the FUTURE (no shared window) ----
cut = pd.Timestamp("2011-09-01")
past = raw[raw["InvoiceDate"] < cut]; fut = raw[raw["InvoiceDate"] >= cut]
rp = build_rfm(past, snapshot=cut, with_country=False).set_index("CustomerID")
fs = fut.groupby("CustomerID")["Revenue"].sum().reindex(rp.index).fillna(0.0)
thr = fs.quantile(0.75)
lab = (fs >= thr).astype(int)
m2["future_design"] = dict(n_customers=int(len(rp)), cutoff=str(cut.date()), future_window_end=str(raw.InvoiceDate.max().date()),
                           share_zero_future_spend=float((fs == 0).mean()), q75_threshold=float(thr), positive_rate=float(lab.mean()))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
m2["future_design"]["cv_4feat"] = summarise(cross_validate(knn(), rp[FEATS_KNN], lab, cv=cv, scoring=SCORING, n_jobs=-1))
m2["future_design"]["cv_5feat_with_past_monetary"] = summarise(cross_validate(knn(), rp[FEATS_KNN + ["Monetary"]], lab, cv=cv, scoring=SCORING, n_jobs=-1))
out["model2"] = m2
json.dump(out, open("results/eval.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
