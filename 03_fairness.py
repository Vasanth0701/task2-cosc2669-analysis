"""Q3 - group-fairness audit of both Task 1 models with Microsoft Fairlearn (MetricFrame + ThresholdOptimizer).
Sensitive attributes are the only demographic proxies available: Market and Segment (Superstore), Country (Online Retail)."""
import json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate, false_positive_rate, count,
                               demographic_parity_difference, equalized_odds_difference)
from fairlearn.postprocessing import ThresholdOptimizer
from common import *
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
out = {}

def dt(): return DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED, class_weight="balanced")
def knn(): return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9, weights="distance"))

def base_rate(y_true, y_pred): return float(np.mean(y_true))
def audit(y, pred, sf):
    mf = MetricFrame(metrics=dict(n=count, base_rate=base_rate, selection_rate=selection_rate, tpr=true_positive_rate, fpr=false_positive_rate, accuracy=accuracy_score),
                     y_true=y, y_pred=pred, sensitive_features=sf)
    return mf, dict(dp_diff=float(demographic_parity_difference(y, pred, sensitive_features=sf)),
                    eo_diff=float(equalized_odds_difference(y, pred, sensitive_features=sf)),
                    tpr_gap=float(mf.difference()["tpr"]), fpr_gap=float(mf.difference()["fpr"]), sel_gap=float(mf.difference()["selection_rate"]))

def _group_rates(y, pred, sf):
    """Fast numpy group rates (used only for bootstrap; validated against Fairlearn below)."""
    codes, uniq = pd.factorize(sf); k = len(uniq)
    pos = np.bincount(codes, weights=(y == 1), minlength=k); neg = np.bincount(codes, weights=(y == 0), minlength=k)
    tp = np.bincount(codes, weights=((y == 1) & (pred == 1)), minlength=k); fp = np.bincount(codes, weights=((y == 0) & (pred == 1)), minlength=k)
    sel = np.bincount(codes, weights=(pred == 1), minlength=k); n = pos + neg
    ok = (pos > 0) & (neg > 0)
    return sel[ok] / n[ok], tp[ok] / pos[ok], fp[ok] / neg[ok]
def fast_eo(y, p, s): _, t, f = _group_rates(y, p, s); return max(t.max() - t.min(), f.max() - f.min())
def fast_dp(y, p, s): sel, _, _ = _group_rates(y, p, s); return sel.max() - sel.min()
def fast_tprgap(y, p, s): _, t, _ = _group_rates(y, p, s); return t.max() - t.min()
def boot_ci(y, pred, sf, stat, B=1000):
    y = np.asarray(y); pred = np.asarray(pred); sf = np.asarray(sf); n = len(y); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n); vals.append(stat(y[i], pred[i], sf[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
eo, dp, tprgap = fast_eo, fast_dp, fast_tprgap

# ---------------------------------- Model 1 : Superstore -----------------------------------------
X, y, meta = load_superstore()
meta["Market7"] = meta["Market"]
meta["Market"] = meta["Market"].replace({"US": "US+Canada", "Canada": "US+Canada"})   # Canada has only 7 loss-making rows: its error rates are meaningless on their own
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
pred = cross_val_predict(dt(), X, y, cv=skf); proba = cross_val_predict(dt(), X, y, cv=skf, method="predict_proba")[:, 1]
m1 = {"oof_accuracy": float(accuracy_score(y, pred)), "oof_auc": float(roc_auc_score(y, proba))}
assert abs(fast_eo(y.values, pred, meta["Market"].values) - equalized_odds_difference(y, pred, sensitive_features=meta["Market"])) < 1e-9
assert abs(fast_dp(y.values, pred, meta["Market"].values) - demographic_parity_difference(y, pred, sensitive_features=meta["Market"])) < 1e-9
_mf7, _s7 = audit(y, pred, meta["Market7"]); m1["raw_7_markets_artefact"] = dict(eo_diff=_s7["eo_diff"], fpr_gap=_s7["fpr_gap"], canada_n=int((meta["Market7"] == "Canada").sum()), canada_loss_rows=int(((meta["Market7"] == "Canada") & (y == 0)).sum()))
print("7-market artefact:", m1["raw_7_markets_artefact"])
for attr in ["Market", "Segment"]:
    mf, s = audit(y, pred, meta[attr]); s["eo_ci"] = boot_ci(y, pred, meta[attr], eo); s["dp_ci"] = boot_ci(y, pred, meta[attr], dp)
    s["by_group"] = mf.by_group.round(4).reset_index().to_dict(orient="records"); m1[attr] = s
    print(attr, {k: v for k, v in s.items() if k != "by_group"}); print(mf.by_group.round(3))
# does dropping the geographic features remove the disparity?  ("fairness through unawareness")
Xd = X.drop(columns=["Market", "Region"])
pred_d = cross_val_predict(dt(), Xd, y, cv=skf); proba_d = cross_val_predict(dt(), Xd, y, cv=skf, method="predict_proba")[:, 1]
mf_d, s_d = audit(y, pred_d, meta["Market"]); s_d["eo_ci"] = boot_ci(y, pred_d, meta["Market"], eo); s_d["accuracy"] = float(accuracy_score(y, pred_d)); s_d["auc"] = float(roc_auc_score(y, proba_d))
s_d["by_group"] = mf_d.by_group.round(4).reset_index().to_dict(orient="records"); m1["Market_without_geo_features"] = s_d
print("no geo feats", {k: v for k, v in s_d.items() if k != "by_group"})
# how much is Market/Region used at all?  (permutation-free check: importance in the fitted tree)
fit = dt().fit(X, y); m1["market_region_importance"] = float(fit.feature_importances_[list(X.columns).index("Market")] + fit.feature_importances_[list(X.columns).index("Region")])
# per-market discount profile: is the disparity a discount-mix effect?
dsc = pd.DataFrame({"Market": meta["Market"], "Discount": X["Discount"], "y": y}).groupby("Market").agg(mean_discount=("Discount", "mean"), share_discounted=("Discount", lambda s: (s > 0).mean()), loss_rate=("y", lambda s: 1 - s.mean()))
m1["market_discount_profile"] = dsc.round(4).reset_index().to_dict(orient="records"); print(dsc.round(3))

# mitigation: Fairlearn ThresholdOptimizer, fitted on the Task 1 training split and scored on its test split
Xtr, Xte, ytr, yte, mtr, mte = train_test_split(X, y, meta, test_size=0.25, random_state=SEED, stratify=y)
base = dt().fit(Xtr, ytr); p0 = base.predict(Xte)
def row(p, sf):
    return dict(acc=float(accuracy_score(yte, p)), f1=float(f1_score(yte, p)), loss_recall=float(recall_score(yte, p, pos_label=0)),
                eo_diff=float(equalized_odds_difference(yte, p, sensitive_features=sf)), dp_diff=float(demographic_parity_difference(yte, p, sensitive_features=sf)),
                tpr_gap=float(MetricFrame(metrics=true_positive_rate, y_true=yte, y_pred=p, sensitive_features=sf).difference()))
mit = {"before": row(p0, mte["Market"])}
for cons in ["true_positive_rate_parity", "equalized_odds"]:
    to = ThresholdOptimizer(estimator=base, constraints=cons, objective="accuracy_score", predict_method="predict_proba", prefit=True)
    to.fit(Xtr, ytr, sensitive_features=mtr["Market"]); p1 = to.predict(Xte, sensitive_features=mte["Market"], random_state=SEED)
    mit[cons] = row(p1, mte["Market"])
m1["mitigation"] = mit; print(json.dumps(mit, indent=1))
out["model1"] = m1

# ---------------------------------- Model 2 : Online Retail --------------------------------------
raw, rfm = load_rfm_task1(); Xk, yk = rfm[FEATS_KNN], rfm["HighValue"]
grp = np.where(rfm["Country"] == "United Kingdom", "UK", "Non-UK")
skf10 = StratifiedKFold(10, shuffle=True, random_state=SEED)
predk = cross_val_predict(knn(), Xk, yk, cv=skf10); probak = cross_val_predict(knn(), Xk, yk, cv=skf10, method="predict_proba")[:, 1]
m2 = {"oof_accuracy": float(accuracy_score(yk, predk)), "oof_auc": float(roc_auc_score(yk, probak)), "n_uk": int((grp == "UK").sum()), "n_nonuk": int((grp == "Non-UK").sum())}
mf, s = audit(yk, predk, grp)
s["eo_ci"] = boot_ci(yk, predk, grp, eo, B=1000); s["dp_ci"] = boot_ci(yk, predk, grp, dp, B=1000)
s["tpr_gap_ci"] = boot_ci(yk, predk, grp, tprgap, B=1000)
s["by_group"] = mf.by_group.round(4).reset_index().to_dict(orient="records"); m2["UK_vs_NonUK"] = s
print("UK vs non-UK", {k: v for k, v in s.items() if k != "by_group"}); print(mf.by_group.round(3))
top = rfm.assign(pred=predk, y=yk).groupby("Country").agg(n=("y", "size"), base_rate=("y", "mean"), selection_rate=("pred", "mean")).sort_values("n", ascending=False).head(8)
m2["top_countries"] = top.round(3).reset_index().to_dict(orient="records"); print(top.round(3))
# is the disparity explained by different spending behaviour rather than model error?  compare with a label-free baseline
m2["median_monetary_by_group"] = pd.Series(rfm["Monetary"].values).groupby(grp).median().round(1).to_dict()
# mitigation for the kNN: cross-fitted ThresholdOptimizer on the out-of-fold probabilities (thresholds are learnt on 4 folds, applied to the 5th)
from sklearn.base import BaseEstimator, ClassifierMixin
class ProbPassthrough(BaseEstimator, ClassifierMixin):
    """Wraps an already-computed out-of-fold probability as a 'model' so Fairlearn can post-process it."""
    def fit(self, X, y=None): self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X): p = np.asarray(X).reshape(-1); return np.c_[1 - p, p]
    def predict(self, X): return (np.asarray(X).reshape(-1) >= 0.5).astype(int)
Pk = pd.DataFrame({"p": probak}); ykv = yk.values; mit2 = {}
def row2(p):
    mfx = MetricFrame(metrics=dict(tpr=true_positive_rate, fpr=false_positive_rate, sel=selection_rate), y_true=ykv, y_pred=p, sensitive_features=grp)
    return dict(acc=float(accuracy_score(ykv, p)), f1=float(f1_score(ykv, p)), tpr_uk=float(mfx.by_group.loc["UK", "tpr"]), tpr_nonuk=float(mfx.by_group.loc["Non-UK", "tpr"]),
                fpr_uk=float(mfx.by_group.loc["UK", "fpr"]), fpr_nonuk=float(mfx.by_group.loc["Non-UK", "fpr"]), sel_uk=float(mfx.by_group.loc["UK", "sel"]), sel_nonuk=float(mfx.by_group.loc["Non-UK", "sel"]),
                tpr_gap=float(abs(mfx.by_group.loc["UK", "tpr"] - mfx.by_group.loc["Non-UK", "tpr"])))
mit2["before"] = row2(predk)
strat = pd.Series(grp) + "_" + yk.astype(str).values
for cons in ["true_positive_rate_parity", "equalized_odds"]:
    pp = np.zeros(len(yk), dtype=int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Pk, strat):
        to = ThresholdOptimizer(estimator=ProbPassthrough().fit(Pk), constraints=cons, objective="accuracy_score", predict_method="predict_proba", prefit=True)
        to.fit(Pk.iloc[tr], ykv[tr], sensitive_features=grp[tr]); pp[te] = to.predict(Pk.iloc[te], sensitive_features=grp[te], random_state=SEED)
    mit2[cons] = row2(pp)
m2["mitigation"] = mit2; print(json.dumps(mit2, indent=1))
out["model2"] = m2
json.dump(out, open("results/fairness.json", "w"), indent=1, default=float)

# ---------------------------------- Figure ------------------------------------------------------
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
g = pd.DataFrame(m1["Market"]["by_group"]).sort_values("tpr")
x = np.arange(len(g)); w = 0.38
ax[0].bar(x - w/2, g["tpr"], w, color="#1f4e79", label="TPR: profitable order kept")
ax[0].bar(x + w/2, g["fpr"], w, color="#c0504d", label="FPR: loss-maker approved")
ax[0].set_xticks(x); ax[0].set_xticklabels(g["Market"], fontsize=7.5); ax[0].set_ylim(0, 1.22); ax[0].set_title("(a) Superstore tree, by Market (out-of-fold)", fontsize=8.5)
ax[0].legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
gk = pd.DataFrame(m2["UK_vs_NonUK"]["by_group"])
xk = np.arange(len(gk))
ax[1].bar(xk - w, gk["base_rate"], w * 0.66, color="#999999", label="Base rate (true high-value)")
ax[1].bar(xk - w/3 + 0.03, gk["selection_rate"], w * 0.66, color="#2e8b57", label="Selection rate")
ax[1].bar(xk + w/3 + 0.06, gk["tpr"], w * 0.66, color="#1f4e79", label="TPR")
ax[1].bar(xk + w + 0.09, gk["fpr"], w * 0.66, color="#c0504d", label="FPR")
ax[1].set_xticks(xk + 0.02); ax[1].set_xticklabels([f"{a}\n(n={int(n)})" for a, n in zip(gk["Country"] if "Country" in gk else gk.iloc[:, 0], gk["n"])], fontsize=7.5)
ax[1].set_ylim(0, 1.22); ax[1].set_title("(b) Online Retail kNN, UK vs non-UK", fontsize=8.5); ax[1].legend(fontsize=6.2, frameon=False, loc="upper left", ncol=2)
[a.set_yticks([0,0.2,0.4,0.6,0.8,1.0]) for a in ax]
plt.tight_layout(); plt.savefig("figs/fairness.pdf", bbox_inches="tight"); plt.savefig("figs/fairness.png", dpi=200, bbox_inches="tight")
