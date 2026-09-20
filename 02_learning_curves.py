"""Q2 - learning curves for both Task 1 models (training-set size vs. cross-validated performance)."""
import json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import *
warnings.filterwarnings("ignore")

fracs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

def curve(est, X, y, scoring):
    n, tr, va = learning_curve(est, X, y, train_sizes=fracs, cv=cv, scoring=scoring, n_jobs=-1, shuffle=True, random_state=SEED)
    return n, tr, va

X, y, _ = load_superstore()
raw, rfm = load_rfm_task1(); Xk, yk = rfm[FEATS_KNN], rfm["HighValue"]
models = {
    "Decision tree (depth 6)": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED, class_weight="balanced"), X, y),
    "Decision tree (unconstrained)": (DecisionTreeClassifier(min_samples_leaf=1, random_state=SEED, class_weight="balanced"), X, y),
    "kNN (k=9)": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9, weights="distance")), Xk, yk),
    "kNN (k=45)": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=45, weights="distance")), Xk, yk),
}
import os
res = json.load(open("results/learning_curves.json")) if os.path.exists("results/learning_curves.json") else {}
for name, (est, XX, yy) in ([] if res else models.items()):
    for sc in ["roc_auc", "f1"]:
        n, tr, va = curve(est, XX, yy, sc)
        res[f"{name}|{sc}"] = dict(n=n.tolist(), train_mean=tr.mean(1).tolist(), train_sd=tr.std(1).tolist(), val_mean=va.mean(1).tolist(), val_sd=va.std(1).tolist())
    print(name, "done", flush=True)
json.dump(res, open("results/learning_curves.json", "w"), indent=1)  # cached; delete to recompute

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
def draw(ax, name, col, label, ls="-", sc="roc_auc", show_train=True):
    r = res[f"{name}|{sc}"]; n = np.array(r["n"])
    if show_train: ax.plot(n, r["train_mean"], ls="--", color=col, marker="o", ms=2.5, lw=1, alpha=.8)
    ax.plot(n, r["val_mean"], ls="-", color=col, marker="o", ms=2.5, lw=1.5, label=label)
    ax.fill_between(n, np.array(r["val_mean"]) - np.array(r["val_sd"]), np.array(r["val_mean"]) + np.array(r["val_sd"]), color=col, alpha=.15, lw=0)
ax = axes[0]
draw(ax, "Decision tree (depth 6)", "#1f4e79", "Depth-6 tree (Task 1)")
draw(ax, "Decision tree (unconstrained)", "#c0504d", "Unconstrained tree")
ax.set_xscale("log"); ax.set_xlabel("Training rows (log scale)"); ax.set_ylabel("ROC-AUC"); ax.set_title("(a) Superstore: decision tree", fontsize=9)
ax.legend(fontsize=7, loc="center right", frameon=False); ax.set_ylim(0.82, 1.01)
ax = axes[1]
draw(ax, "kNN (k=9)", "#1f4e79", "$k=9$ (Task 1)", show_train=False)
draw(ax, "kNN (k=45)", "#2e8b57", "$k=45$ (nested-CV choice)", show_train=False)
ax.set_xscale("log"); ax.set_xlabel("Training customers (log scale)"); ax.set_title("(b) Online Retail: kNN", fontsize=9)
ax.legend(fontsize=7, loc="lower right", frameon=False); ax.set_ylim(0.88, 0.97)
fig.text(0.5, -0.02, "Solid = validation (mean $\\pm$ 1 SD, 5$\\times$3 folds); dashed = training score (omitted for kNN: it is 1.0 by construction).", ha="center", fontsize=7)
plt.tight_layout(); plt.savefig("figs/learning_curves.pdf", bbox_inches="tight"); plt.savefig("figs/learning_curves.png", dpi=200, bbox_inches="tight")
for k in ["Decision tree (depth 6)|roc_auc", "Decision tree (unconstrained)|roc_auc", "kNN (k=9)|roc_auc", "kNN (k=45)|roc_auc"]:
    r = res[k]; print(k); print("  n   :", r["n"]); print("  train:", [round(v, 3) for v in r["train_mean"]]); print("  val  :", [round(v, 3) for v in r["val_mean"]]); print("  valsd:", [round(v, 3) for v in r["val_sd"]])
for k in ["Decision tree (depth 6)|f1", "kNN (k=9)|f1"]:
    r = res[k]; print(k, "val F1:", [round(v, 3) for v in r["val_mean"]])
