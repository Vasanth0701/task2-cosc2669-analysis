"""Q4 - small empirical probes behind the security / privacy discussion:
   (i) membership-inference attack (loss-based; and nearest-neighbour-distance for the kNN)
   (ii) evasion / data-integrity: how easily can a tampered Discount field flip the tree's decision?"""
import json, warnings, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from common import *
warnings.filterwarnings("ignore"); rng = np.random.default_rng(SEED); out = {}

def per_sample_loss(model, X, y):
    p = np.clip(model.predict_proba(X)[np.arange(len(X)), y.astype(int)] if False else model.predict_proba(X), 1e-6, 1 - 1e-6)
    return -np.log(p[np.arange(len(y)), np.asarray(y).astype(int)])

def mia(model, Xtr, ytr, Xte, yte, n=None):
    n = n or min(len(Xte), len(Xtr)); i = rng.choice(len(Xtr), n, replace=False)
    lm = per_sample_loss(model, Xtr.iloc[i] if hasattr(Xtr, "iloc") else Xtr[i], np.asarray(ytr)[i]); ln = per_sample_loss(model, Xte, yte)
    score = -np.r_[lm, ln]; lab = np.r_[np.ones(len(lm)), np.zeros(len(ln))]
    return float(roc_auc_score(lab, score))

# ---------------- Model 1 ----------------
X, y, meta = load_superstore()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED, class_weight="balanced").fit(Xtr, ytr)
out["model1"] = dict(train_acc=float(accuracy_score(ytr, dt.predict(Xtr))), test_acc=float(accuracy_score(yte, dt.predict(Xte))), mia_auc_loss=mia(dt, Xtr, ytr, Xte, yte))
# evasion: rows the tree correctly flags as loss-making; what if the Discount field is tampered with / mis-keyed?
mask = (yte.values == 0) & (dt.predict(Xte) == 0); Xa = Xte[mask].copy(); n0 = len(Xa)
res = {}
for label, newd in [("discount_minus_0.05", (Xa["Discount"] - 0.05).clip(lower=0)), ("discount_minus_0.10", (Xa["Discount"] - 0.10).clip(lower=0)), ("discount_set_to_0", Xa["Discount"] * 0)]:
    Xb = Xa.copy(); Xb["Discount"] = newd; res[label] = float((dt.predict(Xb) == 1).mean())
out["model1"]["evasion_flip_rate_of_correctly_flagged_loss_makers"] = res; out["model1"]["n_correctly_flagged_loss_makers"] = int(n0)
# how many rows sit within 0.05 of a Discount split threshold used by the tree?
thr = np.sort(dt.tree_.threshold[(dt.tree_.feature == list(X.columns).index("Discount"))]); out["model1"]["discount_thresholds_used"] = [round(float(t), 3) for t in np.unique(np.round(thr, 3))]

# ---------------- Model 2 ----------------
raw, rfm = load_rfm_task1(); Xk, yk = rfm[FEATS_KNN], rfm["HighValue"]
Xtr2, Xte2, ytr2, yte2 = train_test_split(Xk, yk, test_size=0.25, random_state=SEED, stratify=yk)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9, weights="distance")).fit(Xtr2, ytr2)
out["model2"] = dict(train_acc=float(accuracy_score(ytr2, knn.predict(Xtr2))), test_acc=float(accuracy_score(yte2, knn.predict(Xte2))), mia_auc_loss=mia(knn, Xtr2, ytr2, Xte2, yte2))
# distance-based attack: the training set IS the model, so an attacker who can query the model with a candidate record and read the neighbour distances (or dump the model) sees a zero distance for members
sc = knn.named_steps["standardscaler"]; nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(Xtr2))
dm = nn.kneighbors(sc.transform(Xtr2.iloc[rng.choice(len(Xtr2), len(Xte2), replace=False)]))[0][:, 0]; dn = nn.kneighbors(sc.transform(Xte2))[0][:, 0]
out["model2"]["mia_auc_nn_distance"] = float(roc_auc_score(np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn]))
out["model2"]["share_members_with_zero_distance"] = float((dm == 0).mean()); out["model2"]["share_nonmembers_with_zero_distance"] = float((dn == 0).mean())
# rarity: how identifiable are individual customers from 4 features?  (share of customers whose feature vector is unique in the file)
out["model2"]["share_unique_feature_vectors"] = float((~Xk.duplicated(keep=False)).mean())
json.dump(out, open("results/privacy_security.json", "w"), indent=1)
print(json.dumps(out, indent=1))
