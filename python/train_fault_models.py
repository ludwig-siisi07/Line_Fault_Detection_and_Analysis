"""
train_fault_models.py
Trains and evaluates the fault detection, classification, location models.
Usage:  python train_fault_models.py fault_dataset_train.csv [fault_dataset_ood.csv]
Outputs: fault_ml_results.pdf, fault_classifier.joblib, fault_locator.joblib
"""
import sys, json
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import beta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, mean_absolute_error,
                             mean_squared_error, r2_score)

CSV = sys.argv[1] if len(sys.argv) > 1 else "fault_dataset_train.csv"
OOD = sys.argv[2] if len(sys.argv) > 2 else None   
SEED = 42
NAMES = ["NF", "LG", "LL", "LLG", "LLL"]

df = pd.read_csv(CSV)

LABEL_COLS = ["class", "pA", "pB", "pC", "pG", "d_km", "Rf_ohm", "tf_s", "class_name", "triggered", "trig_delay_ms"]
FEATS = [c for c in df.columns if c not in LABEL_COLS]

def trigger_stats(D):
    """Detection by the current-RMS trigger (v2 data only)."""
    if "triggered" not in D.columns: return {}
    f, nf = D[D["class"] > 0], D[D["class"] == 0]
    return {"trig_missed_faults": int((f["triggered"] == 0).sum()), "n_faults": int(len(f)),
            "trig_false_alarms": int((nf["triggered"] == 1).sum()), "n_normal": int(len(nf)),
            "trig_delay_median_ms": float(f["trig_delay_ms"].median()), "trig_delay_max_ms": float(f["trig_delay_ms"].max())}

def zero_error_bound(n):
    """95% upper bound on the error rate after 0 errors in n trials (Clopper-Pearson)."""
    return 1 - 0.05 ** (1.0 / n)

TRIG = trigger_stats(df)
df = df.dropna(subset=FEATS).reset_index(drop=True)   
if TRIG: print("Trigger:", TRIG)
def eng(D):
    """Extra features for fault location: aparent impedance per phase and simple extremes."""
    D = D.copy()
    V = D[["Va", "Vb", "Vc"]].values; I = D[["Ia", "Ib", "Ic"]].values
    for k, p in enumerate("abc"): D["Z" + p] = V[:, k] / I[:, k]
    Z = D[["Za", "Zb", "Zc"]].values
    D["Zmin"], D["Zmax"] = Z.min(1), Z.max(1)
    D["Imax"], D["Vmin"] = I.max(1), V.min(1)
    D["Zmin_log"], D["V1_I1_log"] = np.log(D["Zmin"]), np.log(D["V1_I1"])
    D["Vmin_Imax"] = D["Vmin"] / D["Imax"]
    return D
LOC_FEATS = [c for c in eng(df.head(5)).columns if c not in LABEL_COLS]
X, y = df[FEATS], df["class"].astype(int)
print(f"{len(df)} rows, {len(FEATS)} features, classes: {y.value_counts().sort_index().to_dict()}")

# ---------------- 1. Classification (random split) ----------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
tree = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
res = {"tree_acc": accuracy_score(yte, tree.predict(Xte)),
       "rf_acc": accuracy_score(yte, rf.predict(Xte)),
       "rf_macro_f1": f1_score(yte, rf.predict(Xte), average="macro")}
cm_random = confusion_matrix(yte, rf.predict(Xte), labels=range(5))
cv = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
                     X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
res["rf_cv_mean"], res["rf_cv_std"] = cv.mean(), cv.std()
# detection only: fault vs no fault
det_true = (yte > 0).astype(int); det_pred = (rf.predict(Xte) > 0).astype(int)
res["detection_acc"] = accuracy_score(det_true, det_pred)
res["missed_faults"] = int(((det_true == 1) & (det_pred == 0)).sum())
res["false_alarms"] = int(((det_true == 0) & (det_pred == 1)).sum())

# ---------------- 2. Harder test: unseen fault resistance ----------------
rng = np.random.RandomState(SEED)
hard_test = ((df["class"] > 0) & (df["Rf_ohm"] > 10)) | ((df["class"] == 0) & (rng.rand(len(df)) < 0.19))
rf_h = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[~hard_test], y[~hard_test])
ph = rf_h.predict(X[hard_test])
res["hard_acc"] = accuracy_score(y[hard_test], ph)
res["hard_n_test"] = int(hard_test.sum())
cm_hard = confusion_matrix(y[hard_test], ph, labels=range(5))
print("\nHard split classification report (trained Rf<=10 ohm, tested Rf>10 ohm):")
print(classification_report(y[hard_test], ph, target_names=NAMES, digits=3))

# ---------------- 3. Fault location (fault rows only) ----------------
F = df[df["class"] > 0].reset_index(drop=True)
FE = eng(F); XF, dF = FE[LOC_FEATS], F["d_km"]
mk = lambda: ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
Xa, Xb, da, db = train_test_split(XF, dF, test_size=0.2, random_state=SEED, stratify=F["class"])
reg = mk().fit(Xa, da)
pb = reg.predict(Xb)
res["loc_mae_km"] = mean_absolute_error(db, pb)
res["loc_rmse_km"] = mean_squared_error(db, pb) ** 0.5
res["loc_r2"] = r2_score(db, pb)
res["loc_within_1km"] = float((np.abs(db - pb) <= 1).mean())
res["loc_within_2km"] = float((np.abs(db - pb) <= 2).mean())
# held-out resistance band (3-10 ohm never seen) and pure extrapolation (train <=10, test >10)
bandF = (F["Rf_ohm"] > 3) & (F["Rf_ohm"] <= 10)
pbd = mk().fit(XF[~bandF], dF[~bandF]).predict(XF[bandF])
res["loc_band_mae_km"] = mean_absolute_error(dF[bandF], pbd)
hF = F["Rf_ohm"] > 10
ph_loc = mk().fit(XF[~hF], dF[~hF]).predict(XF[hF])
res["loc_hard_mae_km"] = mean_absolute_error(dF[hF], ph_loc)
res["loc_hard_within_2km"] = float((np.abs(dF[hF] - ph_loc) <= 2).mean())
Fb = F.loc[Xb.index].copy(); Fb["pred"] = pb; Fb["err"] = np.abs(Fb["pred"] - Fb["d_km"])
res["loc_mae_by_type"] = {NAMES[k]: float(Fb[Fb["class"] == k]["err"].mean()) for k in range(1, 5)}
bins = pd.cut(Fb["Rf_ohm"], [0, 1, 5, 10, 50])
res["loc_mae_by_Rf_bin"] = {str(k): float(v) for k, v in Fb.groupby(bins, observed=True)["err"].mean().items()}

# classification on the same unseen 3-10 ohm band
rng2 = np.random.RandomState(SEED + 1)
band_test = ((df["class"] > 0) & (df["Rf_ohm"] > 3) & (df["Rf_ohm"] <= 10)) | ((df["class"] == 0) & (rng2.rand(len(df)) < 0.14))
res["band_acc"] = accuracy_score(y[band_test], RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[~band_test], y[~band_test]).predict(X[band_test]))

print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=2))

# ---------------- 4. Save models ----------------
rf_final = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)
reg_final = mk().fit(XF, dF)
joblib.dump({"model": rf_final, "features": FEATS, "classes": NAMES}, "fault_classifier.joblib")
joblib.dump({"model": reg_final, "features": LOC_FEATS}, "fault_locator.joblib") 

# ---------------- 4b. Independent out-of-distribution test ----------------
ood = None
if OOD:
    d2 = pd.read_csv(OOD)
    ood = {"trig": trigger_stats(d2)}
    d2 = d2.dropna(subset=FEATS).reset_index(drop=True)
    yo = d2["class"].astype(int); po = rf_final.predict(d2[FEATS])
    ood["n"] = len(d2); ood["acc"] = accuracy_score(yo, po)
    ood["macro_f1"] = f1_score(yo, po, average="macro")
    ood["cm"] = confusion_matrix(yo, po, labels=range(5))
    ood["report"] = classification_report(yo, po, target_names=NAMES, digits=3)
    Fo = d2[d2["class"] > 0].reset_index(drop=True)
    plo = reg_final.predict(eng(Fo)[LOC_FEATS])
    ood["F"] = Fo.assign(pred=plo, err=np.abs(plo - Fo["d_km"]))
    ood["loc_mae"] = float(ood["F"]["err"].mean())
    ood["loc_within_2km"] = float((ood["F"]["err"] <= 2).mean())
    ood["loc_mae_by_Rf"] = {str(k): float(v) for k, v in ood["F"].groupby(pd.cut(Fo["Rf_ohm"], [0, 1, 5, 10, 50, 100]), observed=True)["err"].mean().items()}
    print("\n=== OUT-OF-DISTRIBUTION TEST (model trained only on the training CSV) ===")
    print(json.dumps({k: v for k, v in ood.items() if k in ("trig", "n", "acc", "macro_f1", "loc_mae", "loc_within_2km", "loc_mae_by_Rf")}, indent=2, default=float))
    print(ood["report"])

# ---------------- 5. Dark-themed results PDF ----------------
BG, FG = "#0d1117", "#e6edf3"
CY, MG, YL, GN, GR = "#38e8ff", "#ff4fd8", "#ffd23f", "#3ddc97", "#9aa4b2"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": "#111826", "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": GR, "ytick.color": GR,
                     "axes.edgecolor": "#30363d", "font.size": 11, "axes.titleweight": "bold"})

def heat(ax, cm, title):
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="cool", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(NAMES); ax.set_yticks(range(5)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, color=CY)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if pct[i, j] > 0.5 else FG, fontsize=12)

with PdfPages("fault_ml_results.pdf") as pdf:
    # Page 1: summary
    fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.05, 0.93, "Transmission line fault detection: ML results", fontsize=22, fontweight="bold", color=FG)
    ax.text(0.05, 0.895, f"{len(df)} simulated runs  |  220 kV, 60 Hz, 40 km line  |  {len(FEATS)} features from one cycle of V and I at the sending end", color=GR, fontsize=11)
    nte = len(yte) // 5
    cards = [("Fault type (same-distribution test)", f"{res['rf_acc']*100:.1f}%", f"0 errors in {nte} per class: error likely < {zero_error_bound(nte)*100:.0f}%", CY),
             ("Location MAE (same-distribution test)", f"{res['loc_mae_km']:.2f} km", f"{res['loc_mae_km']/40*100:.1f}% of line length, {res['loc_within_2km']*100:.0f}% within 2 km", MG)]
    if ood:
        cards += [("Fault type (independent OOD test)", f"{ood['acc']*100:.1f}%", f"{ood['n']} runs: new load/pf, Rf to 100 ohm, 1.5x noise", YL),
                  ("Location MAE (independent OOD test)", f"{ood['loc_mae']:.2f} km", f"{ood['loc_mae']/40*100:.1f}% of line length, {ood['loc_within_2km']*100:.0f}% within 2 km", GN)]
    else:
        cards += [("Fault type, extrapolating to Rf > 10 ohm", f"{res['hard_acc']*100:.1f}%", "trained on Rf <= 10 ohm only", YL),
                  ("Location MAE, extrapolating to Rf > 10 ohm", f"{res['loc_hard_mae_km']:.1f} km", "trained on Rf <= 10 ohm only", GN)]
    for k, (t, v, s, c) in enumerate(cards):
        x = 0.05 + (k % 2) * 0.47; yb = 0.62 - (k // 2) * 0.27
        ax.add_patch(plt.Rectangle((x, yb), 0.43, 0.22, fc="#111826", ec=c, lw=2))
        ax.text(x + 0.02, yb + 0.15, t, color=GR, fontsize=12)
        ax.text(x + 0.02, yb + 0.06, v, color=c, fontsize=30, fontweight="bold")
        ax.text(x + 0.02, yb + 0.02, s, color=GR, fontsize=9.5)
    ax.text(0.05, 0.08, f"5-fold cross-validation accuracy: {res['rf_cv_mean']*100:.1f}% +/- {res['rf_cv_std']*100:.1f}%    |    Decision Tree baseline: {res['tree_acc']*100:.1f}%", color=FG, fontsize=11)
    ax.text(0.05, 0.045, f"Unseen 3-10 ohm band: type accuracy {res['band_acc']*100:.1f}%, location MAE {res['loc_band_mae_km']:.2f} km    |    Extrapolation (Rf > 10): location MAE {res['loc_hard_mae_km']:.1f} km", color=FG, fontsize=11)
    if TRIG:
        ax.text(0.05, 0.01, f"Trigger: {TRIG['trig_missed_faults']}/{TRIG['n_faults']} faults missed, {TRIG['trig_false_alarms']}/{TRIG['n_normal']} false alarms, median detection delay {TRIG['trig_delay_median_ms']:.1f} ms", color=FG, fontsize=11)
    pdf.savefig(fig); plt.close(fig)

    # Page 2: confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    heat(axs[0], cm_random, f"Random split ({res['rf_acc']*100:.1f}%)")
    heat(axs[1], cm_hard, f"Unseen Rf > 10 ohm ({res['hard_acc']*100:.1f}%)")
    fig.suptitle("Fault type classification", fontsize=16, fontweight="bold"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Page 3: feature importance
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))
    for ax, mdl, ttl, col in [(axs[0], rf, "Fault type model", CY), (axs[1], reg, "Fault location model", MG)]:
        imp = pd.Series(mdl.feature_importances_, index=(FEATS if mdl is rf else LOC_FEATS)).sort_values().tail(10)
        ax.barh(imp.index, imp.values, color=col); ax.set_title(ttl, color=col); ax.set_xlabel("Importance")
    fig.suptitle("Top 10 features", fontsize=16, fontweight="bold"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Page 4: location
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    cols = {1: CY, 2: MG, 3: YL, 4: GN}
    for k in range(1, 5):
        m = Fb["class"] == k
        axs[0].scatter(Fb.loc[m, "d_km"], Fb.loc[m, "pred"], s=18, color=cols[k], label=NAMES[k], alpha=0.85)
    axs[0].plot([0, 40], [0, 40], color=GR, ls="--"); axs[0].set_xlabel("True distance (km)"); axs[0].set_ylabel("Predicted (km)")
    axs[0].set_title("Fault location, held-out test", color=CY); axs[0].legend(facecolor="#111826", edgecolor="#30363d")
    sc = axs[1].scatter(Fb["Rf_ohm"], Fb["err"], c=Fb["d_km"], cmap="cool", s=18)
    axs[1].set_xscale("log"); axs[1].set_xlabel("Fault resistance (ohm, log scale)"); axs[1].set_ylabel("Location error (km)")
    axs[1].set_title("Error vs fault resistance", color=MG); fig.colorbar(sc, ax=axs[1], label="True distance (km)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Page 5: why it works - V1 and I0 by class
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.8))
    for ax, col, ttl in [(axs[0], "V1", "Positive-sequence voltage (V)"), (axs[1], "I1", "Positive-sequence current (A)"), (axs[2], "I0", "Zero-sequence current (A)")]:
        data = [df.loc[df["class"] == k, col] for k in range(5)]
        bp = ax.boxplot(data, tick_labels=NAMES, patch_artist=True, medianprops=dict(color=BG))
        for patch, c in zip(bp["boxes"], [GR, CY, MG, YL, GN]): patch.set_facecolor(c)
        for el in ["whiskers", "caps", "fliers"]:
            for item in bp[el]: item.set_color(GR)
        ax.set_title(ttl, fontsize=10)
    fig.suptitle("Why the classes are separable", fontsize=16, fontweight="bold"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    if ood:
        fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
        heat(axs[0], ood["cm"], f"Independent OOD test ({ood['acc']*100:.1f}%)")
        Fo = ood["F"]
        for k in range(1, 5):
            m = Fo["class"] == k
            axs[1].scatter(Fo.loc[m, "d_km"], Fo.loc[m, "pred"], s=18, color=cols[k], label=NAMES[k], alpha=0.85)
        axs[1].plot([0, 40], [0, 40], color=GR, ls="--"); axs[1].set_xlabel("True distance (km)"); axs[1].set_ylabel("Predicted (km)")
        axs[1].set_title(f"Location, OOD (MAE {ood['loc_mae']:.2f} km)", color=MG); axs[1].legend(facecolor="#111826", edgecolor="#30363d")
        fig.suptitle("Independent out-of-distribution test", fontsize=16, fontweight="bold"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
print("\nSaved fault_ml_results.pdf, fault_classifier.joblib, fault_locator.joblib")
