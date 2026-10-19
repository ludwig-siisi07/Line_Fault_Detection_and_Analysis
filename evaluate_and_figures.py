"""
evaluate_and_figures.py
Recomputes every number quoted in the report and draws the report figures.
Run from the repository root:   python python/evaluate_and_figures.py
Reads : data/fault_dataset_train.csv, data/fault_dataset_ood.csv
Writes: results/metrics.json, figures/*.png
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error

SEED = 42
NAMES = ["NF", "LG", "LL", "LLG", "LLL"]
LABEL_COLS = ["class", "pA", "pB", "pC", "pG", "d_km", "Rf_ohm", "tf_s", "class_name", "triggered", "trig_delay_ms"]
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

tr_all = pd.read_csv("data/fault_dataset_train.csv")
od_all = pd.read_csv("data/fault_dataset_ood.csv")
FEATS = [c for c in tr_all.columns if c not in LABEL_COLS]


def eng(D):
    """Extra features used by the locator (apparent impedance per phase and extremes)."""
    D = D.copy()
    V = D[["Va", "Vb", "Vc"]].values
    I = D[["Ia", "Ib", "Ic"]].values
    for k, p in enumerate("abc"):
        D["Z" + p] = V[:, k] / I[:, k]
    Z = D[["Za", "Zb", "Zc"]].values
    D["Zmin"], D["Zmax"] = Z.min(1), Z.max(1)
    D["Imax"], D["Vmin"] = I.max(1), V.min(1)
    D["Zmin_log"], D["V1_I1_log"] = np.log(D["Zmin"]), np.log(D["V1_I1"])
    D["Vmin_Imax"] = D["Vmin"] / D["Imax"]
    return D


LOC_FEATS = [c for c in eng(tr_all.head(5)).columns if c not in LABEL_COLS]
RX = ["Rpa", "Rpb", "Rpc", "Xpa", "Xpb", "Xpc", "Rab", "Rbc", "Rca", "Xab", "Xbc", "Xca", "R1", "X1"]


def trig(D):
    f, n = D[D["class"] > 0], D[D["class"] == 0]
    return {"faults": int(len(f)), "normal": int(len(n)), "missed": int((f["triggered"] == 0).sum()),
            "false_alarms": int((n["triggered"] == 1).sum()),
            "delay_median_ms": float(f["trig_delay_ms"].median()), "delay_max_ms": float(f["trig_delay_ms"].max())}


def cp_interval(k, n):
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


M = {"n_train": int(len(tr_all)), "n_ood": int(len(od_all)), "n_features": len(FEATS), "n_loc_features": len(LOC_FEATS),
     "trigger_train": trig(tr_all), "trigger_ood": trig(od_all)}

tr = tr_all.dropna(subset=FEATS).reset_index(drop=True)
od = od_all.dropna(subset=FEATS).reset_index(drop=True)
X, y = tr[FEATS], tr["class"].astype(int)

# ---------- classification, same distribution ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
tree = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xtr, ytr)
p_same = rf.predict(Xte)
cm_same = confusion_matrix(yte, p_same, labels=range(5))
cv = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1), X, y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
t3 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr[["I2_I1", "I0_I1", "dIa"]], ytr)
M["cls_same"] = {"n_test": int(len(yte)), "rf_acc": float(accuracy_score(yte, p_same)),
                 "tree_acc": float(accuracy_score(yte, tree.predict(Xte))),
                 "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
                 "three_feature_tree_acc": float(accuracy_score(yte, t3.predict(Xte[["I2_I1", "I0_I1", "dIa"]])))}

# stress: extrapolate to fault resistance > 10 ohm, and a held-out 3-10 ohm band
rng = np.random.RandomState(SEED)
hard = ((tr["class"] > 0) & (tr["Rf_ohm"] > 10)) | ((tr["class"] == 0) & (rng.rand(len(tr)) < 0.19))
rf_h = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[~hard], y[~hard])
rep_h = classification_report(y[hard], rf_h.predict(X[hard]), target_names=NAMES, output_dict=True)
M["cls_extrapolate_Rf_gt10"] = {"n_test": int(hard.sum()), "acc": float(accuracy_score(y[hard], rf_h.predict(X[hard]))),
                                "recall": {c: float(rep_h[c]["recall"]) for c in NAMES}}
rng2 = np.random.RandomState(SEED + 1)
band = ((tr["class"] > 0) & (tr["Rf_ohm"] > 3) & (tr["Rf_ohm"] <= 10)) | ((tr["class"] == 0) & (rng2.rand(len(tr)) < 0.14))
rf_b = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[~band], y[~band])
M["cls_heldout_band_3_10"] = {"n_test": int(band.sum()), "acc": float(accuracy_score(y[band], rf_b.predict(X[band])))}

# ---------- classification, independent out-of-distribution set ----------
Xo, yo = od[FEATS], od["class"].astype(int)
rf_final = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)
p_ood = rf_final.predict(Xo)
cm_ood = confusion_matrix(yo, p_ood, labels=range(5))
k, n = int((p_ood == yo).sum()), len(yo)
lo, hi = cp_interval(k, n)
rep = classification_report(yo, p_ood, target_names=NAMES, output_dict=True)
bad = (p_ood != yo).values
wrong = od[bad][["class_name", "d_km", "Rf_ohm"]].assign(pred=[NAMES[i] for i in p_ood[bad]])
M["cls_ood"] = {"n_test": n, "correct": k, "acc": k / n, "ci95": [lo, hi],
                "per_class": {c: {"precision": float(rep[c]["precision"]), "recall": float(rep[c]["recall"])} for c in NAMES},
                "errors": wrong.round(1).to_dict("records")}

# ---------- location ----------
F = tr[tr["class"] > 0].reset_index(drop=True)
FE = eng(F)
XF, dF = FE[LOC_FEATS], F["d_km"]
Fo = od[od["class"] > 0].reset_index(drop=True)
FEo = eng(Fo)
mk = lambda s=SEED, n=400: ExtraTreesRegressor(n_estimators=n, random_state=s, n_jobs=-1)
bins = [0, 1, 5, 10, 50, 100]
blab = ["<1", "1-5", "5-10", "10-50", "50-100"]


def by_bin(rf_ohm, err):
    g = pd.Series(np.asarray(err)).groupby(pd.cut(pd.Series(np.asarray(rf_ohm)), bins, labels=blab), observed=True).mean()
    return {str(k_): float(v) for k_, v in g.items()}


Xa, Xb, da, db = train_test_split(XF, dF, test_size=0.2, random_state=SEED, stratify=F["class"])
reg = mk().fit(Xa, da)
eb = np.abs(reg.predict(Xb) - db.values)
splits = []
for s in range(5):
    a, b = train_test_split(F.index, test_size=0.2, stratify=F["class"], random_state=s)
    splits.append(float(np.abs(mk(s).fit(XF.loc[a], dF.loc[a]).predict(XF.loc[b]) - dF.loc[b]).mean()))
M["loc_same"] = {"n_test": int(len(db)), "mae_seed42": float(eb.mean()), "median": float(np.median(eb)),
                 "within_2km": float((eb <= 2).mean()), "mae_5splits": splits,
                 "mae_5splits_mean": float(np.mean(splits)), "mae_5splits_std": float(np.std(splits)),
                 "by_Rf_bin": by_bin(F.loc[Xb.index, "Rf_ohm"].values, eb)}
reg_final = mk().fit(XF, dF)
po = reg_final.predict(FEo[LOC_FEATS])
eo = np.abs(po - Fo["d_km"].values)
rs = np.random.RandomState(0)
bs = [rs.choice(eo, len(eo)).mean() for _ in range(2000)]
edge = ((Fo["d_km"] < 2) | (Fo["d_km"] > 38)).values
inr = (Fo["Rf_ohm"] <= 50).values
M["loc_ood"] = {"n_test": int(len(eo)), "mae": float(eo.mean()),
                "mae_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "median": float(np.median(eo)), "p90": float(np.percentile(eo, 90)), "within_2km": float((eo <= 2).mean()),
                "mae_Rf_le_50": float(eo[inr].mean()), "n_Rf_le_50": int(inr.sum()),
                "mae_Rf_gt_50": float(eo[~inr].mean()), "n_Rf_gt_50": int((~inr).sum()),
                "share_error_Rf_gt_50": float(eo[~inr].sum() / eo.sum()),
                "mae_edges": float(eo[edge].mean()), "n_edges": int(edge.sum()), "mae_interior": float(eo[~edge].mean()),
                "by_Rf_bin": by_bin(Fo["Rf_ohm"].values, eo)}

hF = (F["Rf_ohm"] > 10).values
bF = ((F["Rf_ohm"] > 3) & (F["Rf_ohm"] <= 10)).values
M["loc_extrapolate_Rf_gt10"] = {"mae": float(mean_absolute_error(dF[hF], mk().fit(XF[~hF], dF[~hF]).predict(XF[hF])))}
M["loc_heldout_band_3_10"] = {"mae": float(mean_absolute_error(dF[bF], mk().fit(XF[~bF], dF[~bF]).predict(XF[bF])))}

# ablation: do the resistance/reactance features matter?
abl = {}
for name, cols in [("all", LOC_FEATS), ("without_RX", [c for c in LOC_FEATS if c not in RX]),
                   ("RX_only_plus_V1_I1", [c for c in LOC_FEATS if c in RX or c in ("V1", "I1")])]:
    r = []
    for s in range(5):
        a, b = train_test_split(F.index, test_size=0.2, stratify=F["class"], random_state=s)
        m_ = mk(s, 200).fit(FE.loc[a, cols], dF.loc[a])
        e_ = np.abs(m_.predict(FE.loc[b, cols]) - dF.loc[b])
        r.append((float(e_.mean()), by_bin(F.loc[b, "Rf_ohm"].values, e_.values)))
    abl[name] = {"mae_mean": float(np.mean([x[0] for x in r])), "mae_std": float(np.std([x[0] for x in r])),
                 "high_Rf_10_50_mean": float(np.mean([x[1]["10-50"] for x in r]))}
M["ablation"] = abl


# classical baseline: single-ended loop-reactance estimate, given the TRUE faulted phases (best case)
def baseline(D):
    D = D.dropna(subset=["X1"]).reset_index(drop=True)
    c = D["class"].values
    pa, pb, pc = D["pA"].values == 1, D["pB"].values == 1, D["pC"].values == 1
    ll = np.where(pa & pb, D["Xab"], np.where(pb & pc, D["Xbc"], D["Xca"]))
    lg = np.where(pa, D["Xpa"], np.where(pb, D["Xpb"], D["Xpc"]))
    x = np.where(c == 4, D["X1"], np.where(c == 2, ll, lg))
    return np.abs(np.clip(x / 0.5, 0, 40) - D["d_km"].values)


M["baseline"] = {}
for nm, D in [("train", F), ("ood", Fo)]:
    e = baseline(D)
    M["baseline"][nm] = {"mae": float(e.mean()), "median": float(np.median(e)), "within_2km": float((e <= 2).mean())}

# simulation vs circuit theory: peak-phase fault current for low-resistance faults (training set)
E = 263.1e3 / np.sqrt(3)
chk = {}
for c, lab in [(4, "LLL"), (1, "LG"), (2, "LL")]:
    S = F[(F["class"] == c) & (F["Rf_ohm"] < 5)]
    z = (1 + 10j) + S["d_km"].values * (0.15 + 0.5j) + S["Rf_ohm"].values
    ana = E / abs(z) if c != 2 else np.sqrt(3) * E / (2 * abs(z))
    r = S[["Ia", "Ib", "Ic"]].max(axis=1).values / ana
    chk[lab] = {"n": int(len(S)), "median_ratio": float(np.median(r)),
                "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))}
M["theory_check"] = chk
json.dump(M, open("results/metrics.json", "w"), indent=2)

# =============== FIGURES (print-friendly, light theme) ===============
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"NF": "#7a7a7a", "LG": "#1f77b4", "LL": "#d62728", "LLG": "#ff9f1c", "LLL": "#2ca02c"}


def box(ax, x, y, w, h, txt, fc="#e8f1fb", ec="#1f4e79", fs=10):
    """Rounded box with centred text. The font shrinks until the text fits inside the box."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc=fc, ec=ec, lw=1.4))
    t = ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.25)
    r = ax.figure.canvas.get_renderer()
    x0, y0 = ax.transData.transform((x, y))
    x1, y1 = ax.transData.transform((x + w, y + h))
    while fs > 5:
        bb = t.get_window_extent(r)
        if bb.width <= 0.88 * (x1 - x0) and bb.height <= 0.80 * (y1 - y0):
            break
        fs -= 0.5
        t.set_fontsize(fs)


def arrow(ax, x0, y0, x1, y1, color="#333333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.5,
                                 color=color, shrinkA=0, shrinkB=0))


# ---- system diagram ----
fig, ax = plt.subplots(figsize=(9.0, 3.8))
ax.set_xlim(-0.15, 10.15); ax.set_ylim(0, 4.2); ax.axis("off")
W, G = 1.6, 0.5                        # box width and gap (5 boxes + 4 gaps = 10)
xs = [i * (W + G) for i in range(5)]
Y, H = 2.0, 1.3
ymid = Y + H / 2
ax.text(5.0, 3.9, "220 kV, 60 Hz, 40 km line:  R = 0.15 ohm/km,  L = 1.3263 mH/km", ha="center", fontsize=11)
labels = ["Source\n263.1 kV (L-L)\n1 + j10 ohm", "Sending-end\nmeasurement\n(V, I)", "Line 1\nd km",
          "Line 2\n(40 - d) km", "Load\n381 MVA\n0.8 pf lag"]
styles = [dict(), dict(fc="#fff4d6", ec="#9c6b00"), dict(), dict(), dict()]
for x, t, st in zip(xs, labels, styles):
    box(ax, x, Y, W, H, t, **st)
for i in range(4):
    arrow(ax, xs[i] + W, ymid, xs[i + 1], ymid)
jx = xs[2] + W + G / 2                  # the fault joins the line between Line 1 and Line 2
ax.plot([jx, jx], [ymid, 1.05], color="#a32020", lw=2)
ax.plot([jx], [ymid], "o", color="#a32020", ms=6)
ax.text(jx + 0.15, 1.5, "at distance d", fontsize=9.5, color="#a32020", va="center")
box(ax, jx - 1.7, 0.1, 3.4, 0.95, "Fault: LG, LL, LLG or LLL\nfault resistance Rf", fc="#fde2e2", ec="#a32020")
fig.savefig("figures/fig_system.png"); plt.close(fig)

# ---- pipeline diagram ----
fig, ax = plt.subplots(figsize=(9.5, 2.6))
ax.set_xlim(-0.15, 10.15); ax.set_ylim(0, 2.6); ax.axis("off")
W, G = 1.76, 0.30                      # 5 boxes + 4 gaps = 10
steps = ["Simulink +\nMATLAB sweep\n(random faults)", "Trigger +\n1-cycle windows",
         "37 features\n(phasors,\nsequence,\nimpedance)", "Models:\nRandom Forest,\nExtra Trees",
         "Output:\nfault? type,\ndistance"]
for i, t in enumerate(steps):
    x = i * (W + G)
    box(ax, x, 0.5, W, 1.6, t, fs=9.5)
    if i < len(steps) - 1:
        arrow(ax, x + W + 0.03, 1.3, x + W + G - 0.03, 1.3)
fig.savefig("figures/fig_pipeline.png"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9))
for ax, col, ttl in [(axs[0], "V1", "Positive-seq. voltage (V)"), (axs[1], "I1", "Positive-seq. current (A)"),
                     (axs[2], "I0", "Zero-seq. current (A)")]:
    bp = ax.boxplot([tr.loc[tr["class"] == k_, col] for k_ in range(5)], tick_labels=NAMES, patch_artist=True,
                    medianprops=dict(color="black"), flierprops=dict(markersize=2))
    for p_, n_ in zip(bp["boxes"], NAMES):
        p_.set_facecolor(C[n_]); p_.set_alpha(0.75)
    ax.set_title(ttl, fontsize=9); ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig("figures/fig_separability.png"); plt.close(fig)


def heat(ax, cm, title):
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(NAMES); ax.set_yticks(range(5)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if pct[i, j] > 0.5 else "black")


fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5))
heat(axs[0], cm_same, f"Same-distribution test (n = {len(yte)})")
heat(axs[1], cm_ood, f"Independent test (n = {n})")
fig.tight_layout(); fig.savefig("figures/fig_confusion.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5))
for c in range(1, 5):
    m = (Fo["class"] == c).values
    axs[0].scatter(Fo.loc[m, "d_km"], po[m], s=10, color=C[NAMES[c]], label=NAMES[c], alpha=0.8)
axs[0].plot([0, 40], [0, 40], "k--", lw=0.8)
axs[0].set_xlabel("True distance (km)"); axs[0].set_ylabel("Predicted distance (km)")
axs[0].set_title("Independent test: location", fontsize=10); axs[0].legend(fontsize=8, frameon=False)
sc = axs[1].scatter(Fo["Rf_ohm"], eo, c=Fo["d_km"], cmap="viridis", s=10)
axs[1].set_xscale("log"); axs[1].axvline(50, color="red", ls=":", lw=1)
axs[1].text(52, eo.max() * 0.9, "training\nlimit", color="red", fontsize=8)
axs[1].set_xlabel("Fault resistance (ohm, log scale)"); axs[1].set_ylabel("Location error (km)")
axs[1].set_title("Error vs fault resistance", fontsize=10)
fig.colorbar(sc, ax=axs[1], label="True distance (km)")
fig.tight_layout(); fig.savefig("figures/fig_location.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.0))
xs = np.arange(len(blab)); w = 0.38
s1 = [M["loc_same"]["by_Rf_bin"].get(b, np.nan) for b in blab]
s2 = [M["loc_ood"]["by_Rf_bin"].get(b, np.nan) for b in blab]
ax.bar(xs - w / 2, s1, w, label="Same-distribution", color="#1f77b4")
ax.bar(xs + w / 2, s2, w, label="Independent test", color="#ff9f1c")
ax.set_xticks(xs); ax.set_xticklabels(blab)
ax.set_xlabel("Fault resistance (ohm)"); ax.set_ylabel("Mean location error (km)"); ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig_rf_bins.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
for ax, mdl, cols, ttl, col in [(axs[0], rf_final, FEATS, "Fault type (Random Forest)", "#1f77b4"),
                                (axs[1], reg_final, LOC_FEATS, "Location (Extra Trees)", "#ff9f1c")]:
    imp = pd.Series(mdl.feature_importances_, index=cols).sort_values().tail(8)
    ax.barh(imp.index, imp.values, color=col); ax.set_title(ttl, fontsize=10); ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig("figures/fig_importance.png"); plt.close(fig)
print("done")
