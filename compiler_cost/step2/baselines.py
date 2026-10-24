#!/usr/bin/env python3
"""baselines.py — Step 2 cheap baselines for Neuron compile cost.

Targets: log10 compile wall (bench, default flags, cc 2.21), log10 peak RSS, and compile-failure classification.
Feature sets: cheap4 (instructions, proto bytes, tensor bytes, FLOPs), step0 (35 counts/bytes/FLOPs), full (features2).
Models: power law (instructions only), ridge on log1p-standardised features, gradient-boosted trees (sklearn HGB).
Splits: random 5-fold; leave-one-family-out; leave-one-source-out; sweep extrapolation (train on non-sweep, test on sweeps);
plus pair-direction accuracy along sweep series. usage: baselines.py --dataset ~/work/ncc/data/dataset.jsonl --report out.md
"""
import argparse, json, math, collections, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

CHEAP4 = ["n_instructions", "proto_bytes", "total_tensor_bytes", "flops"]
META = ("hlo_id", "prog_id", "family", "source", "parent", "sweep_key", "sweep_value", "y", "y_src", "rss", "fail", "has_bench", "is_step")
STEP0 = ["n_instructions", "n_computations", "n_entry_instructions", "n_params", "param_bytes", "output_bytes", "total_tensor_bytes", "largest_tensor_bytes", "max_rank", "n_distinct_shapes",
         "dot_flops", "conv_flops", "elementwise_bytes", "n_dot", "n_conv", "n_reduce", "n_fusion", "n_while", "n_conditional", "n_custom_call", "n_collective", "n_gather", "n_scatter", "n_sort",
         "n_dynamic_slice", "n_transpose", "n_reshape", "n_broadcast", "n_concat", "n_slice", "n_pad", "n_iota", "n_rng", "n_dtypes", "frac_bf16", "frac_f32", "proto_bytes"]

def load(path, target):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    recs = []
    for r in rows:
        f = dict(r["features"]); f["flops"] = (f.get("dot_flops") or 0) + (f.get("conv_flops") or 0)
        y = r.get("bench_wall_s") if target == "bench" else (r.get("bench_wall_s") or r.get("capture_compile_s"))
        rec = dict(hlo_id=r["hlo_id"], prog_id=r["prog_id"], family=r["family"], source=r["source"], parent=r.get("parent"), sweep_key=r.get("sweep_key"), sweep_value=r.get("sweep_value"),
                   y=y, y_src=("bench" if r.get("bench_wall_s") else ("capture" if r.get("capture_compile_s") else None)), rss=r.get("bench_rss_gb"),
                   fail=int((r.get("bench_status") not in (None, "PASS")) or (not r.get("framework_done", True))), has_bench=r.get("bench_wall_s") is not None, is_step=bool(r.get("module_is_new")))
        rec.update({k: v for k, v in f.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}); recs.append(rec)
    df = pd.DataFrame(recs); return df

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(y) < 3: return dict(n=len(y), R2=float("nan"), medAPE=float("nan"), p90APE=float("nan"), within2x=float("nan"), spearman=float("nan"))
    ape = np.abs(10 ** p - 10 ** y) / np.maximum(10 ** y, 1e-9)
    ss = ((y - y.mean()) ** 2).sum(); r2 = 1 - ((y - p) ** 2).sum() / ss if ss > 0 else float("nan")
    return dict(n=len(y), R2=round(r2, 3), medAPE=round(float(np.median(ape)), 3), p90APE=round(float(np.percentile(ape, 90)), 2), within2x=round(float(np.mean(np.abs(y - p) < math.log10(2))), 3), spearman=round(float(spearmanr(y, p).correlation), 3) if len(y) > 2 else float("nan"))

def make_model(kind, n_feat):
    if kind == "powerlaw": return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if kind == "ridge": return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if kind == "gbdt": return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=8, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)

def X_of(df, cols, kind):
    X = df[cols].fillna(0).astype(float).values
    return np.log1p(np.maximum(X, 0)) if kind in ("ridge", "powerlaw") else X

def cv_eval(df, cols, kind, split, target="y"):
    y = np.log10(np.maximum(df[target].values.astype(float), 1e-3)); X = X_of(df, cols, kind); pred = np.full(len(df), np.nan)
    if split == "random": folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    elif split == "family": folds = list(LeaveOneGroupOut().split(X, y, groups=df["family"].values))
    elif split == "source": folds = list(LeaveOneGroupOut().split(X, y, groups=df["source"].values))
    elif split == "sweep":
        te = np.where(df["source"].values == "sweep")[0]; tr = np.where(df["source"].values != "sweep")[0]; folds = [(tr, te)]
    else: raise ValueError(split)
    for tr, te in folds:
        if len(tr) < 10 or len(te) == 0: continue
        m = make_model(kind, X.shape[1]); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    ok = ~np.isnan(pred); return metrics(y[ok], pred[ok]), pred

def per_group(df, pred, key, target="y"):
    y = np.log10(np.maximum(df[target].values.astype(float), 1e-3)); out = []
    for g, idx in df.groupby(key).indices.items():
        ok = ~np.isnan(pred[idx])
        if ok.sum() >= 5: m = metrics(y[idx][ok], pred[idx][ok]); m["group"] = g; out.append(m)
    return pd.DataFrame(out).sort_values("medAPE", ascending=False) if out else pd.DataFrame()

def sweep_pairs(df, pred):
    """direction accuracy along sweep series (consecutive values, actual change > 20%)"""
    y = np.log10(np.maximum(df["y"].values.astype(float), 1e-3)); n = 0; correct = 0; lr_a = []; lr_p = []
    series = []
    for (par, key, _st), labels in df[df["source"] == "sweep"].groupby(["parent", "sweep_key", "is_step"]).groups.items():
        pos = np.array(sorted(labels, key=lambda i: df.at[i, "sweep_value"]))   # RangeIndex labels == positions
        sa = []; sp = []
        for a, b in zip(pos[:-1], pos[1:]):
            if np.isnan(pred[a]) or np.isnan(pred[b]): continue
            da = y[b] - y[a]; dp = pred[b] - pred[a]; lr_a.append(da); lr_p.append(dp); sa.append(da); sp.append(dp)
            if abs(da) > math.log10(1.2): n += 1; correct += int(np.sign(da) == np.sign(dp))
        if len(sa) >= 4: series.append(dict(series=f"{str(par)[:38]}|{key}|{'step' if _st else 'init'}", n=len(pos), swing=round(10 ** (y[pos].max() - y[pos].min()), 1), spearman_steps=round(float(spearmanr(sa, sp).correlation), 2), dir_acc=round(float(np.mean(np.sign(sa) == np.sign(sp))), 2)))
    return dict(pairs_gt20pct=n, direction_acc=round(correct / n, 3) if n else float("nan"), logratio_spearman=round(float(spearmanr(lr_a, lr_p).correlation), 3) if len(lr_a) > 2 else float("nan"), series=series)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default=os.path.expanduser("~/work/ncc/data/dataset.jsonl")); ap.add_argument("--report", default="results/baselines_latest.md")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True); L = []
    def P(s=""): print(s); L.append(s)
    for target in ("bench", "combined", "combined-step"):
        df = load(a.dataset, "bench" if target == "bench" else "combined"); df = df[df["y"].notna() & (df["y"] > 0)].reset_index(drop=True)
        if target == "combined-step": df = df[df["is_step"]].reset_index(drop=True)
        full_cols = [c for c in df.columns if c not in META]
        P(f"\n## Target: log10 compile wall — {target} ({len(df)} modules; sources {dict(collections.Counter(df['source']))}; y from {dict(collections.Counter(df['y_src']))})\n")
        P(f"wall seconds: median {df['y'].median():.1f}, p90 {df['y'].quantile(.9):.1f}, max {df['y'].max():.0f}; families {df['family'].nunique()}\n")
        P("| features | model | split | n | R² | median APE | p90 APE | within 2× | Spearman |\n|---|---|---|---|---|---|---|---|---|")
        preds = {}
        for fname, cols in (("instructions-only", ["n_instructions"]), ("cheap4", CHEAP4), ("step0-35", STEP0), ("full", full_cols)):
            cols = [c for c in cols if c in df.columns]
            for kind in (("powerlaw",) if fname == "instructions-only" else ("ridge", "gbdt")):
                for split in ("random", "family", "source", "sweep"):
                    m, pred = cv_eval(df, cols, kind, split); preds[(fname, kind, split)] = pred
                    P(f"| {fname} | {kind} | {split} | {m['n']} | {m['R2']} | {m['medAPE']} | {m['p90APE']} | {m['within2x']} | {m['spearman']} |")
        P("\n### Sweep-series direction accuracy (train on non-sweep programs, predict along each one-dimensional sweep)\n")
        P("| features | model | pairs (>20% change) | direction acc | Spearman of log-ratios |\n|---|---|---|---|---|")
        best = None
        for (fname, kind, split), pred in preds.items():
            if split != "sweep": continue
            sp = sweep_pairs(df, pred); P(f"| {fname} | {kind} | {sp['pairs_gt20pct']} | {sp['direction_acc']} | {sp['logratio_spearman']} |")
            if (fname, kind) == ("full", "gbdt"): best = sp
        if best and best.get("series"):
            P("\nPer sweep series (full + GBDT): max/min compile-time swing along the series, Spearman of consecutive log-ratios, direction accuracy\n")
            P(pd.DataFrame(best["series"]).sort_values("swing", ascending=False).to_markdown(index=False))
        P("\n### Leave-one-family-out, full features + GBDT: per held-out family (worst first)\n")
        pg = per_group(df, preds[("full", "gbdt", "family")], "family")
        if len(pg): P(pg.head(14).to_markdown(index=False))
        if target == "bench":
            d2 = df[df["rss"].notna() & (df["rss"] > 0)].reset_index(drop=True)
            if len(d2) > 50:
                P(f"\n## Target: log10 peak RSS (GB) — bench ({len(d2)} modules)\n"); P("| features | model | split | n | R² | median APE | p90 APE | within 2× | Spearman |\n|---|---|---|---|---|---|---|---|---|")
                for fname, cols in (("cheap4", CHEAP4), ("full", full_cols)):
                    for kind in ("ridge", "gbdt"):
                        for split in ("random", "family"):
                            m, _ = cv_eval(d2, [c for c in cols if c in d2.columns], kind, split, target="rss"); P(f"| {fname} | {kind} | {split} | {m['n']} | {m['R2']} | {m['medAPE']} | {m['p90APE']} | {m['within2x']} | {m['spearman']} |")
    # failure classifier over all modules with features (label: compile failed in framework or bench)
    df = load(a.dataset, "combined"); full_cols = [c for c in df.columns if c not in META]
    P(f"\n## Compile-failure classifier ({len(df)} modules, {int(df['fail'].sum())} failures; label = framework compile failed or bench FAIL/OOM/TIMEOUT)\n")
    P("| features | split | AUC | PR-AUC | failures caught at 10% flag rate |\n|---|---|---|---|---|")
    yb = df["fail"].values
    for fname, cols in (("cheap4", CHEAP4), ("full", full_cols)):
        X = df[[c for c in cols if c in df.columns]].fillna(0).astype(float).values
        for split in ("random", "family"):
            prob = np.full(len(df), np.nan)
            folds = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, yb, groups=df["prog_id"].values) if split == "random" else LeaveOneGroupOut().split(X, yb, groups=df["family"].values)
            for tr, te in folds:
                if yb[tr].sum() == 0: prob[te] = 0.0; continue
                clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=0).fit(X[tr], yb[tr]); prob[te] = clf.predict_proba(X[te])[:, 1]
            ok = ~np.isnan(prob); thr = np.quantile(prob[ok], 0.9); caught = ((prob[ok] >= thr) & (yb[ok] == 1)).sum() / max(1, yb[ok].sum())
            P(f"| {fname} | {split} | {roc_auc_score(yb[ok], prob[ok]):.3f} | {average_precision_score(yb[ok], prob[ok]):.3f} | {caught:.2f} |")
    open(a.report, "w").write("# Step 2 baselines — " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()) + "\n" + "\n".join(L) + "\n"); print("\nreport ->", a.report)

if __name__ == "__main__":
    main()
