"""Cross-fitted comparison of ensembling arms across collar widths, with paired clip bootstrap.

Each fold is processed independently: optional per-member, per-class calibration and every
arm's decode parameters are fitted on the other fold, then scored on this one.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "2")

import argparse
import json
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression

import sedlib as L

THRS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
MEDS = [1, 5, 9, 15, 25, 41, 61]
COLLARS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
G = {}   # read-only state shared with forked workers


def audioset_overlap():
    """YouTube ids of DESED public-eval clips that appear in any AudioSet split (weak or strong)."""
    with open(os.path.join(L.LISTS, "desed_eval_audioset_overlap.txt")) as fh:
        return {l.strip() for l in fh if l.strip()}


def counts(events, collars=COLLARS):
    return L.event_counts(events, G["refs"], G["names"], G["classes"], collars, G["pct"])


def platt(logits, dev):
    """Per-class logistic calibration of one member, fitted on dev clips only."""
    y = G["yref"][dev]
    out = np.empty_like(logits)
    rng = np.random.default_rng(0)
    for j in range(logits.shape[1]):
        z, t = logits[dev, j].ravel(), y[:, j].ravel()
        idx = rng.choice(len(z), min(len(z), 200_000), replace=False)
        if t[idx].min() == t[idx].max():
            out[:, j] = logits[:, j]
            continue
        lr = LogisticRegression(C=1e4).fit(z[idx, None], t[idx])
        out[:, j] = lr.coef_[0, 0] * logits[:, j] + lr.intercept_[0]
    return out


def select(grid, dev, k):
    return max(grid, key=lambda g: L.f1_from_counts(grid[g][k][dev]))


def select_cw(grid, dev, k):
    """Per-class choice: for each class, the config with the best dev F1 on that class (macro F1 decomposes by class)."""
    cfgs = list(grid)
    s = np.stack([grid[g][k][dev].sum(0) for g in cfgs])            # (G, C, 3)
    f1 = np.where(s[..., 1] + s[..., 2] > 0, 2 * s[..., 0] / np.maximum(s[..., 1] + s[..., 2], 1), 0.0)
    return [cfgs[i] for i in f1.argmax(0)], f1.max(0)


def decode_arm(task):
    """Grid of counts for one probability-curve arm on one fold."""
    f, name = task
    lg = G["lg"][f]
    if name.startswith("sebb:"):
        base = name[5:]
        if base.startswith("single:"):
            prob = L.sigmoid(lg[G["models"].index(base[7:])])
        elif base == "avg_prob":
            prob = L.sigmoid(lg).mean(0)
        else:
            prob = L.sigmoid(lg.mean(0))
        cands = L.sebb_candidates(prob)
        return (f, name), {key + (t,): counts(L.threshold_events(c, t)) for key, c in cands.items() for t in THRS}
    if name.startswith("single:"):
        prob, meds = L.sigmoid(lg[G["models"].index(name[7:])]), MEDS
    elif name == "avg_prob":
        prob, meds = L.sigmoid(lg).mean(0), MEDS
    elif name == "avg_prob_nomed":
        prob, meds = L.sigmoid(lg).mean(0), [1]
    elif name == "avg_prob_gap":   # averaging plus gap bridging: tests whether fusion only repairs fragmentation
        prob = L.sigmoid(lg).mean(0)
        return (f, name), {(t, m, g): counts(L.decode(prob, t, m, g)) for t in THRS for m in (1, 9, 25)
                           for g in (0, 10, 25, 50)}
    elif name == "avg_prob_pp":   # conventional post-processing: filter, gap filling and minimum duration, jointly tuned
        prob = L.sigmoid(lg).mean(0)
        return (f, name), {(t, m, g, d): counts(L.decode(prob, t, m, g, d)) for t in THRS for m in (1, 9, 25)
                           for g in (0, 10, 25, 50) for d in (0, 10, 25, 50)}
    elif name == "avg_logit":
        prob, meds = L.sigmoid(lg.mean(0)), MEDS
    else:
        raise ValueError(name)
    return (f, name), {(t, m): counts(L.decode(prob, t, m)) for t in THRS for m in meds}


def fusion_arm(task):
    """Event-level fusion of members decoded with their own dev-tuned configs, on one fold."""
    f, name = task
    dev = G["folds"] != f
    C, M = len(G["classes"]), len(G["models"])
    if name == "vote":
        cfgs = [(v,) for v in range(1, M + 1)]
        run = lambda evs, cfg: L.frame_vote(evs, C, cfg[0])
    else:
        edge = {"wbf_weighted": "weighted", "wbf_union": "union"}.get(name, "median")
        ious = [float(name[7:])] if name.startswith("wbf_iou") else [0.3, 0.5, 0.7]
        votes = [(M + 1) // 2] if name == "wbf_majority" else range(1, M + 1)
        cfgs = [(i, v) for i in ious for v in votes]
        run = lambda evs, cfg: L.wbf_1d(evs, C, cfg[0], cfg[1], edge)
    if name.startswith("cw:"):
        return fusion_arm_cw(f, name)
    sebb = name == "wbf_median_sebb"
    cands = {}
    out = np.zeros((len(COLLARS), len(G["names"]), C, 3), np.int32)
    chosen = {}
    fk = G.get("fixed_k")
    for k in ([fk] if fk is not None else range(len(COLLARS))):
        evs = []
        for m, model in enumerate(G["models"]):
            prob = L.sigmoid(G["lg"][f][m])
            if sebb:
                cfg = G["member_cfg_sebb"][(f, model, k)]
                if model not in cands:
                    cands[model] = L.sebb_candidates(prob)
                evs.append(L.threshold_events(cands[model][cfg[:3]], cfg[3]))
            elif name == "wbf_shared":
                evs.append(L.decode(prob, *G["shared_cfg"][(f, k)]))
            else:
                evs.append(L.decode(prob, *G["member_cfg"][(f, model, k)]))
        if fk is not None:   # one configuration, chosen at the fixed collar, scored at every collar
            scored = {cfg: counts(run(evs, cfg)) for cfg in cfgs}
            best = max(scored, key=lambda g: L.f1_from_counts(scored[g][fk][dev]))
            out[:] = scored[best]
            chosen = {kk: best for kk in range(len(COLLARS))}
            break
        scored = {cfg: counts(run(evs, cfg), [COLLARS[k]])[0] for cfg in cfgs}
        best = max(scored, key=lambda g: L.f1_from_counts(scored[g][dev]))
        out[k] = scored[best]
        chosen[k] = best
    return (f, name), (out, chosen)


def fusion_arm_cw(f, name):
    """Class-wise event fusion: per-class member configs (from class-wise single selection) and per-class (IoU, votes)."""
    dev = G["folds"] != f
    C, M = len(G["classes"]), len(G["models"])
    sebb = name == "cw:wbf_median_sebb"
    cfgs = [(i, v) for i in (0.3, 0.5, 0.7) for v in range(1, M + 1)]
    out = np.zeros((len(COLLARS), len(G["names"]), C, 3), np.int32)
    chosen = {}
    cache, cands = {}, {}
    for k in range(len(COLLARS)):
        evs = []
        for m, model in enumerate(G["models"]):
            per_class = G["member_cfg_sebb_cw" if sebb else "member_cfg_cw"][(f, model, k)]
            for cfg in set(per_class):
                if (m, cfg) in cache:
                    continue
                prob = L.sigmoid(G["lg"][f][m])
                if sebb:
                    if m not in cands:
                        cands[m] = L.sebb_candidates(prob)
                    cache[(m, cfg)] = L.threshold_events(cands[m][cfg[:3]], cfg[3])
                else:
                    cache[(m, cfg)] = L.decode(prob, *cfg)
            evs.append([{c: cache[(m, per_class[c])][i][c] for c in range(C)} for i in range(len(G["names"]))])
        scored = {cfg: counts([*L.wbf_1d(evs, C, cfg[0], cfg[1], "median")], [COLLARS[k]])[0] for cfg in cfgs}
        best, _ = select_cw({g: v[None] for g, v in scored.items()}, dev, 0)
        for c, g in enumerate(best):
            out[k][:, c] = scored[g][:, c]
        chosen[k] = [list(g) for g in best]
    return (f, name), (out, chosen)


def bootstrap(pooled, ref_arms, n=1000):
    """Percentile CIs per arm, and paired-difference CIs against each reference arm (same resamples)."""
    rng = np.random.default_rng(0)
    N = len(G["names"])
    idx = rng.integers(0, N, (n, N))
    samples = {a: np.stack([L.f1_from_counts(c[:, i]) for i in idx]) for a, c in pooled.items()}
    out = {}
    for a, s in samples.items():
        out[a] = {"lo": np.percentile(s, 2.5, 0).round(4).tolist(), "hi": np.percentile(s, 97.5, 0).round(4).tolist()}
        for r in ref_arms:
            d = s - samples[r]
            out[a][f"d_{r}"] = [np.percentile(d, 2.5, 0).round(4).tolist(), np.percentile(d, 97.5, 0).round(4).tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="desed")
    ap.add_argument("--models", default="BEATs,ATST-F,fpasst,M2D,ASIT,frame_mn10,frame_mn06")
    ap.add_argument("--pct", type=float, default=0.2)
    ap.add_argument("--calibrate", action="store_true", help="per-member, per-class Platt scaling before every arm")
    ap.add_argument("--sebb", action="store_true", help="add cSEBB post-processing arms")
    ap.add_argument("--ablate", action="store_true", help="add fusion-recipe ablation arms")
    ap.add_argument("--gap", action="store_true", help="add averaging with gap bridging")
    ap.add_argument("--pp", action="store_true", help="add averaging with jointly tuned gap filling and minimum duration")
    ap.add_argument("--classwise", action="store_true", help="add arms with per-class post-processing")
    ap.add_argument("--fold-seed", type=int, default=0, help="alternative random two-fold split")
    ap.add_argument("--holdout-list", default=None,
                    help="file of recording ids scored once; every other clip is used only for tuning")
    ap.add_argument("--dev-split", default=None,
                    help="tune on this split (e.g. validate) and score the eval split once, instead of cross-fitting")
    ap.add_argument("--fixed-collar", type=float, default=None,
                    help="select every operating point at this collar and apply it at all collars")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = a.models.split(",")
    stack, names = zip(*[L.load_member(a.dataset, m) for m in models])
    names = names[0]
    keep = np.ones(len(names), bool)
    if a.dataset == "desed":
        seen = audioset_overlap()
        keep = np.array([n[:11] not in seen for n in names])
    names = [n for n, k in zip(names, keep) if k]
    classes = list(L.CLASS_MAPS[a.dataset])
    refs = L.load_refs(a.dataset, names)
    folds = np.array([L.fold_of(n, a.dataset, seed=a.fold_seed) for n in names])
    raw = np.stack([s[keep] for s in stack])
    n_test = len(names)
    if a.holdout_list:   # reorder so the held-out recordings come first, as fold 0
        hold = set(open(a.holdout_list).read().split())
        order = sorted(range(len(names)), key=lambda i: names[i].split("__")[0] not in hold)
        names = [names[i] for i in order]
        raw = raw[:, order]
        n_test = sum(n.split("__")[0] in hold for n in names)
        folds = np.array([0] * n_test + [1] * (len(names) - n_test))
    if a.dev_split:   # fold 0 = the untouched eval split, fold 1 = the development split used only for tuning
        dstack, dnames = zip(*[L.load_member(a.dataset, m, split=a.dev_split) for m in models])
        dnames = list(dnames[0])
        refs.update(L.load_refs(a.dataset, dnames, split=a.dev_split))
        names = names + dnames
        folds = np.array([0] * n_test + [1] * len(dnames))
        raw = np.concatenate([raw, np.stack(dstack)], axis=1)
    G["fixed_k"] = COLLARS.index(a.fixed_collar) if a.fixed_collar is not None else None
    G.update(models=models, names=names, classes=classes, refs=refs, pct=a.pct, folds=folds,
             yref=L.ref_frames(refs, names, classes))
    G["lg"] = {f: (np.stack([platt(raw[m], folds != f) for m in range(len(models))]) if a.calibrate else raw)
               for f in (0, 1)}

    curve_arms = [f"single:{m}" for m in models] + ["avg_prob", "avg_prob_nomed", "avg_logit"]
    if a.gap:
        curve_arms.append("avg_prob_gap")
    if a.pp:
        curve_arms.append("avg_prob_pp")
    if a.sebb:
        curve_arms += [f"sebb:single:{m}" for m in models] + ["sebb:avg_prob", "sebb:avg_logit"]
    with Pool(24) as p:
        grids = dict(p.map(decode_arm, [(f, n) for f in (0, 1) for n in curve_arms]))

    K = len(COLLARS)
    pooled = {n: np.zeros((K, len(names), len(classes), 3), np.int32) for n in curve_arms}
    chosen = {n: {} for n in curve_arms}
    G["member_cfg"] = {}
    for f in (0, 1):
        dev, test = folds != f, folds == f
        for n in curve_arms:
            for k in range(K):
                cfg = select(grids[(f, n)], dev, k if G.get("fixed_k") is None else G["fixed_k"])
                pooled[n][k][test] = grids[(f, n)][cfg][k][test]
                chosen[n][f"{COLLARS[k]}|f{f}"] = list(cfg)
                if n.startswith("single:"):
                    G["member_cfg"][(f, n[7:], k)] = cfg
                elif n.startswith("sebb:single:"):
                    G.setdefault("member_cfg_sebb", {})[(f, n[12:], k)] = cfg

    if a.classwise:
        cw_arms = [n for n in curve_arms if n.startswith(("single:", "sebb:single:")) or n in
                   ("avg_prob", "avg_prob_gap", "avg_prob_pp", "sebb:avg_prob")]
        G["member_cfg_cw"], G["member_cfg_sebb_cw"], cw_dev = {}, {}, {}
        for n in cw_arms:
            pooled["cw:" + n] = np.zeros((K, len(names), len(classes), 3), np.int32)
            chosen["cw:" + n] = {}
            for f in (0, 1):
                dev, test = folds != f, folds == f
                for k in range(K):
                    per_class, dev_f1 = select_cw(grids[(f, n)], dev, k)
                    for c, g in enumerate(per_class):
                        pooled["cw:" + n][k][test, c] = grids[(f, n)][g][k][test, c]
                    chosen["cw:" + n][f"{COLLARS[k]}|f{f}"] = [list(g) for g in per_class]
                    cw_dev[(n, f, k)] = float(dev_f1.mean())
                    if n.startswith("single:"):
                        G["member_cfg_cw"][(f, n[7:], k)] = per_class
                    elif n.startswith("sebb:single:"):
                        G["member_cfg_sebb_cw"][(f, n[12:], k)] = per_class
        for prefix, ref in [("single:", "cw:best_single")] + ([("sebb:single:", "cw:best_single_sebb")] if a.sebb else []):
            pooled[ref] = np.zeros_like(pooled["avg_prob"])
            for f in (0, 1):
                test = folds == f
                for k in range(K):
                    m = max(models, key=lambda m: cw_dev[(prefix + m, f, k)])
                    pooled[ref][k][test] = pooled["cw:" + prefix + m][k][test]

    # one decode setting for every member (best mean dev F1), so fusion gets no per-member thresholds
    G["shared_cfg"] = {}
    for f in (0, 1):
        dev = folds != f
        for k in range(K):
            ks = k if G.get("fixed_k") is None else G["fixed_k"]
            G["shared_cfg"][(f, k)] = max(grids[(f, f"single:{models[0]}")], key=lambda g: np.mean(
                [L.f1_from_counts(grids[(f, f"single:{m}")][g][ks][dev]) for m in models]))

    fusion_arms = ["vote", "wbf_median", "wbf_weighted", "wbf_shared"] + (["wbf_median_sebb"] if a.sebb else [])
    if a.ablate:
        fusion_arms += ["wbf_iou0.3", "wbf_iou0.5", "wbf_iou0.7", "wbf_union", "wbf_majority"]
    if a.classwise:
        fusion_arms += ["cw:wbf_median"] + (["cw:wbf_median_sebb"] if a.sebb else [])
    with Pool(6) as p:
        fused = dict(p.map(fusion_arm, [(f, n) for f in (0, 1) for n in fusion_arms]))
    for n in fusion_arms:
        pooled[n] = np.zeros((K, len(names), len(classes), 3), np.int32)
        chosen[n] = {}
        for f in (0, 1):
            test = folds == f
            out, ch = fused[(f, n)]
            pooled[n][:, test] = out[:, test]
            chosen[n].update({f"{COLLARS[k]}|f{f}": list(v) for k, v in ch.items()})

    # reference arms pick their member on the dev fold, so they are cross-fitted like the rest
    for prefix, ref in [("single:", "best_single")] + ([("sebb:single:", "best_single_sebb")] if a.sebb else []):
        best = np.zeros_like(pooled["avg_prob"])
        for f in (0, 1):
            dev, test = folds != f, folds == f
            for k in range(K):
                def dev_f1(m):
                    g = grids[(f, prefix + m)]
                    ks = k if G.get("fixed_k") is None else G["fixed_k"]
                    return L.f1_from_counts(g[select(g, dev, ks)][ks][dev])
                m = max(models, key=dev_f1)
                best[k][test] = pooled[prefix + m][k][test]
                chosen.setdefault(ref, {})[f"{COLLARS[k]}|f{f}"] = m
        pooled[ref] = best

    if a.dev_split or a.holdout_list:   # report the untouched evaluation clips only
        pooled = {n: c[:, :n_test] for n, c in pooled.items()}
        names = names[:n_test]
        G["names"] = names
    refs_ = ["best_single", "avg_prob"] + (["best_single_sebb", "sebb:avg_prob"] if a.sebb else []) \
        + (["avg_prob_gap"] if a.gap else []) + (["avg_prob_pp"] if a.pp else []) \
        + ((["cw:avg_prob", "cw:best_single"] + (["cw:avg_prob_gap"] if a.gap else [])
            + (["cw:avg_prob_pp"] if a.pp else [])
            + (["cw:sebb:avg_prob"] if a.sebb else [])) if a.classwise else [])
    ci = bootstrap(pooled, refs_)
    table = {n: {"f1": [round(float(L.f1_from_counts(c[k])), 4) for k in range(K)], **ci[n]}
             for n, c in pooled.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out.replace(".json", "_counts.npz"), names=np.array(names), classes=np.array(classes),
                        **{n.replace(":", "__"): c for n, c in pooled.items()})
    with open(a.out, "w") as fh:
        json.dump({"dataset": a.dataset, "pct": a.pct, "calibrate": a.calibrate, "collars": COLLARS,
                   "n_clips": len(names), "models": models, "table": table, "chosen": chosen}, fh, indent=1)

    print(f"{'arm':18s}" + "".join(f"{c:>9g}" for c in COLLARS))
    for n, r in sorted(table.items(), key=lambda t: -t[1]["f1"][3]):
        print(f"{n:18s}" + "".join(f"{v:9.4f}" for v in r["f1"]))
    for ref in refs_:
        print(f"\npaired delta vs {ref}, 95% CI:")
        for n, r in sorted(table.items(), key=lambda t: -t[1]["f1"][3]):
            lo, hi = r[f"d_{ref}"]
            print(f"{n:18s}" + "".join(f" [{x:+.3f},{y:+.3f}]" for x, y in zip(lo, hi)))


if __name__ == "__main__":
    main()
