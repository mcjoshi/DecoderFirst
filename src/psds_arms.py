"""PSDS1/PSDS2 of averaging and event fusion under both decoders, cross-fitted, with a paired clip bootstrap."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "4")

import argparse
import json

import numpy as np
import scipy.ndimage
from sed_scores_eval.intersection_based import psds as psds_fn
from sed_scores_eval.intersection_based.psds import bootstrapped_psds
from sed_scores_eval.base_modules.scores import create_score_dataframe

import ensemble_arms as E
import sedlib as L

PSDS = {"psds1": dict(dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=None, alpha_ct=0., alpha_st=1.),
        "psds2": dict(dtc_threshold=0.1, gtc_threshold=0.1, cttc_threshold=0.3, alpha_ct=0.5, alpha_st=1.)}


def fused_scores(member_events, n_classes, iou_thr, min_votes, n_members):
    """Fused events scored by summed member confidence over the ensemble size (the WBF convention)."""
    out = []
    for clip in L.wbf_1d(member_events, n_classes, iou_thr, min_votes, "median", score="sumconf"):
        out.append({j: [(on, off, c / n_members) for on, off, c in evs] for j, evs in clip.items()})
    return out


def events_to_df(clip, classes, dur=10.0):
    edges = sorted({0.0, dur} | {round(x, 3) for evs in clip.values() for e in evs for x in e[:2]})
    ts = np.array(edges)
    mid = (ts[:-1] + ts[1:]) / 2
    s = np.zeros((len(mid), len(classes)))
    for j, evs in clip.items():
        for on, off, c in evs:
            sel = (mid > on) & (mid < off)
            s[sel, j] = np.maximum(s[sel, j], c)
    return create_score_dataframe(s, ts, classes)


def swept_fusion_scores(prob, meds, n_classes, iou_thr, min_votes, thrs):
    """Frame score = highest shared member threshold at which the fused output (fixed rho, v) covers the frame."""
    score = np.zeros(prob.shape[1:], np.float32)                 # (N, C, T)
    for t in thrs:
        mev = [L.decode(prob[m], t, meds[m]) for m in range(len(prob))]
        fr = L.events_to_frames(L.wbf_1d(mev, n_classes, iou_thr, min_votes, "median"), n_classes)
        score = np.where(fr, np.maximum(score, t), score)
    return score


def frames_to_df(prob, classes):
    return create_score_dataframe(prob.T, L.HOP * np.arange(L.N_FRAMES + 1), classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="desed")
    ap.add_argument("--result", required=True, help="ensemble_arms JSON (with --sebb) whose chosen configs are reused")
    ap.add_argument("--collar", type=float, default=0.2, help="collar whose tuned configs define each arm")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--arms", default=None, help="comma-separated subset of arms to score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.load(open(a.result))
    models, ch = d["models"], d["chosen"]
    stack, names = zip(*[L.load_member(a.dataset, m) for m in models])
    names = names[0]
    keep = np.ones(len(names), bool)
    if a.dataset == "desed":
        seen = E.audioset_overlap()
        keep = np.array([n[:11] not in seen for n in names])
    names = [n for n, k in zip(names, keep) if k]
    lg = np.stack([s[keep] for s in stack])
    if d.get("calibrate"):
        raise SystemExit("calibrated results need the fold-wise Platt maps; run on raw results only")
    classes = list(L.CLASS_MAPS[a.dataset])
    refs = L.load_refs(a.dataset, names)
    folds = np.array([L.fold_of(n, a.dataset) for n in names])
    M, C = len(models), len(classes)
    key = lambda f: f"{a.collar:g}|f{f}"

    scores = {k: {} for k in ("avg", "fusion", "fusion:sweep", "sebb:avg", "sebb:fusion")}
    sweep = np.round(np.arange(0.02, 0.81, 0.02), 2)
    for f in (0, 1):
        test = np.flatnonzero(folds == f)
        prob = L.sigmoid(lg[:, test])
        avg = prob.mean(0)
        _, med = ch["avg_prob"][key(f)]
        avg_f = scipy.ndimage.median_filter(avg, size=(1, 1, med), mode="nearest") if med > 1 else avg
        mev = [L.decode(prob[m], *ch[f"single:{mm}"][key(f)]) for m, mm in enumerate(models)]
        fus = fused_scores(mev, C, *ch["wbf_median"][key(f)], M)
        meds = [ch[f"single:{mm}"][key(f)][1] for mm in models]
        swept = swept_fusion_scores(prob, meds, C, *ch["wbf_median"][key(f)], sweep)
        sebb_avg = sfus = [{}] * len(test)
        if not a.arms or "sebb" in a.arms:
            cfg = ch["sebb:avg_prob"][key(f)]
            sebb_avg = L.sebb_candidates(avg)[tuple(cfg[:3])]      # all candidates; PSDS sweeps their confidence
            smev = []
            for m, mm in enumerate(models):
                c_ = ch[f"sebb:single:{mm}"][key(f)]
                smev.append(L.threshold_events(L.sebb_candidates(prob[m])[tuple(c_[:3])], c_[3]))
            sfus = fused_scores(smev, C, *ch["wbf_median_sebb"][key(f)], M)
        for ii, i in enumerate(test):
            n = names[i]
            scores["avg"][n] = frames_to_df(avg_f[ii], classes)
            scores["fusion"][n] = events_to_df(fus[ii], classes)
            scores["fusion:sweep"][n] = frames_to_df(swept[ii], classes)
            scores["sebb:avg"][n] = events_to_df(sebb_avg[ii], classes)
            scores["sebb:fusion"][n] = events_to_df(sfus[ii], classes)
        print("fold", f, "scored", flush=True)

    if a.arms:
        scores = {k: v for k, v in scores.items() if k in a.arms.split(",")}
    gt = {n: [(on, off, c) for c in classes for on, off in refs[n][c]] for n in names}
    dur = {n: 10.0 for n in names}
    out = {"dataset": a.dataset, "result": os.path.basename(a.result), "n_clips": len(names), "psds": {}, "boot": {}}
    for pname, kw in PSDS.items():
        boots = {}
        for arm, sc in scores.items():
            v = psds_fn(sc, gt, dur, **kw, max_efpr=100.)[0]
            out["psds"].setdefault(pname, {})[arm] = round(float(v), 4)
            if a.n_boot:
                b = bootstrapped_psds(sc, gt, dur, **kw, max_efpr=100., n_bootstrap_samples=a.n_boot)
                boots[arm] = np.array(b[0], float)   # psds values; resamples are identical across arms (seeded)
            print(pname, arm, round(float(v), 4), flush=True)
        for x, y in (("fusion", "avg"), ("fusion:sweep", "avg"), ("sebb:fusion", "sebb:avg"), ("sebb:avg", "avg")):
            if x in boots:
                dlt = boots[x] - boots[y]
                out["boot"].setdefault(pname, {})[f"{x}-{y}"] = [round(float(np.percentile(dlt, p)), 4) for p in (2.5, 50, 97.5)]
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["psds"]), json.dumps(out["boot"]))


if __name__ == "__main__":
    main()
