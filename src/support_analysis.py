"""How many members support the events each ensemble emits? Splits emitted events into true and false positives
at one collar and counts, for each, how many members' own decoded events overlap it."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "2")

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import ensemble_arms as E
import sedlib as L


def matched(ref, est, collar, pct=0.2):
    """Boolean per estimated event: matched one-to-one to a reference under the collar rule."""
    hit = np.zeros(len(est), bool)
    if not ref or not est:
        return hit
    r, e = np.array(ref), np.array([x[:2] for x in est])
    ok = (np.abs(r[:, None, 0] - e[None, :, 0]) <= collar + 1e-9) & \
         (np.abs(r[:, None, 1] - e[None, :, 1]) <= np.maximum(collar, pct * (r[:, 1] - r[:, 0]))[:, None] + 1e-9)
    rows, cols = linear_sum_assignment(-ok.astype(float))
    hit[cols[ok[rows, cols]]] = True
    return hit


def pairs(ref, est, collar, pct=0.2):
    """(reference index, estimate index) of the one-to-one matching under the collar rule."""
    if not ref or not est:
        return []
    r, e = np.array(ref), np.array([x[:2] for x in est])
    ok = (np.abs(r[:, None, 0] - e[None, :, 0]) <= collar + 1e-9) & \
         (np.abs(r[:, None, 1] - e[None, :, 1]) <= np.maximum(collar, pct * (r[:, 1] - r[:, 0]))[:, None] + 1e-9)
    rows, cols = linear_sum_assignment(-ok.astype(float))
    return [(int(a), int(b)) for a, b in zip(rows, cols) if ok[a, b]]


def support(ev, member_events, i, c):
    return sum(any(min(ev[1], b) - max(ev[0], a) > 0 for a, b, _ in me[i].get(c, [])) for me in member_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="desed")
    ap.add_argument("--result", required=True, help="ensemble_arms JSON whose chosen configs are reused")
    ap.add_argument("--collar", type=float, default=0.2)
    ap.add_argument("--sebb", action="store_true", help="also analyse the cSEBB-decoded arms")
    ap.add_argument("--pp-result", default=None, help="ensemble_arms JSON with --pp whose averaging+pp config is added")
    ap.add_argument("--singles", default="", help="members whose own decoded events are analysed as extra arms")
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
    classes = list(L.CLASS_MAPS[a.dataset])
    refs = L.load_refs(a.dataset, names)
    folds = np.array([L.fold_of(n, a.dataset) for n in names])
    M = len(models)
    key = lambda f: f"{a.collar:g}|f{f}"

    singles = [m for m in a.singles.split(",") if m]
    arm_names = ["avg_prob", "wbf_median"] + (["sebb:avg_prob", "wbf_median_sebb"] if a.sebb else []) \
        + [f"single:{m}" for m in singles] + ([f"sebb:single:{m}" for m in singles] if a.sebb else []) \
        + (["avg_prob_pp"] if a.pp_result else [])
    ch_pp = json.load(open(a.pp_result))["chosen"] if a.pp_result else None
    stats = {n: {"tp": [], "fp": []} for n in arm_names}
    for f in (0, 1):
        test = np.flatnonzero(folds == f)
        sub = lg[:, test]
        mev = [L.decode(L.sigmoid(sub[m]), *ch[f"single:{model}"][key(f)]) for m, model in enumerate(models)]
        t, w = ch["avg_prob"][key(f)]
        arms = {"avg_prob": L.decode(L.sigmoid(sub).mean(0), t, w),
                "wbf_median": L.wbf_1d(mev, len(classes), *ch["wbf_median"][key(f)], "median")}
        if a.sebb:   # the same two fusion levels under change-point (cSEBB) decoding
            cfg = ch["sebb:avg_prob"][key(f)]
            arms["sebb:avg_prob"] = L.threshold_events(L.sebb_candidates(L.sigmoid(sub).mean(0))[tuple(cfg[:3])], cfg[3])
            smev = []
            for m, model in enumerate(models):
                c_ = ch[f"sebb:single:{model}"][key(f)]
                smev.append(L.threshold_events(L.sebb_candidates(L.sigmoid(sub[m]))[tuple(c_[:3])], c_[3]))
            arms["wbf_median_sebb"] = L.wbf_1d(smev, len(classes), *ch["wbf_median_sebb"][key(f)], "median")
            for m in singles:
                arms[f"sebb:single:{m}"] = smev[models.index(m)]
        for m in singles:
            arms[f"single:{m}"] = mev[models.index(m)]
        if ch_pp:   # averaging with gap filling and minimum duration: the decoder-side repair of fragmentation
            arms["avg_prob_pp"] = L.decode(L.sigmoid(sub).mean(0), *ch_pp["avg_prob_pp"][key(f)])
        for arm, evs in arms.items():
            for ii, i in enumerate(test):
                for c, cname in enumerate(classes):
                    est = evs[ii].get(c, [])
                    ref = refs[names[i]][cname]
                    hit = matched(ref, est, a.collar)
                    covered = set()   # references overlapped by a matched estimate
                    for e, h in zip(est, hit):
                        if h:
                            covered |= {r for r in ref if min(e[1], r[1]) - max(e[0], r[0]) > 0}
                    for r in ref:   # fragmentation: emitted events overlapping each reference
                        k_ov = sum(min(e[1], r[1]) - max(e[0], r[0]) > 0 for e in est)
                        stats[arm].setdefault("frag", []).append(k_ov)
                    for rj, ej in pairs(ref, est, 1.0):   # per reference, for the analysis on references every arm matches
                        stats[arm].setdefault("pairs", {})[(int(i), c, rj)] = (abs(est[ej][0] - ref[rj][0]),
                                                                              abs(est[ej][1] - ref[rj][1]))
                    hit1 = matched(ref, est, 1.0)   # one-to-one matching at a 1 s collar: boundary error of matched pairs
                    if hit1.any():
                        rr = np.array(ref)
                        for e, h in zip(est, hit1):
                            if h:
                                j = int(np.argmin(np.abs(rr[:, 0] - e[0]) + np.abs(rr[:, 1] - e[1])))
                                stats[arm].setdefault("m_on", []).append(abs(e[0] - rr[j, 0]))
                                stats[arm].setdefault("m_off", []).append(abs(e[1] - rr[j, 1]))
                    for e in est:   # boundary error against the most-overlapping reference
                        ov = [(min(e[1], r[1]) - max(e[0], r[0]), r) for r in ref]
                        ov = [x for x in ov if x[0] > 0]
                        if ov:
                            r = max(ov)[1]
                            stats[arm].setdefault("onset_err", []).append(abs(e[0] - r[0]))
                            stats[arm].setdefault("offset_err", []).append(abs(e[1] - r[1]))
                    for e, h in zip(est, hit):
                        stats[arm]["tp" if h else "fp"].append(support(e, mev, ii, c))
                        if not h:
                            ov = [r for r in ref if min(e[1], r[1]) - max(e[0], r[0]) > 0]
                            kind = "spurious" if not ov else ("duplicate" if all(r in covered for r in ov) else "mislocalized")
                            stats[arm].setdefault("fp_kind", {}).setdefault(kind, 0)
                            stats[arm]["fp_kind"][kind] += 1
    common = set.intersection(*[set(s.get("pairs", {})) for s in stats.values()])
    out = {}
    for arm, s in stats.items():
        pr = s.pop("pairs", {})
        v = np.array([pr[k] for k in sorted(common)]).reshape(-1, 2)
        s_common = {"n": int(len(v)), "on": round(float(np.median(v[:, 0])), 3) if len(v) else None,
                    "off": round(float(np.median(v[:, 1])), 3) if len(v) else None}
        out[arm] = {"fp_kind": s.pop("fp_kind", {}), "common": s_common}
        fr = np.array(s.pop("frag", []))
        out[arm]["frag"] = {"refs": int(len(fr)), "split_2plus": round(float((fr >= 2).mean()), 3),
                            "missed": round(float((fr == 0).mean()), 3)}
        for side in ("m_on", "m_off"):
            v = np.array(s.pop(side, []))
            out[arm][side] = {"n": int(len(v)), "median": round(float(np.median(v)), 3) if len(v) else None}
        for side in ("onset_err", "offset_err"):
            v = np.array(s.pop(side, []))
            out[arm][side] = {"n": int(len(v)), "median": round(float(np.median(v)), 3),
                              "within_100ms": round(float((v <= 0.1).mean()), 3)}
        for kind in ("tp", "fp"):
            v = np.array(s[kind])
            out[arm][kind] = {"n": int(len(v)), "hist": np.bincount(v, minlength=M + 1).tolist(),
                              "share_support_le1": round(float((v <= 1).mean()), 3) if len(v) else None}
    json.dump({"dataset": a.dataset, "collar": a.collar, "members": M, "stats": out}, open(a.out, "w"), indent=1)
    for arm, s in out.items():
        print(arm, "common", s["common"], "frag", s["frag"], "matched1s on/off", s["m_on"], s["m_off"], "onset", s["onset_err"]["median"])


if __name__ == "__main__":
    main()
