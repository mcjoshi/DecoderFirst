"""Figure: one DESED clip, member posteriors, their average, and the events each ensemble emits."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "2")

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ensemble_arms as E  # noqa: E402
import sedlib as L  # noqa: E402
from support_analysis import matched  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--classes", default="Frying,Vacuum_cleaner,Blender,Running_water,Electric_shaver_toothbrush")
    ap.add_argument("--clip", default=None, help="clip name to draw; otherwise list candidates")
    ap.add_argument("--klass", default=None)
    ap.add_argument("--collar", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = json.load(open(a.result))
    models, ch = d["models"], d["chosen"]
    stack, names = zip(*[L.load_member("desed", m) for m in models])
    names = names[0]
    seen = E.audioset_overlap()
    keep = np.array([n[:11] not in seen for n in names])
    names = [n for n, k in zip(names, keep) if k]
    lg = np.stack([s[keep] for s in stack])
    classes = list(L.CLASS_MAPS["desed"])
    refs = L.load_refs("desed", names)
    folds = np.array([L.fold_of(n, "desed") for n in names])
    key = lambda f: f"{a.collar:g}|f{f}"

    def ensembles(i):
        f = folds[i]
        sub = lg[:, i:i + 1]
        mev = [L.decode(L.sigmoid(sub[m]), *ch[f"single:{mm}"][key(f)]) for m, mm in enumerate(models)]
        t, w = ch["avg_prob"][key(f)]
        return mev, L.decode(L.sigmoid(sub).mean(0), t, w)[0], L.wbf_1d(mev, len(classes), *ch["wbf_median"][key(f)], "median")[0], (t, w)

    if a.clip is None:
        for i, n in enumerate(names):
            for cname in a.classes.split(","):
                c = classes.index(cname)
                ref = refs[n][cname]
                if not 1 <= len(ref) <= 2:
                    continue
                _, avg, fus, _ = ensembles(i)
                ha, hf = matched(ref, avg.get(c, []), a.collar), matched(ref, fus.get(c, []), a.collar)
                if hf.sum() == len(ref) and ha.sum() < len(ref) and len(avg.get(c, [])) >= 2:
                    print(n, cname, "avg events", len(avg[c]), "matched", int(ha.sum()), "| fusion", len(fus[c]), "matched", int(hf.sum()))
        return

    i, c = names.index(a.clip), classes.index(a.klass)
    mev, avg, fus, (t, w) = ensembles(i)
    p = L.sigmoid(lg[:, i, c])
    tt = L.HOP / 2 + L.HOP * np.arange(L.N_FRAMES)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.45, 1.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]})
    for m in range(len(models)):
        ax.plot(tt, p[m], color="#9a9a9a", lw=0.6, label="members" if m == 0 else None)
    ax.plot(tt, p.mean(0), color="#eb6834", lw=1.4, label="average")
    ax.axhline(t, color="#eb6834", lw=0.6, ls="--", label="threshold")
    ax.set_ylabel("posterior")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=3)
    rows = [("reference", [(on, off) for on, off in refs[a.clip][a.klass]], "#1d1d1d"),
            ("averaging", [(on, off) for on, off, _ in avg.get(c, [])], "#eb6834"),
            ("event fusion", [(on, off) for on, off, _ in fus.get(c, [])], "#2a78d6")]
    for r, (label, evs, color) in enumerate(rows):
        for on, off in evs:
            bx.add_patch(plt.Rectangle((on, 2 - r - 0.35), off - on, 0.7, color=color, lw=0))
    bx.set_yticks([2, 1, 0])
    bx.set_yticklabels(["reference", "averaging", "event fusion"], fontsize=7)
    bx.set_ylim(-0.6, 2.5)
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("time (s)")
    fig.tight_layout(pad=0.3, h_pad=0.2)
    fig.savefig(a.out, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
