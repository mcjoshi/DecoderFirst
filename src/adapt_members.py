"""Learned class map: per-frame logistic heads from 447 AudioSet-Strong logits to target classes, fit on the train split."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "2")

import argparse
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression

import sedlib as L

A = {}


def frame_targets(dataset, names, classes, n_frames=250, hop=0.04):
    refs = L.load_refs(dataset, names, split="train")   # the split argument is used by URBAN-SED only
    t = hop / 2 + hop * np.arange(n_frames)
    y = np.zeros((len(names), len(classes), n_frames), bool)
    for i, n in enumerate(names):
        for j, c in enumerate(classes):
            for on, off in refs[n][c]:
                y[i, j] |= (t >= on) & (t < off)
    return y


def fit(model):
    d = A["dataset"]
    classes = list(L.CLASS_MAPS[d])
    if d == "tutsed":   # heads are fitted on the development recordings only; evaluation recordings stay untouched
        tr = np.load(f"{L.DATA}/cache/{d}_eval/{model}_s0.npz")
        dev = set(open(os.path.join(L.LISTS, "tutsed_dev_recordings.txt")).read().split())
        keep = np.array([str(n).split("__")[0] in dev for n in tr["names"]])
        names = [str(n) for n, k in zip(tr["names"], keep) if k]
        logits = tr["logits"][keep]
    else:
        tr = np.load(f"{L.DATA}/cache/{d}_train/{model}_s0.npz")
        names = [str(n) for n in tr["names"]]
        logits = tr["logits"]
    x = logits.astype(np.float32).transpose(0, 2, 1).reshape(-1, 447)
    y = frame_targets(d, names, classes).transpose(0, 2, 1).reshape(-1, len(classes))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), min(len(x), A["max_frames"]), replace=False)
    mu, sd = x[idx].mean(0), x[idx].std(0) + 1e-6

    heads = [LogisticRegression(C=A["C"], max_iter=300).fit((x[idx] - mu) / sd, y[idx, j]) for j in range(len(classes))]
    written = []
    for split in A["splits"]:
        src = f"{L.DATA}/cache/{d}_{split}/{model}_s0.npz"
        if not os.path.exists(src):
            continue
        te = np.load(src)
        xt = (te["logits"].astype(np.float32).transpose(0, 2, 1) - mu) / sd        # (N, 250, 447)
        out = np.stack([(xt @ h.coef_[0] + h.intercept_[0]) for h in heads], 1).astype(np.float16)
        path = f"{L.DATA}/cache/{d}_{split}/{model}-adapt_s0.npz"
        np.savez(path, logits=out, names=te["names"], shift_ms=0.0, frame_s=0.04, mapped=1)
        written.append(path)
    return f"{model}: {written}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="urbansed")
    ap.add_argument("--models", default="BEATs,ATST-F,fpasst,M2D,ASIT,frame_mn10,frame_mn06")
    ap.add_argument("--max-frames", type=int, default=400_000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--splits", default="eval,validate")
    a = ap.parse_args()
    A.update(dataset=a.dataset, max_frames=a.max_frames, C=a.C, splits=a.splits.split(","))
    with Pool(7) as p:
        for r in p.imap_unordered(fit, a.models.split(",")):
            print(r, flush=True)
