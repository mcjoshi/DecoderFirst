"""Shared pieces: class maps, references, decoding, ensembling arms and collar-based event metrics."""
import csv
import hashlib
import os

import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SED_DATA", os.path.join(REPO, "data"))   # datasets, caches, checkpoints
LISTS = os.path.join(REPO, "lists")
RESULTS = os.environ.get("SED_RESULTS", os.path.join(REPO, "results"))
HOP = 0.01          # decode grid (s)
N_FRAMES = 1000     # 10 s clip on the decode grid
SRC_HOP = 0.04      # PretrainedSED output resolution

# Target class -> AudioSet-Strong class indices (max over member logits).
CLASS_MAPS = {
    "desed": {
        "Alarm_bell_ringing": [6, 7, 26, 27, 30, 53, 59, 135, 162, 300, 339, 378, 379],
        "Blender": [37],
        "Cat": [64, 65, 244, 283],
        "Dishes": [121, 128],
        "Dog": [20, 44, 130, 187, 204, 265, 421, 443],
        "Electric_shaver_toothbrush": [144, 145, 392],
        "Frying": [173, 331],
        "Running_water": [23, 329, 413, 414],
        "Speech": [78, 103, 157, 238, 254, 353],
        "Vacuum_cleaner": [405],
    },
    "tutsed": {
        "brakes squeaking": [1, 359, 360, 389, 396],
        "car": [58, 60, 249],
        "children": [77, 78, 79, 80],
        "large vehicle": [50, 197, 399],
        "people speaking": [103, 157, 206, 238, 353],
        "people walking": [323, 411],
    },
    "urbansed": {
        "air_conditioner": [2],
        "car_horn": [3, 407],
        "children_playing": [78, 79, 80],
        "dog_bark": [20, 44, 130],
        "drilling": [137],
        "engine_idling": [148, 197, 214, 231, 243],
        "gun_shot": [57, 190, 235],
        "jackhammer": [219],
        "siren": [9, 91, 163, 272, 330],
        "street_music": [253],
    },
}


def fold_of(name, dataset, k=2, seed=0):
    """Deterministic clip fold; DESED clips sharing a YouTube id land in the same fold. `seed` gives other splits."""
    key = name[:11] if dataset == "desed" else (name.split("__")[0] if dataset == "tutsed" else name)
    if seed:
        key = f"{key}#{seed}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % k


# ---------------------------------------------------------------- references

def merge_same_class(events):
    """Merge overlapping or touching same-class intervals (the scoring toolkits reject them)."""
    out = []
    for on, off in sorted(events):
        if out and on <= out[-1][1]:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append([on, off])
    return [tuple(e) for e in out]


def load_refs(dataset, names, split="test"):
    """Return {name: {class: [(on, off), ...]}} with same-class overlaps merged."""
    classes = list(CLASS_MAPS[dataset])
    refs = {n: {c: [] for c in classes} for n in names}
    if dataset == "desed":
        with open(f"{DATA}/datasets/dataset/metadata/eval/public.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["filename"] in refs and r["event_label"]:
                    refs[r["filename"]][r["event_label"]].append((float(r["onset"]), float(r["offset"])))
    elif dataset == "tutsed":
        with open(f"{DATA}/datasets/tutsed/clips_labels.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["filename"] in refs and r["event_label"]:
                    refs[r["filename"]][r["event_label"]].append((float(r["onset"]), float(r["offset"])))
    elif dataset == "urbansed":
        ann = f"{DATA}/datasets/URBAN-SED_v2.0.0/annotations/{split}"
        for n in names:
            with open(os.path.join(ann, n[:-4] + ".txt")) as f:
                for line in f:
                    on, off, c = line.split()
                    refs[n][c].append((float(on), float(off)))
    for n in refs:
        for c in classes:
            refs[n][c] = merge_same_class(refs[n][c])
    return refs


def ref_frames(refs, names, classes):
    """Binary frame targets on the decode grid, shape (N, C, T)."""
    y = np.zeros((len(names), len(classes), N_FRAMES), bool)
    for i, n in enumerate(names):
        for j, c in enumerate(classes):
            for on, off in refs[n][c]:
                y[i, j, int(round(on / HOP)):int(round(off / HOP))] = True
    return y


# ---------------------------------------------------------------- member outputs

def load_member(dataset, model, shift_ms=0.0, split="eval"):
    """Class-mapped logits on the 10 ms grid, shape (N, C, T), plus clip names."""
    tag = f"s{shift_ms:g}"
    z = np.load(f"{DATA}/cache/{dataset}_{split}/{model}_{tag}.npz")
    src = z["logits"].astype(np.float32)
    if "mapped" in z.files:   # already target-class logits (learned class map)
        mapped = src
    else:
        mapped = np.stack([src[:, idx].max(1) for idx in CLASS_MAPS[dataset].values()], 1)   # (N, C, 250)
    hop = float(z["frame_s"]) if "frame_s" in z.files else SRC_HOP
    t_src = hop / 2 + hop * np.arange(mapped.shape[-1]) - shift_ms / 1000
    t_dst = HOP / 2 + HOP * np.arange(N_FRAMES)
    out = np.empty(mapped.shape[:2] + (N_FRAMES,), np.float32)
    for i in range(mapped.shape[0]):
        for j in range(mapped.shape[1]):
            out[i, j] = np.interp(t_dst, t_src, mapped[i, j])
    return out, [str(n) for n in z["names"]]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- decoding

def runs(mask):
    """Start/stop frame indices of True runs in a 1-D boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def decode(prob, thr, med, gap=0, min_len=0):
    """Median-filter probabilities, threshold, optionally bridge gaps of <= `gap` frames, drop events shorter than
    `min_len` frames, and extract events."""
    if med > 1:
        prob = scipy.ndimage.median_filter(prob, size=(1, 1, med), mode="nearest")
    act = prob > thr
    out = []
    for i in range(prob.shape[0]):
        clip = {}
        for j in range(prob.shape[1]):
            s, e = runs(act[i, j])
            if gap > 0 and len(s) > 1:
                keep = np.concatenate([[True], s[1:] - e[:-1] > gap])
                s, e = s[keep], np.concatenate([e[:-1][keep[1:]], e[-1:]])
            if min_len > 0:
                s, e = s[e - s >= min_len], e[e - s >= min_len]
            clip[j] = [(a * HOP, b * HOP, float(prob[i, j, a:b].mean())) for a, b in zip(s, e)]
        out.append(clip)
    return out


SEBB_FILTERS = (0.32, 0.48, 0.64)
SEBB_MERGES = [(a, np.inf) for a in (0.15, 0.2, 0.3)] + [(np.inf, r) for r in (1.5, 2.0, 3.0)]


def sebb_candidates(prob):
    """cSEBB (Ebbers et al. 2024) candidates for every (filter, merge) setting of its default tuning grid."""
    from sebbs.change_detection import change_detection
    from sebbs.csebbs import _merge_segments
    ts = HOP * np.arange(N_FRAMES + 1)
    out = {(f, a, r): [] for f in SEBB_FILTERS for a, r in SEBB_MERGES}
    for i in range(prob.shape[0]):
        for f in SEBB_FILTERS:
            cd = change_detection(prob[i].T.astype(np.float64), ts, f)
            for a, r in SEBB_MERGES:
                clip = {}
                for j in range(prob.shape[1]):
                    on, off, conf = _merge_segments(*cd[j], threshold_abs=a, threshold_rel=r)
                    clip[j] = [(float(x), float(y), float(c)) for x, y, c in zip(on, off, conf)]
                out[(f, a, r)].append(clip)
    return out


def threshold_events(cands, thr):
    return [{j: [e for e in evs if e[2] > thr] for j, evs in clip.items()} for clip in cands]


def events_to_frames(events, n_classes):
    y = np.zeros((len(events), n_classes, N_FRAMES), bool)
    for i, clip in enumerate(events):
        for j, evs in clip.items():
            for on, off, _ in evs:
                y[i, j, int(round(on / HOP)):int(round(off / HOP))] = True
    return y


def frame_vote(member_events, n_classes, min_votes):
    """DOVER-style fusion: a frame is active when at least `min_votes` decoded members say so."""
    votes = sum(events_to_frames(ev, n_classes).astype(np.int16) for ev in member_events)
    act = votes >= min_votes
    out = []
    for i in range(act.shape[0]):
        clip = {}
        for j in range(n_classes):
            s, e = runs(act[i, j])
            clip[j] = [(a * HOP, b * HOP, 1.0) for a, b in zip(s, e)]
        out.append(clip)
    return out


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return inter / (max(a[1], b[1]) - min(a[0], b[0])) if inter > 0 else 0.0


def wbf_1d(member_events, n_classes, iou_thr, min_votes, edge="median", score="votes"):
    """Event-level fusion: cluster decoded events across members by IoU, keep voted clusters, fuse edges.
    Fused confidence is the vote count, or the summed member confidence with score="sumconf"."""
    out = []
    for i in range(len(member_events[0])):
        clip = {}
        for j in range(n_classes):
            pool = sorted(((ev, m) for m, me in enumerate(member_events) for ev in me[i].get(j, [])),
                          key=lambda t: -t[0][2])
            clusters = []   # [fused (on, off), [(ev, member), ...]]
            for ev, m in pool:
                best, best_iou = None, iou_thr
                for cl in clusters:
                    v = iou(cl[0], ev)
                    if v >= best_iou and all(mm != m for _, mm in cl[1]):
                        best, best_iou = cl, v
                if best is None:
                    clusters.append([(ev[0], ev[1]), [(ev, m)]])
                else:
                    best[1].append((ev, m))
                    ons = np.array([e[0] for e, _ in best[1]])
                    offs = np.array([e[1] for e, _ in best[1]])
                    w = np.array([e[2] for e, _ in best[1]])
                    if edge == "median":
                        best[0] = (float(np.median(ons)), float(np.median(offs)))
                    elif edge == "union":
                        best[0] = (float(ons.min()), float(offs.max()))
                    else:
                        best[0] = (float((ons * w).sum() / w.sum()), float((offs * w).sum() / w.sum()))
            conf = (lambda cl: len(cl[1])) if score == "votes" else (lambda cl: sum(e[2] for e, _ in cl[1]))
            clip[j] = merge_conf([(cl[0][0], cl[0][1], conf(cl)) for cl in clusters if len(cl[1]) >= min_votes])
        out.append(clip)
    return out


def merge_conf(evs):
    out = []
    for on, off, c in sorted(evs):
        if out and on <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], off), max(out[-1][2], c))
        else:
            out.append((on, off, c))
    return out


# ---------------------------------------------------------------- metrics

def match_count(ref, est, collar, pct):
    """Maximum one-to-one matching; onset within `collar`, offset within max(collar, pct * ref duration)."""
    if not ref or not est:
        return 0
    r = np.array(ref)
    e = np.array([x[:2] for x in est])
    on_ok = np.abs(r[:, None, 0] - e[None, :, 0]) <= collar + 1e-9
    off_tol = np.maximum(collar, pct * (r[:, 1] - r[:, 0]))
    off_ok = np.abs(r[:, None, 1] - e[None, :, 1]) <= off_tol[:, None] + 1e-9
    ok = on_ok & off_ok
    if not ok.any():
        return 0
    rows, cols = linear_sum_assignment(-ok.astype(np.float64))
    return int(ok[rows, cols].sum())


def event_counts(events, refs, names, classes, collars, pct=0.2):
    """Per-clip, per-class (tp, n_est, n_ref) for each collar: array (len(collars), N, C, 3)."""
    out = np.zeros((len(collars), len(names), len(classes), 3), np.int32)
    for i, n in enumerate(names):
        for j, c in enumerate(classes):
            ref, est = refs[n][c], events[i].get(j, [])
            out[:, i, j, 1] = len(est)
            out[:, i, j, 2] = len(ref)
            for k, col in enumerate(collars):
                out[k, i, j, 0] = match_count(ref, est, col, pct)
    return out


def f1_from_counts(counts, macro=True):
    """counts (..., N, C, 3) -> event F1 (macro over classes as in DCASE, or micro)."""
    s = counts.sum(-3)                                  # (..., C, 3)
    if macro:
        tp, ne, nr = s[..., 0], s[..., 1], s[..., 2]
        f = np.where(ne + nr > 0, 2 * tp / np.maximum(ne + nr, 1), 0.0)
        return f.mean(-1)
    s = s.sum(-2)
    return 2 * s[..., 0] / np.maximum(s[..., 1] + s[..., 2], 1)


def segment_f1(pred_frames, ref_frames_, seg=100):
    """Macro segment F1 over 1 s segments (seg frames on the 10 ms grid)."""
    n, c, t = pred_frames.shape
    p = pred_frames.reshape(n, c, t // seg, seg).any(-1)
    r = ref_frames_.reshape(n, c, t // seg, seg).any(-1)
    tp = (p & r).sum((0, 2))
    fp = (p & ~r).sum((0, 2))
    fn = (~p & r).sum((0, 2))
    return float(np.mean(np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)))
