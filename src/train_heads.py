"""Train a BiGRU SED head on a frozen member's frame logits (URBAN-SED train split), select on validation,
and write target-class logits for the validation and test splits."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "4")

import argparse

import numpy as np
import torch
import torch.nn as nn

import sedlib as L
from adapt_members import frame_targets


class Head(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(2 * hidden, n_out)

    def forward(self, x):            # x: (B, T, F)
        h, _ = self.gru(self.proj(x))
        return self.out(h)           # (B, T, C) logits


def load_split(d, model, split):
    z = np.load(f"{L.DATA}/cache/{d}_{split}/{model}_s0.npz")
    return z["logits"].astype(np.float32).transpose(0, 2, 1), [str(n) for n in z["names"]]   # (N, T, 447)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="urbansed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    classes = list(L.CLASS_MAPS[a.dataset])

    xtr, ntr = load_split(a.dataset, a.model, "train")
    xva, nva = load_split(a.dataset, a.model, "validate")
    mu, sd = xtr.mean((0, 1)), xtr.std((0, 1)) + 1e-6
    ytr = frame_targets(a.dataset, ntr, classes).transpose(0, 2, 1).astype(np.float32)
    refs_va = L.load_refs(a.dataset, nva, split="validate")
    t = 0.02 + 0.04 * np.arange(xva.shape[1])
    yva = np.zeros((len(nva), xva.shape[1], len(classes)), np.float32)
    for i, n in enumerate(nva):
        for j, c in enumerate(classes):
            for on, off in refs_va[n][c]:
                yva[i, :, j] = np.maximum(yva[i, :, j], (t >= on) & (t < off))

    xtr_t = torch.from_numpy((xtr - mu) / sd)
    xva_t = torch.from_numpy((xva - mu) / sd)
    ytr_t, yva_t = torch.from_numpy(ytr), torch.from_numpy(yva)
    net = Head(xtr.shape[2], len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state = np.inf, None
    for ep in range(a.epochs):
        net.train()
        for idx in np.array_split(rng.permutation(len(ntr)), len(ntr) // 32):
            opt.zero_grad()
            loss = loss_fn(net(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(net(xva_t[i:i + 256]), yva_t[i:i + 256]).item()
                                for i in range(0, len(nva), 256)]))
        print(f"{a.model} seed {a.seed} epoch {ep} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    tag = f"{a.model}-gru{a.seed}"
    for split in ("validate", "eval"):
        x, names = (xva, nva) if split == "validate" else load_split(a.dataset, a.model, "eval")
        with torch.no_grad():
            lg = torch.cat([net(torch.from_numpy((x[i:i + 256] - mu) / sd)) for i in range(0, len(x), 256)])
        path = f"{L.DATA}/cache/{a.dataset}_{split}/{tag}_s0.npz"
        np.savez(path, logits=lg.numpy().transpose(0, 2, 1).astype(np.float16), names=np.array(names),
                 shift_ms=0.0, frame_s=0.04, mapped=1)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
