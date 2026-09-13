"""Cache target-class logits of PANNs CNN14 (decision-level max, AudioSet weak labels) over 10 s clips.

Target classes use the same AudioSet class names as the AudioSet-Strong members, matched by name in PANNs' 527 labels.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "2")

import argparse
import sys

import librosa
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "PretrainedSED"))
from data_util.audioset_classes import as_strong_train_classes  # noqa: E402

import sedlib as L  # noqa: E402

SR = 32_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=f"{L.DATA}/checkpoints/Cnn14_DecisionLevelMax_mAP=0.385.pth")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(12)

    from panns_inference import labels
    from panns_inference.models import Cnn14_DecisionLevelMax
    model = Cnn14_DecisionLevelMax(sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64, fmin=50,
                                   fmax=14000, classes_num=527, interpolate_mode="nearest")
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval()

    index = {n: i for i, n in enumerate(labels)}
    # the three AudioSet-Strong names that differ from the 527-class release
    for strong, weak in [("Blender, food processor", "Blender"), ("Pant (dog)", "Pant"),
                         ("Vehicle horn, car horn, honking, toot", "Vehicle horn, car horn, honking")]:
        index[strong] = index[weak]
    cmap = []
    for target, strong_idx in L.CLASS_MAPS[a.dataset].items():
        idx = [index[as_strong_train_classes[k]] for k in strong_idx if as_strong_train_classes[k] in index]
        assert idx, f"no PANNs label for {target}"
        cmap.append(idx)

    names = sorted(n for n in os.listdir(a.audio_dir) if n.endswith(".wav") and not n.startswith("._"))
    out = []
    with torch.no_grad():
        for i in range(0, len(names), a.batch):
            xs = []
            for n in names[i:i + a.batch]:
                x, _ = librosa.load(os.path.join(a.audio_dir, n), sr=SR, mono=True)
                xs.append(np.pad(x[:10 * SR], (0, max(0, 10 * SR - len(x)))))
            fw = model(torch.from_numpy(np.stack(xs)))["framewise_output"].numpy()     # (B, T, 527)
            p = np.stack([fw[:, :, idx].max(-1) for idx in cmap], 1).clip(1e-6, 1 - 1e-6)  # (B, C, T)
            out.append(np.log(p / (1 - p)))
    logits = np.concatenate(out).astype(np.float16)
    frame_s = 10.0 / logits.shape[-1]
    np.savez(a.out, logits=logits, names=np.array(names), shift_ms=0.0, frame_s=frame_s, mapped=1)
    print(a.out, logits.shape, f"frame_s={frame_s:.4f}")


if __name__ == "__main__":
    main()
