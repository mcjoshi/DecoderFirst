"""Cache 447-class frame logits (40 ms, 250 frames per 10 s clip) of a PretrainedSED checkpoint."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # keep BLAS from taking every core
    os.environ.setdefault(_v, "2")

import argparse
import sys
import time

import librosa
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSED = os.path.join(ROOT, "third_party", "PretrainedSED")
sys.path.insert(0, PSED)

SR = 16_000
SEG = 10 * SR


def build(name):
    os.chdir(PSED)  # checkpoints resolve relative to resources/
    from models.prediction_wrapper import PredictionsWrapper
    if name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        return PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    if name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return PredictionsWrapper(ATSTWrapper(), checkpoint="ATST-F_strong_1")
    if name == "fpasst":
        from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
        return PredictionsWrapper(FPaSSTWrapper(), checkpoint="fpasst_strong_1")
    if name == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        m = M2DWrapper()
        return PredictionsWrapper(m, checkpoint="M2D_strong_1", embed_dim=m.m2d.cfg.feature_d)
    if name == "ASIT":
        from models.asit.ASIT_wrapper import ASiTWrapper
        return PredictionsWrapper(ASiTWrapper(), checkpoint="ASIT_strong_1")
    if name.startswith("frame_mn"):
        from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
        from models.frame_mn.utils import NAME_TO_WIDTH
        m = FrameMNWrapper(NAME_TO_WIDTH(name))
        dim = m.state_dict()["frame_mn.features.16.1.bias"].shape[0]
        return PredictionsWrapper(m, checkpoint=f"{name}_strong_1", embed_dim=dim)
    raise ValueError(name)


def load(path, shift):
    """Load a clip as exactly 10 s; a positive shift delays the audio by `shift` samples."""
    x, _ = librosa.load(path, sr=SR, mono=True)
    x = x[:SEG]
    if shift > 0:
        x = np.concatenate([np.zeros(shift, np.float32), x])
    elif shift < 0:
        x = x[-shift:]
    x = x[:SEG]
    return np.pad(x, (0, SEG - len(x))), min(len(x), SEG) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift-ms", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    audio_dir = os.path.abspath(a.audio_dir)
    out = os.path.abspath(a.out)
    names = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav") and not f.startswith("._"))
    shift = int(round(a.shift_ms * SR / 1000))

    model = build(a.model).eval().to(a.device)
    logits = np.zeros((len(names), 447, 250), np.float16)
    t0 = time.time()
    for i in range(0, len(names), a.batch):
        chunk = [load(os.path.join(audio_dir, n), shift)[0] for n in names[i:i + a.batch]]
        x = torch.from_numpy(np.stack(chunk)).to(a.device)
        with torch.no_grad():
            y, _ = model(model.mel_forward(x))
        logits[i:i + len(chunk)] = y.float().cpu().numpy().astype(np.float16)
        if i // a.batch % 10 == 0:
            print(f"{a.model} {i + len(chunk)}/{len(names)} {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, logits=logits, names=np.array(names), shift_ms=a.shift_ms, frame_s=0.04)
    print(f"done {out} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
