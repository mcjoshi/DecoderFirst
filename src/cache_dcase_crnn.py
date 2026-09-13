"""Cache DESED-class logits of the DCASE 2023 Task 4 baseline CRNN (trained from scratch on DESED)."""
import argparse
import os
import sys

import librosa
import numpy as np
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "DESED_task"))
from desed_task.nnet.CRNN import CRNN  # noqa: E402
from desed_task.utils.scaler import TorchScaler  # noqa: E402

SR, SEG, HOP, POOL = 16_000, 160_000, 256, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", choices=["student", "teacher"], default="student")
    a = ap.parse_args()
    torch.set_num_threads(16)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    net = CRNN(**hp["net"])
    prefix = f"sed_{a.net}."
    sd = {k[len(prefix):]: v for k, v in ck["state_dict"].items()
          if k.startswith(prefix) and not k.endswith(("total_ops", "total_params"))}
    net.load_state_dict(sd)
    net.eval()

    f = hp["feats"]
    mel = MelSpectrogram(sample_rate=f["sample_rate"], n_fft=f["n_window"], win_length=f["n_window"],
                         hop_length=f["hop_length"], f_min=f["f_min"], f_max=f["f_max"], n_mels=f["n_mels"],
                         window_fn=torch.hamming_window, wkwargs={"periodic": False}, power=1)
    to_db = AmplitudeToDB(stype="amplitude")
    to_db.amin = 1e-5
    scaler = TorchScaler("instance", hp["scaler"]["normtype"], hp["scaler"]["dims"])

    names = sorted(n for n in os.listdir(a.audio_dir) if n.endswith(".wav") and not n.startswith("._"))
    out = []
    with torch.no_grad():
        for n in names:
            x, _ = librosa.load(os.path.join(a.audio_dir, n), sr=SR, mono=True)
            x = np.pad(x[:SEG], (0, max(0, SEG - len(x))))
            feats = scaler(to_db(mel(torch.from_numpy(x)[None])).clamp(min=-50, max=80))
            strong, _ = net(feats)                                    # (1, C, T) probabilities
            p = strong[0].clamp(1e-6, 1 - 1e-6).numpy()
            out.append(np.log(p / (1 - p)))
    logits = np.stack(out).astype(np.float16)
    np.savez(a.out, logits=logits, names=np.array(names), shift_ms=0.0, frame_s=HOP * POOL / SR, mapped=1)
    print(a.out, logits.shape)


if __name__ == "__main__":
    main()
