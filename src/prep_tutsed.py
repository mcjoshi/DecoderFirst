"""Cut TUT Sound Events 2017 (street) recordings into 10 s clips and write per-clip strong labels."""
import csv
import glob
import os

import numpy as np
import soundfile as sf

import sedlib as L

ROOT = f"{L.DATA}/datasets/tutsed"
OUT = f"{ROOT}/clips"
SEG = 10.0


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for part in ("development", "evaluation"):
        base = f"{ROOT}/TUT-sound-events-2017-{part}"
        for ann in sorted(glob.glob(f"{base}/meta/street/*.ann")):
            rec = os.path.basename(ann)[:-4]
            audio, sr = sf.read(f"{base}/audio/street/{rec}.wav", always_2d=True)
            audio = audio.mean(1).astype(np.float32)
            events = []
            with open(ann) as fh:
                for line in fh:   # development rows carry path/scene columns, evaluation rows do not
                    parts = line.strip().split("\t")
                    hit = [i for i, p in enumerate(parts) if p in L.CLASS_MAPS["tutsed"]]
                    if hit:
                        i = hit[0]
                        events.append((float(parts[i - 2]), float(parts[i - 1]), parts[i]))
            n_seg = int(len(audio) / sr // SEG)
            for k in range(n_seg):
                t0 = k * SEG
                name = f"{rec}__{k:03d}.wav"
                sf.write(f"{OUT}/{name}", audio[int(t0 * sr):int((t0 + SEG) * sr)], sr)
                for on, off, c in events:
                    a, b = max(on, t0) - t0, min(off, t0 + SEG) - t0
                    if b - a >= 0.05:
                        rows.append((name, round(a, 3), round(b, 3), c))
                rows.append((name, "", "", ""))   # keeps event-free clips in the file list
    with open(f"{ROOT}/clips_labels.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["filename", "onset", "offset", "event_label"])
        w.writerows(rows)
    names = {r[0] for r in rows}
    print(len(names), "clips,", sum(1 for r in rows if r[3]), "events")


if __name__ == "__main__":
    main()
