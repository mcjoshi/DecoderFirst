# Decoder First, Fusion Second: Ensembling Sound Event Detectors

Code and results for the paper by Manish Joshi and Neeraj Singh Aithani (equal contribution).

Ensembles of sound event detection (SED) models usually average the members' frame posteriors and decode the average with a threshold and a median filter.
This repository separates what the decoder and the ensemble rule each contribute to the errors of such ensembles.
It compares posterior averaging with event-level fusion (weighted boxes fusion on time intervals), under median-filter and change-point (cSEBB) decoding, on DESED, URBAN-SED and TUT-SED.

## Findings

- Edge placement is set by the decoder: with a median filter, averaged ensembles misplace onsets as much as their best single member, and cSEBB roughly halves the error.
- Averaging leaves most of its members' fragmented events and weakly supported false alarms; event fusion largely removes them.
- With a median-filter decoder, fusion beats averaging with tuned gap filling and minimum duration by 0.02 to 0.04 event F1 in seven of ten settings.
- After cSEBB, the fusion level changes event F1 by less than 0.01 in most settings, except for uncalibrated zero-shot members.
- Under threshold-swept PSDS, averaging is at least as good, so fusion is a tool for a fixed operating point.

## Layout

| path | contents |
|---|---|
| `src/sedlib.py` | class maps, references, decoders (median filter, cSEBB), event fusion, collar-based event F1 |
| `src/ensemble_arms.py` | every ensembling arm, cross-fitted or tuned on a development split, collar sweep, paired clip bootstrap |
| `src/cache_*.py` | frame logits of the members (PretrainedSED, PANNs, DCASE 2023 baseline CRNN) |
| `src/adapt_members.py`, `src/train_heads.py` | adapted members (per-frame logistic heads) and trained heads (BiGRU) |
| `src/support_analysis.py` | error decomposition: edge error on commonly matched references, fragmentation, false alarms |
| `src/psds_arms.py` | PSDS1 and PSDS2 of averaging and fusion |
| `src/make_tables.py`, `src/fig_*.py` | the paper's tables and figures |
| `lists/` | DESED clips excluded for AudioSet overlap, TUT-SED development and evaluation recordings |
| `results/` | the result files behind every number in the paper (see `results/README.md`) |
| `scripts/` | third-party setup and the full reproduction pipeline |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install "numpy<2" cython
pip install -r requirements.txt --no-build-isolation
bash scripts/setup_third_party.sh
```

`scripts/setup_third_party.sh` clones PretrainedSED and DESED_task at the commits used for the paper.
PretrainedSED downloads its AudioSet-Strong checkpoints on first use.

## Data

Set `SED_DATA` to a folder with the layout below (default: `./data`).

| source | place under `$SED_DATA` |
|---|---|
| DESED public evaluation, [Zenodo 3588172](https://zenodo.org/records/3588172) (`DESEDpublic_eval.tar.gz`) | `datasets/dataset/` |
| URBAN-SED v2.0.0, [Zenodo 1324404](https://zenodo.org/records/1324404) | `datasets/URBAN-SED_v2.0.0/` |
| TUT Sound Events 2017, development [Zenodo 814831](https://zenodo.org/records/814831) and evaluation [Zenodo 1040179](https://zenodo.org/records/1040179) | `datasets/tutsed/TUT-sound-events-2017-{development,evaluation}/` |
| PANNs CNN14 DecisionLevelMax, [Zenodo 3987831](https://zenodo.org/records/3987831) (`Cnn14_DecisionLevelMax_mAP=0.385.pth`) | `checkpoints/` |
| DCASE 2023 Task 4 baseline, [Zenodo 7759146](https://zenodo.org/records/7759146) (`dcase2023_task4_baseline.zip`) | `checkpoints/dcase2023/` |

No audio or third-party checkpoint is redistributed here.
The 25 DESED clips whose YouTube identifier appears in any AudioSet split are listed in `lists/desed_eval_audioset_overlap.txt` and excluded from every DESED result.

## Reproduce

Tables from the shipped results (minutes, no audio needed):

```bash
python src/make_tables.py        # writes tables/tab_main.tex, tab_controls.tex, tab_mech.tex
```

Everything from audio (CPU only; the cSEBB runs take hours):

```bash
bash scripts/reproduce.sh            # or one stage: cache | adapt | ensembles | analysis | tables
```

`scripts/ensembles.sh` holds the exact command behind each file in `results/`.

| paper | command | result files |
|---|---|---|
| Table 1 | `python src/make_tables.py` | `cw_*`, `ut_*`, `tut_*`, `pp_*` |
| Table 2 | `python src/make_tables.py` | `pp_*` |
| Table 3 | `python src/make_tables.py` | `support_v5_desed_raw`, `support_v4_urban_*` |
| Fig. 1 | `bash scripts/reproduce.sh tables` | `cw_desed_raw`, `ut_urban_zs_raw`, `ut_urban_ad_raw` |
| Fig. 2 | `bash scripts/reproduce.sh tables` (needs cached member logits) | `v2_desed_raw` |
| PSDS | `bash scripts/reproduce.sh analysis` | `psds_*` |
| fixed 200 ms operating point | `scripts/ensembles.sh` | `fxs_*` |
| further DESED splits | `scripts/ensembles.sh` | `rs*` |
| ensemble composition | `scripts/ensembles.sh` | `desed_raw_top*`, `het_*` |

## Protocol notes

- DESED parameters are cross-fitted on two folds of the evaluation set (split by YouTube identifier), so absolute DESED scores are not comparable to published systems; only differences between arms are.
- URBAN-SED and TUT-SED parameters are chosen on the validation or development split and the test split is scored once.
- Event F1 follows sed_eval: onset collar c, offset collar max(c, 0.2 x duration), macro over classes, at collars of 20 ms to 1 s.

## Citation

```bibtex
@misc{joshi2026decoderfirst,
  title  = {Decoder First, Fusion Second: Ensembling Sound Event Detectors},
  author = {Joshi, Manish and Aithani, Neeraj Singh},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT, see `LICENSE`.
Datasets and third-party models keep their own licenses.
