# Results

Each `*.json` is written by `src/ensemble_arms.py` (or the analysis script named below) and holds, per arm, event F1 at the seven collars, paired bootstrap intervals against the reference arms (`d_<arm>`), and the configuration chosen on development data for every fold and collar.
Each `*_counts.npz` holds per-clip, per-class true-positive, false-positive and false-negative counts, from which `src/make_tables.py` computes collar-averaged differences and their clip-bootstrap intervals.
The exact command behind every file is in `scripts/ensembles.sh`.

| prefix | setting |
|---|---|
| `cw_desed_*` | DESED, cross-fitted, with per-class post-processing (Table 1) |
| `ut_urban_*`, `tut_ad_*` | URBAN-SED and TUT-SED, tuned on the development split, test split scored once (Table 1) |
| `pp_*` | the same settings with averaging plus jointly tuned gap filling and minimum duration (Tables 1 and 2) |
| `v2_*` | cross-fitted runs with global post-processing, used by the error decomposition and PSDS |
| `fxs_*` | every choice fixed at 200 ms and scored at all collars |
| `rs<k>_*`, `rs<k>s_*` | further random DESED splits, median filter and cSEBB |
| `desed_raw_top*`, `het_*` | ensemble composition |
| `support_v5_desed_raw`, `support_v4_urban_*` | error decomposition (Table 3), from `src/support_analysis.py` |
| `psds_*` | PSDS1 and PSDS2, from `src/psds_arms.py` |

Suffixes: `raw` uses the members' posteriors as they are, `cal` applies per-member, per-class Platt calibration fitted on development data; `zs` is zero-shot, `ad` adapted, `gru<s>` trained BiGRU heads with seed `s`.
