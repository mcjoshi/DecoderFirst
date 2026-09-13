#!/usr/bin/env bash
# Rebuild every result, table and figure of the paper. Run from the repository root.
# Stages can be run separately: bash scripts/reproduce.sh cache|adapt|ensembles|analysis|tables
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${SED_DATA:-data}
MEMBERS="BEATs ATST-F fpasst M2D ASIT frame_mn10 frame_mn06"
ADAPTED=BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt
STAGE=${1:-all}

cache() {
  python src/prep_tutsed.py
  for m in $MEMBERS; do
    python src/cache_logits.py --model $m --audio-dir $DATA/datasets/dataset/audio/eval/public --out $DATA/cache/desed_eval/${m}_s0.npz
    for split in train validate test; do
      out=$([ $split = test ] && echo eval || echo $split)
      python src/cache_logits.py --model $m --audio-dir $DATA/datasets/URBAN-SED_v2.0.0/audio/$split --out $DATA/cache/urbansed_$out/${m}_s0.npz
    done
    python src/cache_logits.py --model $m --audio-dir $DATA/datasets/tutsed/clips --out $DATA/cache/tutsed_eval/${m}_s0.npz
  done
  python src/cache_panns.py --dataset desed --audio-dir $DATA/datasets/dataset/audio/eval/public --out $DATA/cache/desed_eval/panns_s0.npz
  python src/cache_panns.py --dataset urbansed --audio-dir $DATA/datasets/URBAN-SED_v2.0.0/audio/test --out $DATA/cache/urbansed_eval/panns_s0.npz
  python src/cache_dcase_crnn.py --ckpt "$DATA/checkpoints/dcase2023/ckpt/legacy_epoch=176-step=20886.ckpt" --net teacher \
    --audio-dir $DATA/datasets/dataset/audio/eval/public --out $DATA/cache/desed_eval/crnn2023t_s0.npz
}

adapt() {
  python src/adapt_members.py --dataset urbansed
  python src/adapt_members.py --dataset tutsed --splits eval
  for s in 0 1; do
    for m in BEATs ATST-F fpasst M2D ASIT; do python src/train_heads.py --model $m --seed $s; done
  done
}

ensembles() {
  bash scripts/ensembles.sh
}

analysis() {
  python src/support_analysis.py --dataset desed --result results/v2_desed_raw.json --pp-result results/pp_desed_raw.json \
    --sebb --singles BEATs,ATST-F --out results/support_v5_desed_raw.json
  python src/support_analysis.py --dataset urbansed --result results/v2_urban_zs_raw.json --sebb --singles BEATs,ASIT \
    --out results/support_v4_urban_zs_raw.json
  python src/support_analysis.py --dataset urbansed --result results/v2_urban_ad_raw.json --sebb --singles BEATs-adapt,ASIT-adapt \
    --out results/support_v4_urban_ad_raw.json
  for d in "desed v2_desed_raw psds_desed_raw" "urbansed v2_urban_zs_raw psds_urban_zs_raw"; do
    set -- $d
    python src/psds_arms.py --dataset $1 --result results/$2.json --out results/$3.json
    python src/psds_arms.py --dataset $1 --result results/$2.json --arms avg,fusion:sweep --out results/$3_sweep.json
  done
}

tables() {
  python src/make_tables.py
  python src/fig_curves.py --settings "(a) DESED, zero-shot=results/cw_desed_raw.json" \
    "(b) URBAN-SED, zero-shot=results/ut_urban_zs_raw.json" "(c) URBAN-SED, adapted=results/ut_urban_ad_raw.json" \
    --out figures/fig_curves.pdf
  python src/fig_timeline.py --result results/v2_desed_raw.json --clip X7b5DITgHCQ_174_184.wav --klass Frying \
    --out figures/fig_timeline.pdf
}

case $STAGE in
  cache) cache ;; adapt) adapt ;; ensembles) ensembles ;; analysis) analysis ;; tables) tables ;;
  all) cache; adapt; ensembles; analysis; tables ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
