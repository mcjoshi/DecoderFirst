#!/usr/bin/env bash
# Every ensemble run behind the paper, one line per results/*.json. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/ensemble_arms.py --dataset desed --calibrate --sebb --gap --classwise --out results/cw_desed_cal.json
python src/ensemble_arms.py --dataset desed --sebb --gap --classwise --out results/cw_desed_raw.json
python src/ensemble_arms.py --dataset desed --models BEATs,ATST-F --sebb --out results/desed_raw_top2.json
python src/ensemble_arms.py --dataset desed --models BEATs,ATST-F,ASIT --sebb --out results/desed_raw_top3.json
python src/ensemble_arms.py --dataset desed --fixed-collar 0.2 --sebb --pp --out results/fxs_desed_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --dev-split validate --fixed-collar 0.2 --sebb --pp --out results/fxs_urban_ad_raw.json
python src/ensemble_arms.py --dataset desed --models BEATs,panns,crnn2023t --calibrate --sebb --gap --classwise --out results/het_desed_cal_3fam.json
python src/ensemble_arms.py --dataset desed --models BEATs,ATST-F,panns,crnn2023t --calibrate --sebb --gap --classwise --out results/het_desed_cal_4m.json
python src/ensemble_arms.py --dataset desed --models BEATs,ATST-F,fpasst,M2D,ASIT,frame_mn10,frame_mn06,panns,crnn2023t --calibrate --sebb --gap --classwise --out results/het_desed_cal_9m.json
python src/ensemble_arms.py --dataset urbansed --models BEATs,ATST-F,fpasst,M2D,ASIT,frame_mn10,frame_mn06,panns --calibrate --sebb --gap --classwise --out results/het_urban_cal_8m.json
python src/ensemble_arms.py --dataset desed --calibrate --gap --pp --classwise --out results/pp_desed_cal.json
python src/ensemble_arms.py --dataset desed --gap --pp --classwise --out results/pp_desed_raw.json
python src/ensemble_arms.py --dataset tutsed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --calibrate --holdout-list lists/tutsed_eval_recordings.txt --gap --pp --classwise --out results/pp_tut_ad_cal.json
python src/ensemble_arms.py --dataset tutsed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --holdout-list lists/tutsed_eval_recordings.txt --gap --pp --classwise --out results/pp_tut_ad_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --calibrate --dev-split validate --gap --pp --classwise --out results/pp_urban_ad_cal.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --dev-split validate --gap --pp --classwise --out results/pp_urban_ad_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-gru0,ATST-F-gru0,fpasst-gru0,M2D-gru0,ASIT-gru0 --dev-split validate --gap --pp --classwise --out results/pp_urban_gru0_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-gru1,ATST-F-gru1,fpasst-gru1,M2D-gru1,ASIT-gru1 --dev-split validate --gap --pp --classwise --out results/pp_urban_gru1_raw.json
python src/ensemble_arms.py --dataset urbansed --calibrate --dev-split validate --gap --pp --classwise --out results/pp_urban_zs_cal.json
python src/ensemble_arms.py --dataset urbansed --dev-split validate --gap --pp --classwise --out results/pp_urban_zs_raw.json
python src/ensemble_arms.py --dataset desed --fold-seed 1 --gap --classwise --out results/rs1_desed_raw.json
python src/ensemble_arms.py --dataset desed --fold-seed 1 --sebb --classwise --out results/rs1s_desed_raw.json
python src/ensemble_arms.py --dataset desed --fold-seed 2 --gap --classwise --out results/rs2_desed_raw.json
python src/ensemble_arms.py --dataset desed --fold-seed 2 --sebb --classwise --out results/rs2s_desed_raw.json
python src/ensemble_arms.py --dataset desed --fold-seed 3 --gap --classwise --out results/rs3_desed_raw.json
python src/ensemble_arms.py --dataset desed --fold-seed 3 --sebb --classwise --out results/rs3s_desed_raw.json
python src/ensemble_arms.py --dataset tutsed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --calibrate --holdout-list lists/tutsed_eval_recordings.txt --sebb --gap --classwise --out results/tut_ad_cal.json
python src/ensemble_arms.py --dataset tutsed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --holdout-list lists/tutsed_eval_recordings.txt --sebb --gap --classwise --out results/tut_ad_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --calibrate --dev-split validate --sebb --gap --classwise --out results/ut_urban_ad_cal.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --dev-split validate --sebb --gap --classwise --out results/ut_urban_ad_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-gru0,ATST-F-gru0,fpasst-gru0,M2D-gru0,ASIT-gru0 --dev-split validate --sebb --gap --classwise --out results/ut_urban_gru0_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-gru1,ATST-F-gru1,fpasst-gru1,M2D-gru1,ASIT-gru1 --dev-split validate --sebb --gap --classwise --out results/ut_urban_gru1_raw.json
python src/ensemble_arms.py --dataset urbansed --calibrate --dev-split validate --sebb --gap --classwise --out results/ut_urban_zs_cal.json
python src/ensemble_arms.py --dataset urbansed --dev-split validate --sebb --gap --classwise --out results/ut_urban_zs_raw.json
python src/ensemble_arms.py --dataset desed --calibrate --sebb --gap --out results/v2_desed_cal.json
python src/ensemble_arms.py --dataset desed --sebb --gap --ablate --out results/v2_desed_raw.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --calibrate --sebb --gap --out results/v2_urban_ad_cal.json
python src/ensemble_arms.py --dataset urbansed --models BEATs-adapt,ATST-F-adapt,fpasst-adapt,M2D-adapt,ASIT-adapt,frame_mn10-adapt,frame_mn06-adapt --sebb --gap --out results/v2_urban_ad_raw.json
python src/ensemble_arms.py --dataset urbansed --calibrate --sebb --gap --out results/v2_urban_zs_cal.json
python src/ensemble_arms.py --dataset urbansed --sebb --gap --out results/v2_urban_zs_raw.json
