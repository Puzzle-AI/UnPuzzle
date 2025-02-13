
set -e

cd ../

# TCGA-lung
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga-lung" \
  --gpu_list "0%1" \
  --dataset "TCGA-lung" \
  --csv_fname "task_description_tcga-lung_reduced_20241203.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%FEV1_PERCENT_REF_POSTBRONCHOLIATOR%lung-cancer-subtyping%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-lung explore
# ("TransMIL" "SlideMax" "SlideAve" "gigapath" "PathRWKV" "DTFD" "CLAM" "UNI" "DSMIL" "ABMIL" "SlideViT" "SlideVPT" "LongNet")
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga-lung-explore" \
  --gpu_list "0%1" \
  --dataset "TCGA-lung" \
  --csv_fname "task_description_tcga-lung_reduced_20241203.csv" \
  --fold_range 1 \
  --model_names "TransMIL%SlideMax%SlideAve%gigapath%PathRWKV%DTFD%CLAM%UNI%DSMIL%ABMIL%SlideViT%SlideVPT%LongNet" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%FEV1_PERCENT_REF_POSTBRONCHOLIATOR%lung-cancer-subtyping%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 10000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 200 \
  --warmup_epochs 10 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-BRCA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_brca_scripts" \
  --gpu_list "0%1" \
  --dataset "TCGA-BRCA" \
  --csv_fname "task_description_tcga_brca.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%IHC_HER2%HISTOLOGICAL_DIAGNOSIS%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-MESO
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_meso_scripts" \
  --gpu_list "0%1" \
  --dataset "TCGA-MESO" \
  --csv_fname "task_description_tcga_meso.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%HISTOLOGICAL_DIAGNOSIS%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-UCEC
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_ucec_scripts" \
  --gpu_list "0%1" \
  --dataset "TCGA-UCEC" \
  --csv_fname "task_description_tcga_ucec.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "GRADE%HISTOLOGICAL_DIAGNOSIS%TUMOR_INVASION_PERCENT%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-BLCA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_blca_scripts" \
  --gpu_list "0%1" \
  --dataset "TCGA-BLCA" \
  --csv_fname "task_description_tcga_blca.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%GRADE%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# PANDA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "panda_scripts" \
  --gpu_list "0%1" \
  --dataset "PANDA" \
  --csv_fname "task_description_panda.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "isup_grade%gleason_score" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "image_id"

set +e