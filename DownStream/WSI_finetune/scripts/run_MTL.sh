#!/bin/sh
# demo for running MTL for WSI tasks Script  ver： Nov 06th 00:00
cd /BigModel/DownStream/WSI_finetune

# Train
python MTL_Train.py \
    --model_name gigapath \
    --tag no_slidePT_gigapath_tiles \
    --local_weight_path False \
    --save_model_path /data/hdd_1/BigModel/saved_models \
    --root_path /data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath \
    --ROI_feature_dim 1536 \
    --runs_path /data/hdd_1/BigModel/runs \
    --enable_tensorboard \
    --task_description_csv /data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-1 \
    --max_tiles 10000 \
    --num_epochs 100 \
    --warmup_epochs 10 \
    --intake_epochs 50

# Test
python MTL_Test.py \
    --model_name gigapath \
    --tag no_slidePT_gigapath_tiles \
    --save_model_path /data/hdd_1/BigModel/saved_models \
    --root_path /data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath \
    --ROI_feature_dim 1536 \
    --runs_path /data/hdd_1/BigModel/runs \
    --task_description_csv /data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-1 \
    --max_tiles 10000

# Decode the test results to csv
cd /BigModel/Utils

python Decode_MTL_pred.py \
    --model_name gigapath \
    --tag no_slidePT_gigapath_tiles \
    --root_path /data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath \
    --runs_path /data/hdd_1/BigModel/runs \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds