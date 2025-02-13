#!/bin/bash
# Script ver: ${GENERATED_SCRIPT}

# To run this script:
# nohup bash ${GENERATED_SCRIPT}_train.sh > ./logs/${GENERATED_SCRIPT}_train.log 2>&1 &

# To kill the related processes:
# ps -ef | grep ${GENERATED_SCRIPT}_train.sh | awk '{print $2}' |xargs kill

set -e

# Select GPU to use
export CUDA_VISIBLE_DEVICES=${SELECTED_GPU}

# Activate conda env and go to project dir
source ${CONDA_ACTIVATE_SRC}
conda activate ${CONDA_ENV_NAME}

# cd /home/workenv/BigModel/DataPipe

# # prepare labels
# python Slide_dataset_tools.py \
#     --root_path /data/ssd_1/WSI_resnet18/${DATASET_NAME}/tiles-embeddings/ \
#     --task_description_csv /data/ssd_1/WSI_resnet18/${DATASET_NAME}/${DATASET_NAME}_clinical_data.csv \
#     --slide_id_key patientId \
#     --slide_id slide_id \
#     --split_target_key fold_information \
#     --task_setting_folder_name task-settings-5folds \
#     --mode TCGA \
#     --dataset_name ${DATASET_NAME} \
#     --k 5

cd /home/workenv/BigModel/DownStream/WSI_finetune

# Train
python -u MTL_Train.py \
    --model_name ${MODEL_NAME} \
    --tag no_slidePT_tiles_${DATASET_NAME}_${TASK_NAME}_${LR} \
    --local_weight_path False \
    --root_path /data/ssd_1/WSI_resnet18/${DATASET_NAME}/tiles-embeddings \
    --ROI_feature_dim ${ROI_FEATURE_DIM} \
    --runs_path /data/private/BigModel/runs_${DATASET_NAME} \
    --enable_tensorboard \
    --task_description_csv /data/ssd_1/WSI_resnet18/${DATASET_NAME}/tiles-embeddings/task-settings-5folds/${CSV_FNAME} \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key ${SLIDE_ID_KEY} \
    --split_target_key fold_information_5fold-${FOLD} \
    --max_tiles ${MAX_TILES} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --num_epochs ${NUM_EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --intake_epochs ${INTAKE_EPOCHS} \
    --gpu_idx ${SELECTED_GPU} \
    --lr ${LR} \
    --tasks_to_run ${TASK_TO_RUN} \
    --compress_to_runs_path

# Test
python -u MTL_Test.py \
    --model_name ${MODEL_NAME} \
    --tag no_slidePT_tiles_${DATASET_NAME}_${TASK_NAME}_${LR} \
    --root_path /data/ssd_1/WSI_resnet18/${DATASET_NAME}/tiles-embeddings \
    --ROI_feature_dim ${ROI_FEATURE_DIM} \
    --runs_path /data/private/BigModel/runs_${DATASET_NAME} \
    --task_description_csv /data/ssd_1/WSI_resnet18/${DATASET_NAME}/tiles-embeddings/task-settings-5folds/${CSV_FNAME} \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key ${SLIDE_ID_KEY} \
    --split_target_key fold_information_5fold-${FOLD} \
    --max_tiles ${MAX_TILES} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --gpu_idx ${SELECTED_GPU} \
    --tasks_to_run ${TASK_TO_RUN} \
    --compress_to_runs_path

# Decode the test results to csv
cd /home/workenv/BigModel/Utils
python -u Decode_MTL_pred.py \
    --model_name ${MODEL_NAME} \
    --tag no_slidePT_tiles_${DATASET_NAME}_${TASK_NAME}_${LR} \
    --root_path /data/ssd_1/WSI_resnet18/${DATASET_NAME}/tiles-embeddings \
    --runs_path /data/private/BigModel/runs_${DATASET_NAME} \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds \
    --compress_to_runs_path

set +e