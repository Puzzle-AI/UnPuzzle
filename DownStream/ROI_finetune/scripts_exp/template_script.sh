#!/bin/bash
# Script ver: GENERATED_SCRIPT

# To run this script:
# nohup bash GENERATED_SCRIPT_train.sh > ./GENERATED_SCRIPT_train.log 2>&1 &

# To kill the related processes:
# ps -ef | grep GENERATED_SCRIPT_train.sh | awk '{print $2}' |xargs kill
# ps -ef | grep Tiles_dataset.py | awk '{print $2}' |xargs kill
# ps -ef | grep Embedded_dataset.py | awk '{print $2}' |xargs kill

set -e

# Activate conda env and go to project dir
source /root/miniforge3/bin/activate
conda activate BigModel

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE="/data/ssd_1/XM_dev/share_mounted/ai_module/pretrained/"
export TORCH_HOME="/data/ssd_1/XM_dev/share_mounted/ai_module/pretrained/"

cd /home/workenv/BigModel/DownStream/ROI_finetune

# Train
python CLS_Train.py \
    --task_name TASK_NAME \
    --model_idx MODEL_NAME \
    --dataroot DATASET_NAME \
    --gpu_idx GPU_IDX \
    --enable_tensorboard \
    --augmentation_name CellMix-Random \
    --batch_size BATCH_SIZE \
    --num_epochs NUM_EPOCHS \
    --edge_size 224 \
    --lr LR \
    --num_workers NUM_WORKERS \
    --draw_root /data/private/BigModel/runs/
# Test
python CLS_Test.py \
  --model_idx MODEL_NAME \
  --gpu_idx GPU_IDX \
  --dataroot DATASET_NAME \
  --data_augmentation_mode 2 \
  --enable_tensorboard \
  --edge_size 224 \
  --batch_size BATCH_SIZE \
  --draw_root /data/private/BigModel/runs/CLS_TASK_NAME/MODEL_NAME_lr_LR/ \
  --model_path_by_hand /data/private/BigModel/runs/CLS_TASK_NAME/MODEL_NAME_lr_LR/CLS_TASK_NAME_MODEL_NAME_lr_LR.pth

set +e
