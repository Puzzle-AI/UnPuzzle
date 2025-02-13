#!/bin/bash
# Script ver: GENERATED_SCRIPT

# To run this script:
# nohup bash GENERATED_SCRIPT_prepare_dataset.sh > ./logs/GENERATED_SCRIPT_prepare_dataset.log 2>&1 &

# To kill the related processes:
# ps -ef | grep GENERATED_SCRIPT_prepare_dataset.sh | awk '{print $2}' |xargs kill
# ps -ef | grep Tiles_dataset.py | awk '{print $2}' |xargs kill
# ps -ef | grep Embedded_dataset.py | awk '{print $2}' |xargs kill

set -e

# # Select GPU to use
# export CUDA_VISIBLE_DEVICES=SELECTED_GPU

# Activate conda env and go to project dir
source /opt/conda/bin/activate
conda activate BigModel
cd /home/workenv/BigModel

# Tile the datasets
python -u DataPipe/Build_tiles_dataset.py \
    --WSI_dataset_path /data/hdd_2/WSI_raw/DATASET_NAME \
    --tiled_WSI_dataset_path /data/ssd_1/WSI_resnet18/DATASET_NAME/tiles-datasets \
    --edge_size 224 \
    --target_mpp 0.5

# Embed the datasets
python -u DataPipe/Build_embedded_dataset.py \
    --WSI_dataset_path /data/ssd_1/WSI_resnet18/DATASET_NAME/tiles-datasets \
    --embedded_WSI_dataset_path /data/ssd_1/WSI_resnet18/DATASET_NAME/tiles-embeddings \
    --model_name ResNet18 \
    --model_weight_path /data/hdd_1/model_weights/roi_models/resnet18.pth \
    --edge_size 224 \
    --PrefetchDataLoader_num_workers 32 \
    --batch_size BATCH_SIZE

set +e