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
    --WSI_dataset_path /data/hdd_1/CPIA/DATASET_NAME \
    --tiled_WSI_dataset_path /data/ssd_1/WSI/DATASET_NAME/tiles-datasets \
    --edge_size EDGE_SIZE \
    --target_mpp TARGET_MPP

# Embed the datasets
python -u DataPipe/Build_embedded_dataset.py \
    --WSI_dataset_path /data/ssd_1/WSI/DATASET_NAME/tiles-datasets \
    --embedded_WSI_dataset_path /data/ssd_1/WSI/DATASET_NAME/tiles-embeddings \
    --model_name MODEL_NAME \
    --model_weight_path /data/hdd_1/model_weights/MODEL_NAME_tile_encoder.pt \
    --edge_size EDGE_SIZE \
    --PrefetchDataLoader_num_workers 32 \
    --batch_size BATCH_SIZE

rm -rf /data/ssd_1/WSI/DATASET_NAME/tiles-datasets
mv /data/ssd_1/WSI/DATASET_NAME /data/hdd_1/BigModel/

set +e