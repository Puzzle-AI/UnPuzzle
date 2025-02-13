#!/bin/bash
# Script ver: camelyon16

# To run this script:
# nohup bash tile_and_embed_camelyon16.sh > ./logs/tile_and_embed_camelyon16.log 2>&1 &

# To kill the related processes:
# ps -ef | grep tile_and_embed_camelyon16.sh | awk '{print $2}' |xargs kill
# ps -ef | grep Build_tiles_dataset.py | awk '{print $2}' |xargs kill
# ps -ef | grep Build_embedded_dataset.py | awk '{print $2}' |xargs kill

set -e

# Select GPU to use
# export CUDA_VISIBLE_DEVICES=1,2

# Activate conda env and go to project dir
source /root/miniforge3/bin/activate
conda activate BigModel
cd /home/workenv/BigModel

# Tile the datasets
python -u DataPipe/Build_tiles_dataset.py \
    --WSI_dataset_path /data/hdd_1/DevDatasets/WSI/CAMELYON16 \
    --tiled_WSI_dataset_path /data/ssd_1/WSI/CAMELYON16/tiles-datasets \
    --edge_size 224 \
    --mode general \
    --target_mpp 0.5

# Embed the datasets
python -u DataPipe/Build_embedded_dataset.py \
    --WSI_dataset_path /data/ssd_1/WSI/CAMELYON16/tiles-datasets \
    --embedded_WSI_dataset_path /data/ssd_1/WSI/CAMELYON16/tiles-embeddings \
    --model_name gigapath \
    --model_weight_path /data/hdd_1/model_weights/gigapath_tile_encoder.pt \
    --edge_size 224 \
    --PrefetchDataLoader_num_workers 32 \
    --batch_size 2048 \
    --target_mpp 0.5

mkdir -p /data/hdd_1/BigModel/CAMELYON16
cp -r /data/ssd_1/WSI/CAMELYON16/tiles-embeddings /data/hdd_1/BigModel/

set +e