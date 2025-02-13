#!/bin/bash
# Script ver: PANDA

# To run this script:
# nohup bash tile_and_embed_panda.sh > ./logs/tile_and_embed_panda.log 2>&1 &

# To kill the related processes:
# ps -ef | grep tile_and_embed_panda.sh | awk '{print $2}' |xargs kill
# ps -ef | grep Build_tiles_dataset.py | awk '{print $2}' |xargs kill
# ps -ef | grep Build_embedded_dataset.py | awk '{print $2}' |xargs kill

set -e

# Select GPU to use
export CUDA_VISIBLE_DEVICES=1,2

# Activate conda env and go to project dir
source /root/miniforge3/bin/activate
conda activate BigModel
cd /home/workenv/BigModel

# Tile the datasets
python -u DataPipe/Build_tiles_dataset.py \
    --WSI_dataset_path /data/hdd_1/DevDatasets/WSI/PANDA/train_images \
    --tiled_WSI_dataset_path /data/ssd_1/WSI/PANDA/tiles-datasets \
    --edge_size 224 \
    --force_read_level 0 \
    --force_roi_scale 1 \
    --mode general

# Embed the datasets
python -u DataPipe/Build_embedded_dataset.py \
    --WSI_dataset_path /data/ssd_1/WSI/PANDA/tiles-datasets \
    --embedded_WSI_dataset_path /data/ssd_1/WSI/PANDA/tiles-embeddings \
    --model_name gigapath \
    --model_weight_path /data/hdd_1/model_weights/gigapath_tile_encoder.pt \
    --edge_size 224 \
    --force_read_level 0 \
    --force_roi_scale 1 \
    --PrefetchDataLoader_num_workers 32 \
    --batch_size 2048

mkdir -p /data/hdd_1/BigModel/PANDA
cp -r /data/ssd_1/WSI/PANDA/tiles-embeddings /data/hdd_1/BigModel/PANDA/

set +e