#!/bin/bash


set -e

cd /home/workenv/BigModel/DataPipe
mkdir -p ./logs

# b=1
python -u dataset_framework.py \
    2>&1 | tee ./logs/TCGA-lung-b1_nw0.log
python -u dataset_framework.py \
    --num_workers 2 \
    2>&1 | tee ./logs/TCGA-lung-b1_nw2.log
python -u dataset_framework.py \
    --num_workers 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung-b1_nw2_pinmem_pworker.log
python -u dataset_framework.py \
    --num_workers 4 \
    2>&1 | tee ./logs/TCGA-lung-b1_nw4.log
python -u dataset_framework.py \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung-b1_nw4_pinmem_pworker.log
python -u dataset_framework.py \
    --num_workers 8 \
    2>&1 | tee ./logs/TCGA-lung-b1_nw8.log
python -u dataset_framework.py \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung-b1_nw8_pinmem_pworker.log
python -u dataset_framework.py \
    --num_workers 16 \
    2>&1 | tee ./logs/TCGA-lung-b1_nw16.log
python -u dataset_framework.py \
    --num_workers 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung-b1_nw16_pinmem_pworker.log
python -u dataset_framework.py \
    --num_workers 32 \
    2>&1 | tee ./logs/TCGA-lung-b1_nw32.log
python -u dataset_framework.py \
    --num_workers 32 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung-b1_nw32_pinmem_pworker.log
python -u dataset_framework.py \
    --num_workers 64 \
    2>&1 | tee ./logs/TCGA-lung-b1_nw64.log
python -u dataset_framework.py \
    --num_workers 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung-b1_nw64_pinmem_pworker.log

# b=2
python -u dataset_framework.py \
    --batch_size 2 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw0.log
python -u dataset_framework.py \
    --batch_size 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw0_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 2 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw2.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw2_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 4 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw4.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw4_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 8 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw8.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw8_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 16 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw16.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw16_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 32 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw32.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 32 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw32_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 64 \
    2>&1 | tee ./logs/TCGA-lung_b2_nw64.log
python -u dataset_framework.py \
    --batch_size 2 \
    --num_workers 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b2_nw64_pinmem_pworker.log

# b=4
python -u dataset_framework.py \
    --batch_size 4 \
    2>&1 | tee ./logs/TCGA-lung_b4_nw0.log
python -u dataset_framework.py \
    --batch_size 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw0_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 4 \
    --num_workers 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw2_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 4 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw4_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 4 \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw8_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 4 \
    --num_workers 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw16_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 4 \
    --num_workers 32 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw32_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 4 \
    --num_workers 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b4_nw64_pinmem_pworker.log

# b=8
python -u dataset_framework.py \
    --batch_size 8 \
    2>&1 | tee ./logs/TCGA-lung_b8_nw0.log
python -u dataset_framework.py \
    --batch_size 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw0_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 8 \
    --num_workers 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw2_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 8 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw4_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 8 \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw8_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 8 \
    --num_workers 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw16_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 8 \
    --num_workers 32 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw32_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 8 \
    --num_workers 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b8_nw64_pinmem_pworker.log

# b=16
python -u dataset_framework.py \
    --batch_size 16 \
    2>&1 | tee ./logs/TCGA-lung_b16_nw0.log
python -u dataset_framework.py \
    --batch_size 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw0_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw2_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw4_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw8_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw16_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 32 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw32_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b16_nw64_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 64 \
    --prefetch_factor 4 \
    2>&1 | tee ./logs/TCGA-lung_b16_nw64_pf4.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 64 \
    --prefetch_factor 8 \
    2>&1 | tee ./logs/TCGA-lung_b16_nw64_pf8.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 64 \
    --prefetch_factor 16 \
    2>&1 | tee ./logs/TCGA-lung_b16_nw64_pf16.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 64 \
    --prefetch_factor 32 \
    2>&1 | tee ./logs/TCGA-lung_b16_nw64_pf32.log
python -u dataset_framework.py \
    --batch_size 16 \
    --num_workers 64 \
    --prefetch_factor 64 \
    2>&1 | tee ./logs/TCGA-lung_b16_nw64_pf64.log

# b=64
python -u dataset_framework.py \
    --batch_size 64 \
    2>&1 | tee ./logs/TCGA-lung_b64_nw0.log
python -u dataset_framework.py \
    --batch_size 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw0_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 64 \
    --num_workers 2 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw2_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 64 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw4_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 64 \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw8_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 64 \
    --num_workers 16 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw16_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 64 \
    --num_workers 32 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw32_pinmem_pworker.log
python -u dataset_framework.py \
    --batch_size 64 \
    --num_workers 64 \
    --pin_memory \
    --persistent_workers \
    2>&1 | tee ./logs/TCGA-lung_b64_nw64_pinmem_pworker.log

set +e