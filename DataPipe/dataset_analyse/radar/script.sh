#!/bin/bash

set -e

# Step1: process csv file to get an aggregated csv result
python extract_result.py \
  --dataset_folder "./1_dataset_result/results_lung" \
  --dataset_name TCGA-lung \
  --data_type WSI \
  --output_csv "./2_formatted_result/TCGA-overall.csv"
python extract_result.py \
  --dataset_folder "./1_dataset_result/results_lung" \
  --dataset_name TCGA-brca \
  --data_type WSI \
  --output_csv "./2_formatted_result/TCGA-overall.csv" \
  --append

# Step2: Draw a Radar Diagram
# For all datasets
python Draw_radar_plot.py \
  --input_file "./2_formatted_result/TCGA-overall.csv" \
  --tick_length 5 \
  --save_filename "./3_visual_result/radar_overall.png" \
  --labels_away_axis 'Overall Survival (Months)'

# For specific datasets, eg. TCGA-Lung and TCGA-BRCA
python Draw_radar_plot.py \
  --input_file "./2_formatted_result/TCGA-overall.csv" \
  --tick_length 5 \
  --datasets TCGA-lung \
  --save_filename "./3_visual_result/radar_TCGA-lung.png"

# # Step3: Redraw if you find need to adjust the offset of some label
# # Eg. if you want label 'Overall Survival (Months)\n(TCGA-Lung)' far away from axis
# python Draw_radar_plot.py \
#   --input_file "./2_formatted_result/TCGA-overall.csv" \
#   --tick_length 5 \
#   --datasets TCGA-lung \
#   --save_filename "./3_visual_result/radar_TCGA-lung.png" \
#   --labels_away_axis 'AJCC Tumor Stage'

set +e