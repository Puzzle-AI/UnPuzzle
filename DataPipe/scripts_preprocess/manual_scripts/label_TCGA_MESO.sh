set -e

cd /home/workenv/BigModel/DataPipe

python Slide_dataset_tools.py \
    --root_path /data/ssd_1/WSI/TCGA-MESO/tiles-embeddings/ \
    --task_description_csv /data/hdd_1/CPIA_label/cbioportal/TCGA-MESO_clinical_data.csv \
    --subject_key patientId \
    --sample_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode TCGA \
    --dataset_name meso \
    --fix_random_seed \
    --k 5

set +e