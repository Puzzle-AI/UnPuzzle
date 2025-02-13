set -e

cd /home/workenv/BigModel/DataPipe

python Slide_dataset_tools.py \
    --root_path /data/ssd_1/WSI/TCGA-BRCA/tiles-embeddings/ \
    --task_description_csv /data/hdd_1/CPIA_label/cbioportal/TCGA-BRCA_clinical_data.csv \
    --subject_key patientId \
    --sample_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode TCGA \
    --dataset_name brca \
    --fix_random_seed \
    --k 5


# /home/workenv/test_env/task_description_tcga-brca_20241206.csv

python Slide_dataset_tools.py \
    --root_path /data/ssd_1/WSI/TCGA-BRCA/tiles-embeddings/ \
    --task_description_csv /data/ssd_1/WSI/TCGA-BRCA/tiles-embeddings/task-settings-5folds/task_description_tcga-brca_20241206.csv \
    --subject_key patientId \
    --sample_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode TCGA \
    --tasks_to_run AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%IHC_HER2%HISTOLOGICAL_DIAGNOSIS \
    --dataset_name brca \
    --fix_random_seed \
    --k 5


set +e