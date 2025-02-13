
set -e

cd /home/workenv/BigModel/DataPipe

python Slide_dataset_tools.py \
    --root_path /data/ssd_1/WSI/TCGA-COAD-READ/tiles-embeddings/ \
    --task_description_csv /home/workenv/BigModel/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode TCGA \
    --dataset_name coad-read \
    --tasks_to_run iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A \
    --k 5

set +e