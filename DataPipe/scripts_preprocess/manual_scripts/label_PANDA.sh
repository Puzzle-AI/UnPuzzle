set -e

cd /home/workenv/BigModel/DataPipe

python Slide_dataset_tools.py \
    --root_path /data/ssd_1/WSI/PANDA/tiles-embeddings/ \
    --task_description_csv /data/ssd_1/WSI_label/PANDA_train.csv \
    --subject_key fileId \
    --sample_key fileId \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode basic \
    --dataset_name camelyon16 \
    --tasks_to_run "BREAST_METASTASIS" \
    --fix_random_seed \
    --k 5

set +e