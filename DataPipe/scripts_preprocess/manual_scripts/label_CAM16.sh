set -e

cd /home/workenv/BigModel/DataPipe

python Slide_dataset_tools.py \
    --root_path /data/ssd_1/WSI/CAMELYON16/tiles-embeddings/ \
    --task_description_csv /data/ssd_1/WSI_label/CAMELYON16.csv \
    --subject_key image_id \
    --sample_key image_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode basic \
    --dataset_name CAMELYON16 \
    --tasks_to_run "isup_grade%gleason_score" \
    --fix_random_seed \
    --cls_task isup_grade \
    --k 5

set +e