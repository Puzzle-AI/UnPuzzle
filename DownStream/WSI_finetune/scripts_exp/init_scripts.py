"""
Script ver: Jan 30th 2025 00:30
Generate experiment scripts
"""

import os
import argparse

"""
Scripts generation functions
"""

def get_local_weight_path(model_name):
  # local weight path, change with your own weight path
  model_weight_path_dict = {
    "gigapath": "/data/ssd_1/model_weights/wsi_models/slide_encoder.pth",
    "UNI": "/data/ssd_1/model_weights/wsi_models/uni.pt",
    "Virchow": "/data/ssd_1/model_weights/wsi_models/virchow.pt",
  }
  if model_name in model_weight_path_dict:
    return model_weight_path_dict[model_name]
  else:
    return "False"

def generate_shell_scripts(template_path, output_dir, gpu_list, conda_activate_src, conda_env_name,
                           dataset, csv_fname, fold_range, model_name_list, tasks_to_run, use_weight_path,
                           roi_feature_dim, max_tiles, batch_size, num_workers, num_epochs, warmup_epochs, 
                           intake_epochs, lr_range, slide_id_key='slide_id', mtl_only=False, no_mtl=True):
    os.makedirs(output_dir, exist_ok=True)

    # Load the template
    with open(template_path, "r") as template_file:
        template = template_file.read()

    # Sort to run in correct sequence
    model_name_list = sorted(model_name_list)
    lr_range = sorted(lr_range)

    if len(lr_range) > 1:
        assert fold_range == 1    # only try multiple learning rates in single fold

    # make MTL task
    mtl_task = '%'.join(tasks_to_run)
    if mtl_only:
        tasks_to_run = [mtl_task]
    elif no_mtl:
        pass
    else:
        tasks_to_run.append(mtl_task)

    gpu_idx = 0
    gpu_script_dict = {}    # divided by gpu, each gpu run sequentially
    for task in tasks_to_run:
        for model_name in model_name_list:
            for fold in range(1, fold_range+1):
                for lr in lr_range:

                    # select gpu
                    gpu = gpu_list[gpu_idx]
                    gpu_script_dict[gpu] = [] if gpu not in gpu_script_dict else gpu_script_dict[gpu]
                    gpu_idx = (gpu_idx + 1) % len(gpu_list)

                    # select valid batchsize and max tiles
                    task_batch_size = 1 if model_name in ['SETMIL', 'LongNet'] else batch_size
                    task_max_tiles = 1000 if model_name in ['SETMIL'] else max_tiles

                    # get model weight path
                    local_weight_path = get_local_weight_path(model_name) if use_weight_path else "False"

                    # Replace placeholders in the template

                    # script specs
                    task_name = task if len(task) < 50 else 'MTL'
                    script_name = f'fold-{fold}_{model_name}_{task_name}_lr-{lr:.0e}'
                    script_content = template.replace("${GENERATED_SCRIPT}", script_name)
                    script_content = script_content.replace("${SELECTED_GPU}", str(gpu))
                    script_content = script_content.replace("${CONDA_ACTIVATE_SRC}", str(conda_activate_src))
                    script_content = script_content.replace("${CONDA_ENV_NAME}", str(conda_env_name))

                    # dataset specs
                    script_content = script_content.replace("${DATASET_NAME}", str(dataset))
                    script_content = script_content.replace("${TASK_NAME}", str(task_name))
                    script_content = script_content.replace("${MODEL_NAME}", str(model_name))
                    script_content = script_content.replace("${CSV_FNAME}", str(csv_fname))
                    script_content = script_content.replace("${FOLD}", str(fold))
                    script_content = script_content.replace("${ROI_FEATURE_DIM}", str(roi_feature_dim))
                    script_content = script_content.replace("${SLIDE_ID_KEY}", str(slide_id_key))

                    # task specs
                    script_content = script_content.replace("${TASK_TO_RUN}", str(task))

                    # model specs
                    script_content = script_content.replace("${LOCAL_WEIGHT_PATH}", str(local_weight_path))

                    # training specs
                    script_content = script_content.replace("${MAX_TILES}", str(task_max_tiles))
                    script_content = script_content.replace("${BATCH_SIZE}", str(task_batch_size))
                    script_content = script_content.replace("${NUM_WORKERS}", str(num_workers))
                    script_content = script_content.replace("${NUM_EPOCHS}", str(num_epochs))
                    script_content = script_content.replace("${WARMUP_EPOCHS}", str(warmup_epochs))
                    script_content = script_content.replace("${INTAKE_EPOCHS}", str(intake_epochs))
                    script_content = script_content.replace("${LR}", f'{lr:.0e}')

                    # Write the individual script
                    script_path = os.path.join(output_dir, f'{script_name}_train.sh')
                    with open(script_path, "w") as script_file:
                        script_file.write(script_content)

                    gpu_script_dict[gpu].append(script_name)
    
    for gpu in gpu_script_dict:
        gpu_script_dict[gpu].sort()

    for gpu in gpu_script_dict:
        master_script_path = os.path.join(output_dir, f"run_all_scripts_gpu-{gpu}.sh")
        with open(master_script_path, "w") as master_script:
            master_script.write("#!/bin/bash\n\n")
            master_script.write(f"# Master script to run all task scripts sequentially on gpu {gpu}\n\n")
            master_script.write(f"source {conda_activate_src}\n")
            master_script.write(f"conda activate {conda_env_name}\n\n")
            master_script.write("set -e\n\n")

            # Add the script to the master script
            for script_name in gpu_script_dict[gpu]:
                master_script.write(f'bash {script_name}_train.sh 2>&1 | tee ./{script_name}_train.log\n')

            master_script.write("\n\nset +e\n\n")
        print(f"Master script for gpu {gpu}: {master_script_path}")

    # Make all scripts executable
    os.system(f"chmod +x {output_dir}/*.sh")
    print(f"Scripts generated in: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate experiment scripts for deep learning tasks.")
    
    parser.add_argument("--conda_activate_src", type=str, default="/opt/conda/bin/activate", help="Conda activate path.")
    parser.add_argument("--conda_env_name", type=str, default="BigModel", help="Conda env name.")

    parser.add_argument("--model_names", required=True, type=str, help="Comma-separated list of model names.")
    parser.add_argument("--tasks", required=True, type=str, help="Comma-separated list of task names.")
    parser.add_argument("--lr_range", required=True, type=str, help="Comma-separated list of learning rates.")

    parser.add_argument("--template_path", required=True, type=str, help="Path to the template script.")
    parser.add_argument("--output_name", required=True, type=str, help="Directory prefix to save generated scripts.")
    parser.add_argument("--gpu_list", required=True, type=str, help="Comma-separated list of GPU IDs (e.g., 0,1,2).")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name.")
    parser.add_argument("--csv_fname", required=True, type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--fold_range", default=1, type=int, help="Number of folds for cross-validation.")
    parser.add_argument("--slide_id_key", default="slide_id", type=str, help="Slide ID key.")
    
    parser.add_argument("--roi_feature_dim", default=1536, type=int, help="Feature dimension of ROI.")
    parser.add_argument("--max_tiles", default=2000, type=int, help="Maximum number of tiles.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--warmup_epochs", default=20, type=int, help="Number of warm-up epochs.")
    parser.add_argument("--intake_epochs", default=50, type=int, help="Number of intake epochs.")

    return parser.parse_args()


def gen_scripts_for_dataset(args):
    model_names = args.model_names.split("%")
    tasks = args.tasks.split("%")
    lr_range = [float(lr) for lr in args.lr_range.split("%")]

    template_path = args.template_path
    output_name = args.output_name
    gpu_list = args.gpu_list.split("%")
    dataset_name = args.dataset
    csv_fname = args.csv_fname
    fold_range = args.fold_range
    slide_id_key = args.slide_id_key

    # STL
    generate_shell_scripts(
        template_path=template_path,
        output_dir=f'{output_name}_stl',
        gpu_list=gpu_list,
        conda_activate_src=args.conda_activate_src,
        conda_env_name=args.conda_env_name,

        dataset=dataset_name,
        csv_fname=csv_fname,
        fold_range=fold_range,

        model_name_list = model_names,
        tasks_to_run=tasks,

        use_weight_path=True,

        roi_feature_dim = args.roi_feature_dim,
        max_tiles=args.max_tiles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        intake_epochs=args.intake_epochs,
        lr_range=lr_range,

        slide_id_key=slide_id_key,

        mtl_only=False,
        no_mtl=True
    )

    # MTL
    generate_shell_scripts(
        template_path=template_path,
        output_dir=f'{output_name}_mtl',
        gpu_list=gpu_list,
        conda_activate_src=args.conda_activate_src,
        conda_env_name=args.conda_env_name,

        dataset=dataset_name,
        csv_fname=csv_fname,
        fold_range=fold_range,

        model_name_list = model_names,
        tasks_to_run=tasks,

        use_weight_path=True,

        roi_feature_dim = args.roi_feature_dim,
        max_tiles=args.max_tiles,
        batch_size=1,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        intake_epochs=args.intake_epochs,
        lr_range=lr_range,

        slide_id_key=slide_id_key,

        mtl_only=True
    )


if __name__ == "__main__":
    args = parse_arguments()
    gen_scripts_for_dataset(args)


"""
# TCGA-BLCA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_blca" \
  --gpu_list "0%1" \
  --dataset "TCGA-BLCA" \
  --csv_fname "task_description.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE%GRADE%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-BRCA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_brca" \
  --gpu_list "0%1" \
  --dataset "TCGA-BRCA" \
  --csv_fname "task_description_tcga-brca_20241206.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%IHC_HER2%HISTOLOGICAL_DIAGNOSIS%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-lung
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_lung" \
  --gpu_list "0%1" \
  --dataset "TCGA-lung" \
  --csv_fname "task_description_tcga-lung_reduced_20241203.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "FEV1_PERCENT_REF_POSTBRONCHOLIATOR%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-CESC
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_cesc" \
  --gpu_list "0%1" \
  --dataset "TCGA-CESC" \
  --csv_fname "task_description.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "GRADE%LYMPHOVASCULAR_INVOLVEMENT%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-MESO
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_meso" \
  --gpu_list "0%1" \
  --dataset "TCGA-MESO" \
  --csv_fname "task_description_tcga-meso_20241217.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%HISTOLOGICAL_DIAGNOSIS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-UCEC
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_ucec" \
  --gpu_list "0%1" \
  --dataset "TCGA-UCEC" \
  --csv_fname "task_description_tcga-ucec_20241218.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "GRADE%HISTOLOGICAL_DIAGNOSIS%TUMOR_INVASION_PERCENT%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-UCS
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_ucs" \
  --gpu_list "0%1" \
  --dataset "TCGA-UCS" \
  --csv_fname "task_description_tcga-ucs_20241219.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "HISTOLOGICAL_DIAGNOSIS%OS_STATUS%TUMOR_INVASION_PERCENT" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# TCGA-UVM
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_uvm" \
  --gpu_list "0%1" \
  --dataset "TCGA-UVM" \
  --csv_fname "task_description_tcga-uvm_20241219.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "HISTOLOGICAL_DIAGNOSIS%OS_STATUS%AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%TUMOR_THICKNESS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# CAMELYON16
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "camelyon16" \
  --gpu_list "2%3" \
  --dataset "CAMELYON16" \
  --csv_fname "task_description.csv" \
  --fold_range 1 \
  --model_names "SlideMax%gigapath%SlideVPT" \
  --tasks "BREAST_METASTASIS" \
  --lr_range "1e-5%1e-4%1e-3" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 32 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "fileId" \
  --conda_activate_src "/root/miniforge3/bin/activate"


# PANDA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "panda_validate" \
  --gpu_list "1%2%3" \
  --dataset "PANDA" \
  --csv_fname "task_description_20241230.csv" \
  --fold_range 1 \
  --model_names "ABMIL" \
  --tasks "isup_grade" \
  --lr_range "1e-7%1e-6%1e-5%1e-4%1e-3" \
  --roi_feature_dim 1536 \
  --max_tiles 10000 \
  --batch_size 16 \
  --num_workers 4 \
  --num_epochs 400 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "image_id" \
  --conda_activate_src "/root/miniforge3/bin/activate"


# TCGA-lung-resnet18
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "tcga_lung_resnet18" \
  --gpu_list "0%1" \
  --dataset "TCGA-lung" \
  --csv_fname "task_description_tcga-lung_20250108.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%FEV1_PERCENT_REF_POSTBRONCHOLIATOR%lung-cancer-subtyping%TUMOR_STATUS%OS_MONTHS" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 512 \
  --max_tiles 10000 \
  --batch_size 4 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "slide_id"

# PANDA
python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "panda" \
  --gpu_list "0%1" \
  --dataset "PANDA" \
  --csv_fname "task_description_20241230.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "isup_grade%gleason_score" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "image_id" \
  --conda_activate_src "/root/miniforge3/bin/activate"

python init_scripts.py \
  --template_path "template_script.sh" \
  --output_name "panda" \
  --gpu_list "0%1" \
  --dataset "PANDA" \
  --csv_fname "task_description_20241230.csv" \
  --fold_range 1 \
  --model_names "SlideAve%ABMIL%DSMIL%CLAM%LongNet%TransMIL%SlideViT" \
  --tasks "isup_grade" \
  --lr_range "1e-6%1e-5%1e-4" \
  --roi_feature_dim 1536 \
  --max_tiles 2000 \
  --batch_size 1 \
  --num_workers 64 \
  --num_epochs 100 \
  --warmup_epochs 20 \
  --intake_epochs 50 \
  --slide_id_key "image_id" \
  --conda_activate_src "/root/miniforge3/bin/activate"

"""