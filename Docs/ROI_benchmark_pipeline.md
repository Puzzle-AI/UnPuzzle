# ROI Classification Benchmark Pipeline
> Notice: some content of this file is not up-to-date, please await the release of our most recent version!

Version: Dec 19th 2024

This doc states how we process & run a ROI benchmark from scratch.

## **1. Overall Pipeline**
prepare dataset -> train&test -> summary results


## **2. Important Paths**

**Results:** you can put them at `/data/private/XXX/`.


# **3. Prepare Dataset**
The standard directory structure of the dataset is shown as follows.
If your directory does not have the following structure, please refer to [3.1 Split Dataset](#31-split-dataset); otherwise, skip to [4. Run Benchmark Pipeline](#4-run-benchmark-pipeline).
```
DATASET_NAME
├── DATASET_NAME
│   ├── test
│   │   ├── Class 1
│   │   ├── Class 2
│   │   ├── Class 3
│   ├── train
│   │   ├── Class 1
│   │   ├── Class 2
│   │   ├── Class 3
│   └── val
│   │   ├── Class 1
│   │   ├── Class 2
│   │   ├── Class 3
└── task-settings
│   ├── task_configs.yaml
```


### **3.1 Split Dataset**
If the directory structure of the dataset is as follows, you can use the `Utils/Split_ROI_dataset.py` script to split the dataset.
```
├── DATASET_NAME
│   ├── Class 1
│   ├── Class 2
│   ├── Class 3
```


## **4. Run Benchmark Pipeline**

### **4.1 Locate Pipeline Scripts**  
   The pipeline scripts are in:  
   `BigModel/DownStream/ROI_finetune/scripts_exp/`


### **4.2 Modify `init_scripts.py`**
   - Inside `init_scripts.py`, `generate_shell_scripts` function generates scripts for a specific dataset.
   - The parameters of `generate_shell_scripts` function which you are likely to modify are as follows:
      * `output_dir`: the directory for saving scripts.
      * `gpu_list`: which gpus to use, if -1 for using all gpus.
      * `dataset`: the parent directory of train dataset. 
         For example, `/data/hdd_1/DevDatasets/ROI/Gleason_2019/Gleason_2019`
      * `task_name`: the parameter for naming the result folder, usually an abbreviation of the dataset. For example, `Gleason`.


### **4.3 Generate Scripts:**  
   - Run:  
     ```bash
     python init_scripts.py
     ```
   - The scripts will be generated in `output_dir` you set inside `init_scripts.py`:


### **4.4 Run Benchmark Experiments:**  
   - Navigate to `output_dir` and locate scripts like `run_all_scripts_gpu_n`, where `n` is the GPU index.
   - Use `tmux` to start background terminals and execute:  
     ```bash
     nohup bash run_all_scripts_gpu_n.sh >/dev/null 2>&1 &
     ```

## **5. Gather Results**

### **5.1 Locate Results**
- The results will be saved in `/data/private/xxx/CLS_{TASK_NAME}/{MODEL_NAME}_lr_{LR_RATE}/`
**NOTICE**: The "{ }" part in the path will vary depending on different datasets, models, and learning rates.
   

### **5.2 Summarize Results to a CSV File**

Locate script `Utils/`, and execute:
```shell
python Decode_ROI_pred.py --run_root {RUN_ROOT} --task_name {TASK_NAME} --save_path {SAVE_PATH}
```
**NOTICE**: The "{ }" part in the path will vary depending on the settings your configured above.

---
# Additional Functions

1. Run ROI level SSL pretraining

```Shell
# Using DDP
python -m torch.distributed.launch --nproc_per_node=12 --nnodes 1 --node_rank 0 PuzzleTuning.py --DDP_distributed --model sae_vit_base_patch16 --batch_size 64 --group_shuffle_size 8 --blr 1.5e-4 --epochs 20 --accum_iter 2 --print_freq 200 --check_point_gap 5 --input_size 224 --warmup_epochs 10 --pin_mem --num_workers 32 --strategy loop --PromptTuning Deep --basic_state_dict /data/hdd_1/saved_models/ViT_b16_224_Imagenet.pth --data_path /data/hdd_1/CPIA/All

# Or using DP (not recomended)
python PuzzleTuning.py --model sae_vit_base_patch16 --batch_size 64 --group_shuffle_size 8 --blr 1.5e-4 --epochs 20 --accum_iter 2 --print_freq 200 --check_point_gap 5 --input_size 224 --warmup_epochs 10 --pin_mem --num_workers 32 --strategy loop --PromptTuning Deep --basic_state_dict /data/hdd_1/saved_models/ViT_b16_224_Imagenet.pth --data_path /data/hdd_1/CPIA/All
```

2. Run ROI level VQA application

```Shell
# todo
```