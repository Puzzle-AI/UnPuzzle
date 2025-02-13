import os


def generate_shell_scripts(template_path, output_dir, gpu_list, dataset,
                           num_workers, model_name_list, batch_size,
                           task_name, num_epochs, lr_range):
    os.makedirs(output_dir, exist_ok=True)

    # Load the template
    with open(template_path, "r") as template_file:
        template = template_file.read()

    # Sort to run in correct sequence
    model_name_list = sorted(model_name_list)
    lr_range = sorted(lr_range)

    gpu_idx = 0
    gpu_script_dict = {}  # divided by gpu, each gpu run sequentially
    for model_name in model_name_list:
        for num_worker in num_workers:
            for lr in lr_range:
                # select gpu
                gpu = gpu_list[gpu_idx]
                gpu_script_dict[gpu] = [] if gpu not in gpu_script_dict else gpu_script_dict[gpu]
                gpu_idx = (gpu_idx + 1) % len(gpu_list)

                # Replace placeholders in the template

                # script specs
                script_name = f'{model_name}_lr-{lr:.0e}_nw-{num_worker}'
                script_content = template.replace("GENERATED_SCRIPT", script_name)
                script_content = script_content.replace("GPU_IDX", str(gpu))

                # dataset specs
                script_content = script_content.replace("DATASET_NAME", str(dataset))
                script_content = script_content.replace("MODEL_NAME", str(model_name))

                # task specs
                script_content = script_content.replace("TASK_NAME", str(task_name))

                # training specs
                script_content = script_content.replace("NUM_EPOCHS", str(num_epochs))
                script_content = script_content.replace("BATCH_SIZE", str(batch_size))
                script_content = script_content.replace("NUM_WORKERS", str(num_worker))
                script_content = script_content.replace("LR", str(lr))

                # Write the individual script
                script_path = os.path.join(output_dir, f'{script_name}_train.sh')
                with open(script_path, "w") as script_file:
                    script_file.write(script_content)

                gpu_script_dict[gpu].append(script_name)

    for gpu in gpu_script_dict:
        master_script_path = os.path.join(output_dir, f"run_all_scripts_gpu-{gpu}.sh")
        with open(master_script_path, "w") as master_script:
            master_script.write("#!/bin/bash\n\n")
            master_script.write(f"# Master script to run all task scripts sequentially on gpu {gpu}\n\n")
            master_script.write(f"# nohup bash run_all_scripts_gpu-{gpu}.sh >/dev/null 2>&1 &\n\n")
            master_script.write("source /root/miniforge3/bin/activate\n")
            master_script.write("conda activate BigModel\n\n")
            master_script.write("set -e\n\n")

            # Add the script to the master script
            for script_name in gpu_script_dict[gpu]:
                master_script.write(f'bash {script_name}_train.sh 2>&1 | tee ./{script_name}_train.log\n')

            master_script.write("\n\nset +e\n\n")
        print(f"Master script for gpu {gpu}: {master_script_path}")

    # Make all scripts executable
    os.system(f"chmod +x {output_dir}/*.sh")
    print(f"Scripts generated in: {output_dir}")


# Example Usage
if __name__ == "__main__":
    # "ViT_h", "UNI", "VPT", "ResNet101", "Virchow", "mobilevit_s", "swin_b_224"
    # 1.00E-06, 1.00E-05, 1.00E-04
    generate_shell_scripts(
        template_path="template_script.sh",
        output_dir="/home/workenv/BigModel/DownStream/ROI_finetune/scripts_exp/SipakMed_scripts",
        gpu_list=[-1],
        dataset="/data/hdd_1/DevDatasets/ROI/SipakMed/SipakMed",
        task_name='SipakMed',
        # model_name_list=["UNI", "VPT","ViT_h", "Virchow", "ResNet101", "mobilevit_s", "swin_b_224"],
        #model_name_list = ["Virchow"],
        model_name_list = ["UNI", "VPT"],
        num_epochs= 100,
        batch_size= 64,
        num_workers=[8],
        lr_range=[1.00E-06, 1.00E-05, 1.00E-04]
        #lr_range=[1.00E-05]
    )
