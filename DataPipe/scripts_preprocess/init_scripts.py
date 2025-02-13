import os

def generate_shell_scripts(template_path, datasets, output_dir, batch_size, model_name, edge_size=224, target_mpp=0.5, gpu=0):
    """
    Generates shell scripts for processing datasets using a template.

    Args:
        template_path (str): Path to the shell script template.
        datasets (list): List of dataset names.
        output_dir (str): Directory to save the generated scripts.
        batch_size (int): Batch size for embedding.
        model_name (str): Model name for embedding.
        edge_size (int): Edge size for tiling. Default is 224.
        target_mpp (float): Target MPP for tiling. Default is 0.5.
        gpu (int): GPU to use. Default is 0.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the template
    with open(template_path, "r") as template_file:
        template = template_file.read()

    master_script_path = os.path.join(output_dir, "run_all_scripts.sh")
    with open(master_script_path, "w") as master_script:
        master_script.write("#!/bin/bash\n\n")
        master_script.write("# Master script to run all dataset processing scripts sequentially\n\n")
        # master_script.write("source /root/miniforge3/bin/activate\n")
        # master_script.write("conda activate gigapath\n\n")
        master_script.write("set -e\n\n")

        # Sort to run in correct sequence
        datasets = sorted(datasets)

        for dataset in datasets:
            # Replace placeholders in the template
            script_content = template.replace("GENERATED_SCRIPT", dataset)
            script_content = script_content.replace("DATASET_NAME", dataset)
            script_content = script_content.replace("BATCH_SIZE", str(batch_size))
            script_content = script_content.replace("MODEL_NAME", model_name)
            script_content = script_content.replace("EDGE_SIZE", str(edge_size))
            script_content = script_content.replace("TARGET_MPP", str(target_mpp))
            script_content = script_content.replace("SELECTED_GPU", str(gpu))

            # Write the individual script
            script_name = f"{dataset}_prepare_dataset.sh"
            script_path = os.path.join(output_dir, script_name)
            with open(script_path, "w") as script_file:
                script_file.write(script_content)

            # Add the script to the master script
            master_script.write(f'bash {script_name} 2>&1 | tee ./{dataset}_prepare_dataset.log\n')

        master_script.write("\n\nset +e\n\n")

    # Make all scripts executable
    os.system(f"chmod +x {output_dir}/*.sh")
    print(f"Scripts generated in: {output_dir}")
    print(f"Master script: {master_script_path}")

# Example Usage: to process all the TCGA datasets
# Two additional datasets:
# TCGA-lung: TCGA-LUAD + TCGA-LUSC
# TCGA-COAD-READ: TCGA-COAD + TCGA-READ
if __name__ == "__main__":
    # Define datasets and parameters
    datasets = [
        "TCGA-ACC", "TCGA-COAD", "TCGA-KICH", "TCGA-LUAD", "TCGA-PCPG", "TCGA-STAD", "TCGA-UCS",
        "TCGA-BLCA", "TCGA-DLBC", "TCGA-KIRC", "TCGA-LUSC", "TCGA-PRAD", "TCGA-TGCT", "TCGA-UVM",
        "TCGA-BRCA", "TCGA-ESCA", "TCGA-KIRP", "TCGA-MESO", "TCGA-READ", "TCGA-THCA",
        "TCGA-CESC", "TCGA-GBM", "TCGA-LGG", "TCGA-OV", "TCGA-SARC", "TCGA-THYM",
        "TCGA-CHOL", "TCGA-HNSC", "TCGA-LIHC", "TCGA-PAAD", "TCGA-SKCM", "TCGA-UCEC"
    ]
    # Remove some datasets
    to_remove = ["TCGA-COAD", "TCGA-READ"]
    datasets = [item for item in datasets if item not in to_remove]

    # datasets = ['PANDA']

    generate_shell_scripts(
        template_path="template_tile_and_embed_tcga_resnet18.sh",
        datasets=datasets,
        output_dir="generated_scripts",
        batch_size=2048,
        model_name="gigapath",
        gpu=1
    )