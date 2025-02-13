"""
Version: Dec 29 2024 9:00
"""

from tools.analyse_func import *
from tools.tools import * 


def analyse_dataset(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    if dataset_name == 'TCGA-lung':
        analyse_tcga_lung(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-BRCA':
        analyse_tcga_brca(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-MESO':
        analyse_tcga_meso(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-UCEC':
        analyse_tcga_ucec(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-UCS':
        analyse_tcga_ucs(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-UVM':
        analyse_tcga_uvm(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-BLCA':
        analyse_tcga_blca(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-CESC':
        analyse_tcga_cesc(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-CHOL':
        analyse_tcga_chol(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'TCGA-DLBC':
        analyse_tcga_chol(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'PANDA':
        analyse_panda(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    elif dataset_name == 'CAMELYON16':
        analyse_camelyon16(dataset_name, subset_name, subset_df, subset_save_path, dpi)
    else:
        raise ValueError(f'Not a valid dataset name: {dataset_name}')


def analyse_tcga_chol(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-CHOL
    """
    print(f"only 51 samples, aborted!")


def analyse_tcga_dlbc(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-DLBC
    """
    print(f"only 48 samples, aborted!")


def analyse_camelyon16(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['fileId'],
        figsize=(10, 6),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['BREAST_METASTASIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )


def analyse_panda(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for PANDA

    Task selection:
    isup_grade:
        Reason:
            Tumor grading involves assessing cell differentiation and architectural patterns of the prostate, 
            which are directly observable in Whole Slide Images (WSIs). A deep learning model can analyze these 
            histopathological features, such as glandular structure and the degree of cell atypia, to predict the ISUP grade.
        Result:
            Selected
    gleason_score:
        Reason:
            The Gleason score is determined by summing the primary and secondary Gleason patterns, which reflect the 
            aggressiveness of prostate cancer. These patterns are identified based on the structure and arrangement of 
            cancerous glands in WSIs. A deep learning model can identify these patterns and calculate the Gleason score effectively.
        Result:
            Selected
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['isup_grade', 'gleason_score'],
        figsize=(10, 6),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['isup_grade'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['gleason_score'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols=['isup_grade', 'gleason_score'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )



def analyse_tcga_cesc(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-CESC

    Task selection:
    AJCC_TUMOR_PATHOLOGIC_PT:
        Reason:
            The tumor's pathologic staging (e.g., tumor size, local invasion) can often be predicted by analyzing morphological features in WSIs. 
            Deep learning models are highly capable of classifying and segmenting tumor regions to determine their stage.
        Result:
            Too much classes, not enough data, aborted.
    LYMPHOVASCULAR_INVOLVEMENT:
        Reason:
            WSIs can show the presence of tumor cells in lymphovascular spaces, which is critical for understanding metastasis. 
            Models trained on labeled data can identify such features.
        Result:
            Selected
    HISTOLOGICAL_DIAGNOSIS:
        Reason:
            The diagnosis of different histological subtypes of cervical cancer is one of the primary uses of WSIs. 
            Deep learning models are effective at classifying different histological types based on tissue and cell morphology.
        Result:
            Too much classes, not enough data, aborted.
    LYMPH_NODES_EXAMINED:
        Reason:
            WSIs of lymph nodes can be analyzed to determine the presence of metastatic cancer. This is crucial for staging and prognosis.
        Result:
            Selected
    GRADE:
        Reason:
            Tumor grading involves assessing cell differentiation and structure, which is directly observable in WSIs. A deep learning model can learn these patterns to predict the tumor grade.
        Result:
            Selected
    OS_MONTHS:
        Reason:
            Predicting survival outcomes from WSIs requires extracting features such as tumor morphology, immune cell infiltration, and other histological patterns associated with prognosis. 
            These tasks can be approached using survival prediction models that incorporate features derived from WSIs.
        Result:
            Selected
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['AJCC_TUMOR_PATHOLOGIC_PT', 
                        'LYMPHOVASCULAR_INVOLVEMENT', 'LYMPH_NODES_EXAMINED', 'HISTOLOGICAL_DIAGNOSIS', 'OS_MONTHS', 'GRADE'],
        figsize=(10, 6),
        dpi=dpi
    )
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_TUMOR_PATHOLOGIC_PT'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['LYMPHOVASCULAR_INVOLVEMENT'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['LYMPH_NODES_EXAMINED'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['GRADE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 

    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['OS_MONTHS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )  

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols=['AJCC_TUMOR_PATHOLOGIC_PT', 
                        'LYMPHOVASCULAR_INVOLVEMENT', 'LYMPH_NODES_EXAMINED', 'HISTOLOGICAL_DIAGNOSIS', 'OS_MONTHS', 'GRADE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )


def analyse_tcga_blca(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-BLCA
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
                        'CLIN_T_STAGE', 'HISTOLOGICAL_DIAGNOSIS', 'OS_MONTHS', 'GRADE'],
        figsize=(10, 6),
        dpi=dpi
    )
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['CLIN_T_STAGE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['GRADE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 

    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['OS_MONTHS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )  

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
                        'CLIN_T_STAGE', 'HISTOLOGICAL_DIAGNOSIS', 'OS_MONTHS', 'GRADE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )


def analyse_tcga_meso(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-MESO
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 'HISTOLOGICAL_DIAGNOSIS'],
        figsize=(10, 6),
        dpi=dpi
    )
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )


def analyse_tcga_brca(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-BRCA
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
                        'IHC_HER2', 'HISTOLOGICAL_DIAGNOSIS', 'OS_MONTHS'],
        figsize=(10, 6),
        dpi=dpi
    )
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['IHC_HER2'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 

    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['OS_MONTHS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )  

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
                        'IHC_HER2', 'HISTOLOGICAL_DIAGNOSIS', 'OS_MONTHS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )



def analyse_tcga_lung(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-lung
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols=['lung-cancer-subtyping', 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
                        'OS_MONTHS', 'FEV1_PERCENT_REF_POSTBRONCHOLIATOR'],
        figsize=(10, 6),
        dpi=dpi
    )
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols=['lung-cancer-subtyping', 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
                        'OS_MONTHS', 'FEV1_FVC_RATIO_POSTBRONCHOLIATOR', 'FEV1_PERCENT_REF_POSTBRONCHOLIATOR'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )
    
    print(f"--- Generating <count and boxplots>  for {dataset_name}-{subset_name}.")
    plot_count_and_boxplot(
        subset_df,
        x_var='lung-cancer-subtyping',
        y_var='OS_MONTHS',
        plot_name=f"{dataset_name}-subtyping-OS_MONTHS",
        save_path=subset_save_path,
        sel_palette='Blues',
        dpi=dpi
    )
    plot_count_and_boxplot(
        subset_df,
        x_var='AJCC_PATHOLOGIC_TUMOR_STAGE_reduced',
        y_var='OS_MONTHS',
        plot_name=f"{dataset_name}-subtyping-staging",
        save_path=subset_save_path,
        sel_palette='Blues',
        dpi=dpi
    )
    
    print(f"--- Generating <pairplot>  for {dataset_name}-{subset_name}.")
    plot_pairplot(
        subset_df,
        selected_columns=['OS_STATUS', 'FEV1_PERCENT_REF_POSTBRONCHOLIATOR', 'FEV1_PERCENT_REF_PREBRONCHOLIATOR','OS_MONTHS','SMOKING_PACK_YEARS','TOBACCO_SMOKING_HISTORY_INDICATOR'],
        hue_var='OS_STATUS',
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )

    # -------------------------------
    # Cls tasks
    # -------------------------------
    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    ) 

    # -------------------------------
    # Reg tasks
    # -------------------------------
    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['FEV1_PERCENT_REF_POSTBRONCHOLIATOR'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )  

def analyse_tcga_ucec(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-UCEC
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols = [
            'GRADE', 
            'HISTOLOGICAL_DIAGNOSIS', 
            'AJCC_STAGING_EDITION', 
            'TUMOR_STATUS', 
            'TUMOR_INVASION_PERCENT', 
            'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
            'OS_MONTHS'
        ],
        figsize=(10, 6),
        dpi=dpi
    )

    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['GRADE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_STAGING_EDITION'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['TUMOR_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['TUMOR_INVASION_PERCENT'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_reg(
        subset_df,
        selected_cols=['OS_MONTHS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols = [
            'GRADE', 
            'HISTOLOGICAL_DIAGNOSIS', 
            'AJCC_STAGING_EDITION', 
            'TUMOR_STATUS', 
            'TUMOR_INVASION_PERCENT', 
            'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced', 
            'OS_MONTHS'
        ],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )


def analyse_tcga_ucs(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-UCS
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols = [
            'HISTOLOGICAL_DIAGNOSIS',
            'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced',
            'TUMOR_INVASION_PERCENT',
            'TUMOR_STATUS',
            'OS_STATUS',
            'DFS_STATUS',
            'LYMPH_NODES_AORTIC_POS_TOTAL',
            'LYMPH_NODES_PELVIC_POS_TOTAL',
            'RESIDUAL_TUMOR'
        ],
        figsize=(10, 6),
        dpi=dpi
    )

    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    # For classification tasks
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['TUMOR_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['OS_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['DFS_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['RESIDUAL_TUMOR'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['LYMPH_NODES_AORTIC_POS_TOTAL'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['LYMPH_NODES_PELVIC_POS_TOTAL'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    # For regression tasks
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['TUMOR_INVASION_PERCENT'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols = [
            'HISTOLOGICAL_DIAGNOSIS',
            'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced',
            'TUMOR_INVASION_PERCENT',
            'TUMOR_STATUS',
            'OS_STATUS',
            'OS_MONTHS',
            'DFS_STATUS'
        ],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )

def analyse_tcga_uvm(dataset_name, subset_name, subset_df, subset_save_path, dpi):
    """draw figures for single train/val/test set for TCGA-UVM
    """
    print(f"--- Generating <non-missing value bar chart>  for {dataset_name}-{subset_name}.")
    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        selected_cols = [
            'HISTOLOGICAL_DIAGNOSIS',
            'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced',
            'TUMOR_STATUS',
            'OS_STATUS',
            'DFS_STATUS',
            'OS_MONTHS',
            'TUMOR_THICKNESS',
            'AJCC_CLINICAL_TUMOR_STAGE'
        ],
        figsize=(10, 6),
        dpi=dpi
    )

    draw_exist_value_fig(
        subset_df,
        dataset_name=f"{dataset_name}-{subset_name}-all-tasks",
        save_path=subset_save_path,
        figsize=(10, 24),
        dpi=dpi
    )

    print(f"--- Generating <classification task analysis>  for {dataset_name}-{subset_name}.")
    # For classification tasks
    plot_task_analyse_cls(
        subset_df,
        selected_cols=['HISTOLOGICAL_DIAGNOSIS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['TUMOR_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['OS_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['DFS_STATUS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_cls(
        subset_df,
        selected_cols=['AJCC_CLINICAL_TUMOR_STAGE'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    print(f"--- Generating <regression task analysis>  for {dataset_name}-{subset_name}.")
    plot_task_analyse_reg(
        subset_df,
        selected_cols=['TUMOR_THICKNESS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )

    plot_task_analyse_reg(
        subset_df,
        selected_cols=['OS_MONTHS'],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=os.path.join(subset_save_path, 'task_analyse'),
        sel_palette="Blues",
        dpi=dpi
    )


    print(f"--- Generating <correlation heatmap>  for {dataset_name}-{subset_name}.")
    plot_corr_matrix(
        subset_df,
        selected_cols = [
            'HISTOLOGICAL_DIAGNOSIS',
            'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced',
            'TUMOR_STATUS',
            'OS_STATUS',
            'DFS_STATUS',
            'OS_MONTHS',
            'TUMOR_THICKNESS',
            'AJCC_CLINICAL_TUMOR_STAGE'
        ],
        dataset_name=f"{dataset_name}-{subset_name}",
        save_path=subset_save_path,
        dpi=dpi
    )