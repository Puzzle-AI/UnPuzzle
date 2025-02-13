"""
This script provides a temporary solution to modify the original task configuration CSV file, 
ensuring it is compatible with the updated format of the embedded dataset.

Version: 2025.01.08 15:30
"""
import pandas as pd

csv_file = '/data/ssd_1/WSI/TCGA-lung/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_reduced_20241203.csv'
new_csv_file = '/data/ssd_1/WSI_resnet18/TCGA-lung/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_20250108.csv'

content = pd.read_csv(csv_file)

content['slide_id'] = content['slide_id'].apply(lambda x: x[:23])

content.to_csv(new_csv_file, index=False)