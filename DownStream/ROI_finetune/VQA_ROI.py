"""
Train VQA by CLS for ROI   Script  ver： Sep 25th 15:00
"""
import os
import sys
from pathlib import Path
# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up two levels

import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from PuzzleAI.DownStream.MLLM.VQA_datasets import Tile_VQA_Dataset, custom_collate_fn
from PuzzleAI.ModelBase.Get_VQA_model import get_VQA_model


# ------------------- Training and Evaluation -------------------
def train_and_validate(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs=10):
    best_val_accuracy = 0.0  # To track the best validation accuracy
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        # Add tqdm to show the progress for each batch
        train_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        for batch in train_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()  # Clear gradients
            # forward
            with torch.cuda.amp.autocast():  # automatic mix precision training
                logits = model(images, input_ids, attention_mask)  # Forward pass
                loss = loss_fn(logits, labels)  # Calculate loss
                total_train_loss += loss.item()
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm description with current loss and accuracy
            train_bar.set_postfix(loss=loss.item(), accuracy=correct_train / total_train)

        # Calculate average training loss and accuracy for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation step at the end of each epoch
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Track the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    return train_losses, train_accuracies, val_losses, val_accuracies


# ------------------- Validation Function (evaluate) -------------------
def evaluate(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss, correct, total = 0, 0, 0

    # Add tqdm to show progress for the validation/test loop
    val_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in val_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(images, input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm description with current loss and accuracy
            val_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == '__main__':
    # ------------------- Setting Up the Constants -------------------
    # Load the Excel file
    excel_path = "/home/changhan/VQA_Histology/Path_VQA/PathVQA.xlsx"
    image_folder = "/home/changhan/VQA_Histology/Path_VQA/Images"

    # Constants
    IMG_SIZE = 224
    MAX_SEQ_LENGTH = 256  # Adjust based on typical question length
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DROP_RATE = 0.1
    HEADS = 8
    EMBED_SIZE = 768

    model_idx = 'uni'
    tokenizer_name = 'gpt2'
    fusion_method = 'MHSA'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------- Prepare the excel dataset -------------------
    # fixme this part Tianyi will build baseded on BigModel later
    # Read Excel file into pandas DataFrame
    df = pd.read_excel(excel_path)

    # Shuffle and split the data into train (75%), validation (15%), and test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42)  # 75% train
    val_df, test_df = train_test_split(temp_df, test_size=0.4, random_state=42)  # 15% validation, 10% test

    # Print dataset sizes to verify
    print(f"Train Size: {len(train_df)}, Validation Size: {len(val_df)}, Test Size: {len(test_df)}")

    # Map each unique answer to an index
    answer_to_index = {ans: idx for idx, ans in enumerate(df['answer'].unique())}
    num_classes = len(answer_to_index)  # Number of classes

    print(f"Number of unique answers: {num_classes}")

    # ------------------- Create Datasets & DataLoaders -------------------
    train_dataset = Tile_VQA_Dataset(train_df, image_folder, answer_to_index=answer_to_index)
    val_dataset = Tile_VQA_Dataset(val_df, image_folder, answer_to_index=answer_to_index)
    test_dataset = Tile_VQA_Dataset(test_df, image_folder, answer_to_index=answer_to_index)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # ------------------- Create Model -------------------
    model = get_VQA_model(model_idx=model_idx, tokenizer_name=tokenizer_name,
                          fusion_method=fusion_method, embed_size=EMBED_SIZE,
                          dropout_rate=DROP_RATE, heads=HEADS, num_classes=num_classes)
    model = torch.compile(model)
    model.to(device)

    # ------------------- Create training config -------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()

    # ------------------- Run the training and validation loop -------------------
    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs=EPOCHS)

    # ------------------- Test Code -------------------
    # Evaluate the model on the test dataset
    test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)

    # Print out test results
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
