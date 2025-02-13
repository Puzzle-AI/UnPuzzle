"""
Build VQA dataset   Script  ver： Sep 25th 15:00
"""
import os
import torch
from transformers import GPT2Tokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def get_tokenizer(tokenizer_name='gpt2'):
    if tokenizer_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Padding is used to ensure that all input sequences in a batch are of the same length.
        # EOS (End of sequence model) make the end of a seq to let model know input text is done
        tokenizer.pad_token = tokenizer.eos_token  # pad with eos, (use eos_token as pad_token)
    else:
        raise NotImplementedError
    return tokenizer


# ------------------- Dataset Class for VQA -------------------
class Tile_VQA_Dataset(Dataset):
    def __init__(self, dataframe, image_folder, tokenizer_name='gpt2',
                 max_seq_length=256, img_size=224, transform=None, answer_to_index=None):
        super().__init__(self)

        self.dataframe = dataframe
        self.image_folder = image_folder
        self.img_size = img_size
        self.max_seq_length = max_seq_length

        # ------------------- Image Preprocessing Function -------------------
        # default_transform is only resize and to tensor
        default_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        # specified slide_feature image Transform can be assigned
        self.transform = transform or default_transform

        # ------------------- Text Preprocessing Function -------------------
        # The GPT-2 model operates on tokenized input, which is essentially converting text into sequences of
        # integers that represent individual tokens (words or subwords).
        self.answer_to_index = answer_to_index
        # Calling tokenizer ensures that the input text is properly formatted and tokenized in the same way
        # the GPT-2 model was trained, which is critical for effective performance.
        self.tokenizer = get_tokenizer(tokenizer_name)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # fetch question, answer and image path from the dataset
        item = self.dataframe.iloc[idx]
        image_path = f"{self.image_folder}/{item['image']}"
        question = item['question']
        answer = item['answer']  # Use preprocessed answer

        # Image Preprocessing
        img = Image.open(image_path).convert('RGB')
        # regardless of greyscale or RGB, Convert to RGB as transformer expects RCG 3 channel input
        img_tensor = self.transform(img)
        # todo make here to fit both roi and wsi

        # Tokenize the question using GPT-2 tokenizer
        inputs = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length
        )

        # Map the processed answer to an integer index for classification
        if answer in self.answer_to_index:
            answer_idx = self.answer_to_index[answer]
        else:
            # Handle missing or unknown answers by assigning a default valid class
            answer_idx = 0  # Or any valid class index

        return {
            'image': img_tensor,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(answer_idx, dtype=torch.long)
        }


# Custom Collate Function for Batch stacking
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item['labels'] for item in batch])

    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
