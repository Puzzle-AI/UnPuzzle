import os
import torch
import timm
import torch.nn as nn
from tqdm import tqdm
from timm.layers import SwiGLUPacked
from transformers import GPT2Tokenizer, GPT2Model, ViTModel
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import re
from word2number import w2n  # Optional if you are converting words to numbers
from huggingface_hub import login


def build_language_model(tokenizer_name='gpt2'):
    if tokenizer_name == 'gpt2':
        language_model = GPT2Model.from_pretrained('gpt2')
    else:
        raise NotImplementedError
    return language_model
