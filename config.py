import argparse
import os

import pandas as pd
import torch
from App_dangerrousness import (BERTClassifier, TextClassificationDataset,
                                evaluate, predict_sentiment, train)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)
from util import load_data, predict_batch

# Set up parameters
bert_model_name = "bert-base-uncased"
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model_path = "bert_classifier.pth"
