
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from util import load_data, predict_batch
from App_dangerrousness import BERTClassifier, TextClassificationDataset,train,evaluate, predict_sentiment
import argparse


# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model_path = "bert_classifier.pth"