import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from collections import Counter
from src.constants import MAX_VOCAB_SIZE, MAX_SEQ_LEN, BATCH_SIZE
from torch.utils.data import TensorDataset, DataLoader
import os

DATA_PATH = os.getenv(
    "DATA_PATH",
    "E:/datasets/IMDB Dataset.csv"
)

def load_imdb_data(path = DATA_PATH):
    df = pd.read_csv(path)
    X_numpy = df["review"].values
    y_numpy = df["sentiment"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
    X_numpy,
    y_numpy,
    test_size=0.2,
    random_state=42,
    stratify=y_numpy
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def tokenize(text):
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = text.lower()
    return text.split()

def build_vocab(X_train):
    vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    }
    counter = Counter()

    for review in X_train:
        tokens = tokenize(review)
        counter.update(tokens)
    for word, count in counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab

def pad_sequence(sequence, max_len, pad_id=0):
    sequence = sequence[:max_len]
    if len(sequence) < max_len:
        sequence += [pad_id] * (max_len - len(sequence))
    return sequence

def text_to_id(text, vocab):
    tokens = tokenize(text)
    return[
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
        ]

def prepare_data():
    X_train, X_val, X_test, y_train, y_val, y_test = load_imdb_data()
    vocab = build_vocab(X_train)
    X_train_ids = [text_to_id(text, vocab) for text in X_train]
    X_val_ids = [text_to_id(text, vocab) for text in X_val]
    X_test_ids = [text_to_id(text, vocab) for text in X_test]

    X_train_padded = [
    pad_sequence(seq, MAX_SEQ_LEN)
    for seq in X_train_ids
    ]

    X_val_padded = [
        pad_sequence(seq, MAX_SEQ_LEN)
        for seq in X_val_ids
    ]

    X_test_padded = [
        pad_sequence(seq, MAX_SEQ_LEN)
        for seq in X_test_ids
    ]

    X_train_tensor = torch.tensor(
    X_train_padded,
    dtype=torch.long
    )

    X_val_tensor = torch.tensor(
        X_val_padded,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test_padded,
        dtype=torch.long
    )

    label_to_id ={
    "positive": 1,
    "negative": 0
    }

    y_train = np.array([label_to_id[label] for label in y_train])
    y_val = np.array([label_to_id[label] for label in y_val])
    y_test = np.array([label_to_id[label] for label in y_test])
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

def create_dataloaders():
    X_train_tensor, X_val_tensor, _, y_train_tensor, y_val_tensor, _ = prepare_data()
    train_dataset = TensorDataset(
        X_train_tensor, y_train_tensor
    )

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, val_loader