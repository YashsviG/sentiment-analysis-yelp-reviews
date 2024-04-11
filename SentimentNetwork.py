import numpy as np
import sklearn.metrics
import torch
import transformers
from transformers import AutoTokenizer
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from YelpDataset import YelpDataset

MIN_FREQUENCY = 5
SPECIAL_TOKENS = ["<unk>", "<pad>"]


class SentimentNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_index, is_classification, n_filters, filter_sizes, dropout_rate):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_index)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, n_filters, filter_size)
                for filter_size in filter_sizes
            ]
        )

        self.is_classification = is_classification
        if is_classification:
            output_dim = 5
        else:
            output_dim = 1

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        embedded = embedded.permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [conv.max(dim=-1).values for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        prediction = self.fc(cat)
        return prediction
