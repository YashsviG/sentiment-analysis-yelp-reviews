from io import StringIO

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class YelpDataset(Dataset):
    def __init__(self, json_file, chunk_size, tokenizer_name):
        self.json_file = json_file
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(json_file, 'r') as file:
            self.length = sum(1 for _ in file)

    def calculate_length(self):
        json_reader = pd.read_json(self.json_file, lines=True, chunksize=self.chunk_size)
        return sum(1 for chunk in json_reader for _ in chunk)

    def __len__(self):
        return (self.length + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, idx):
        chunk_data = []
        start_line = idx * self.chunk_size
        end_line = start_line + self.chunk_size
        current_line = 0
        with open(self.json_file, 'r') as file:
            for line in file:
                if start_line <= current_line < end_line:
                    chunk_data.append(pd.read_json(StringIO(line), lines=True))
                current_line += 1
                if current_line >= end_line:
                    break

        chunk = pd.concat(chunk_data, ignore_index=True)
        review_values = chunk.iloc[:, 4]
        review_values = review_values[review_values != ""]
        tokenized_text = self.tokenizer(review_values.tolist(), padding=True, truncation=True, max_length=512)

        X = torch.tensor(tokenized_text["input_ids"])
        y = torch.tensor(chunk.iloc[:, 0].values)
        z = torch.tensor(chunk.iloc[:, 1:4].values)
        return X, y, z
