import transformers
from torch import nn
from torch.utils.data import DataLoader
from YelpDataset import YelpDataset

class SentimentNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, hidden):
        print('forward')


def tokenize(data_set, _tokenizer):
    ids = _tokenizer(data_set["text"], truncation=True)["input_ids"]
    return {"ids": ids}

if __name__ == "__main__":
    chunk_size = 100
    transformer_name = "bert-base-uncased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)

    training_data = YelpDataset("../train_data.json", chunk_size)
    training_data = training_data.map(
        tokenize, fn_kwargs={"_tokenizer": tokenizer}
        )
    train_dataloader = DataLoader(training_data, batch_size=chunk_size, shuffle=True)

    test_data = YelpDataset("../test_data.json", chunk_size)
    test_dataloader = DataLoader(test_data, batch_size=chunk_size, shuffle=True)

