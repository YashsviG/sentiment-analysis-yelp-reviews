import collections

import numpy as np
import torch
import torchtext
from tqdm import tqdm
import transformers
from torch import nn, tanh, device, cuda, optim
from torch.utils.data import DataLoader
from YelpDataset import YelpDataset

MIN_FREQUENCY = 5
SPECIAL_TOKENS = ["<unk>", "<pad>"]


class SentimentNetwork(nn.Module):
    def __init__(self, _transformer):
        super().__init__()
        self.transformer = _transformer
        hidden_size = _transformer.config.hidden_size
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        output = self.transformer(x, output_attentions=True)
        hidden = output.last_hidden_state

        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction


def tokenize(data_set, _tokenizer):
    ids = _tokenizer(data_set["text"], truncation=True)["input_ids"]
    return {"ids": ids}


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train(data_loader, _model, _criterion, _optimizer, _device):
    _model.train()
    epoch_losses = []
    epoch_accs = []
    count = 0
    for batch in tqdm(data_loader, desc="training..."):
        ids = batch[0].to(_device)
        label = batch[1].to(_device)

        for i in range(len(ids)):
            count += 1
            print(count)
            prediction = _model(ids[i])
            loss = _criterion(prediction, label[i])
            accuracy = get_accuracy(prediction, label[i])
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


if __name__ == "__main__":
    chunk_size = 50
    transformer_name = "bert-base-uncased"

    training_data = YelpDataset("train_data.json", chunk_size, transformer_name)
    print(training_data.__getitem__(0))
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=1)

    test_data = YelpDataset("test_data.json", chunk_size, transformer_name)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    transformer = transformers.AutoModel.from_pretrained(transformer_name)
    model = SentimentNetwork(transformer)

    _device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    criterion.to(_device)

    n_epochs = 3
    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(
            train_dataloader, model, criterion, optimizer, _device
        )
        valid_loss, valid_acc = evaluate(test_dataloader, model, criterion, _device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "transformer.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
