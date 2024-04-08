import collections
import numpy as np
import torch
import transformers
from torch import nn, tanh, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from YelpDataset import YelpDataset

MIN_FREQUENCY = 5
SPECIAL_TOKENS = ["<unk>", "<pad>"]


class SentimentNetwork(nn.Module):
    def __init__(self, _transformer):
        super().__init__()
        self.transformer = _transformer
        hidden_size = _transformer.config.hidden_size
        self.stars_fc = nn.Linear(hidden_size, 5)
        self.ratings_fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        output = self.transformer(x, output_attentions=True)
        hidden = output.last_hidden_state

        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        stars_prediction = self.stars_fc(tanh(cls_hidden))
        ratings_prediction = self.ratings_fc(cls_hidden)
        return stars_prediction, ratings_prediction


def tokenize(data_set, _tokenizer):
    ids = _tokenizer(data_set["text"], truncation=True)["input_ids"]
    return {"ids": ids}


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def get_regression_accuracy(predicted_values, true_values, tolerance):
    absolute_errors = torch.abs(true_values - predicted_values)

    # Determine if each prediction is within the tolerance for each target
    within_tolerance = absolute_errors <= tolerance

    # Calculate the "accuracy" for each target
    accuracies = torch.mean(within_tolerance.float(), dim=0)

    return accuracies.tolist()


def train(data_loader, _model, _classification_criterion, _regression_criterion, _optimizer, _device):
    _model.train()
    epoch_losses = []
    epoch_stars_accs = []
    epoch_ratings_accs = []
    for batch in tqdm(data_loader, desc="training..."):
        ids = batch[0].to(_device)
        star_labels = batch[1].to(_device)
        ratings_values = batch[2].to(_device)

        for i in range(len(ids)):
            stars_predictions, ratings_predictions = _model(ids[i])
            stars_loss = _classification_criterion(stars_predictions, star_labels[i] - 1)
            stars_accuracy = get_accuracy(stars_predictions, star_labels[i] - 1)

            ratings_loss = _regression_criterion(ratings_predictions, ratings_values[i].float())
            ratings_accuracy = get_regression_accuracy(ratings_predictions, ratings_values[i], 0.2)

            loss = stars_loss + ratings_loss
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            epoch_losses.append(loss.item())
            epoch_stars_accs.append(stars_accuracy.item())
            epoch_ratings_accs.append(ratings_accuracy)

    return np.mean(epoch_losses), np.mean(epoch_stars_accs), np.mean(epoch_ratings_accs)


def evaluate(data_loader, _model, _classification_criterion, _regression_criterion, _device):
    _model.eval()
    epoch_losses = []
    epoch_stars_accs = []
    epoch_ratings_accs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="evaluating..."):
            ids = batch[0].to(_device)
            star_labels = batch[1].to(_device)
            ratings_values = batch[2].to(_device)

            for i in range(len(ids)):
                stars_predictions, ratings_predictions = _model(ids[i])
                stars_loss = _classification_criterion(stars_predictions, star_labels[i] - 1)
                stars_accuracy = get_accuracy(stars_predictions, star_labels[i] - 1)

                ratings_loss = _regression_criterion(ratings_predictions, ratings_values[i])
                ratings_accuracy = get_regression_accuracy(ratings_predictions, ratings_values[i], 0.2)

                loss = stars_loss + ratings_loss
                epoch_losses.append(loss.item())
                epoch_stars_accs.append(stars_accuracy.item())
                epoch_ratings_accs.append(ratings_accuracy.item())

    return np.mean(epoch_losses), np.mean(epoch_stars_accs), np.mean(epoch_ratings_accs)


if __name__ == "__main__":
    chunk_size = 10
    transformer_name = "bert-base-uncased"

    training_data = YelpDataset("train_data.json", chunk_size, transformer_name)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=0)

    test_data = YelpDataset("test_data.json", chunk_size, transformer_name)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    transformer = transformers.AutoModel.from_pretrained(transformer_name)
    model = SentimentNetwork(transformer)

    _device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    classification_criterion = nn.CrossEntropyLoss()
    classification_criterion.to(_device)

    regression_criterion = nn.MSELoss()
    regression_criterion.to(_device)

    n_epochs = 3
    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_stars_acc, train_ratings_acc = train(train_dataloader, model, classification_criterion,
                                                               regression_criterion, optimizer, _device)
        valid_loss, valid_stars_acc, valid_ratings_acc = evaluate(test_dataloader, model, classification_criterion,
                                                                  regression_criterion, _device)
        metrics["train_losses"].append(train_loss)
        metrics["train_stars_accs"].append(train_stars_acc)
        metrics["train_ratings_accs"].append(train_ratings_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_stars_accs"].append(valid_stars_acc)
        metrics["valid_ratings_accs"].append(valid_ratings_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "transformer.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_stars_acc: {train_stars_acc:.3f}, train_ratings_acc: "
              f"{train_ratings_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_stars_acc: {valid_stars_acc:.3f}, valid_ratings_acc: "
              f"{valid_ratings_acc:.3f}")
