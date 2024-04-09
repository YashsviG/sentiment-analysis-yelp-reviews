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
    def __init__(self, vocab_size, embedding_dim, is_classification):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=128)

        self.is_classification = is_classification
        if is_classification:
            self.fc = nn.Linear(128, 5)
        else:
            self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, (_, _) = self.lstm(x)
        x = x[:, -1, :]

        if self.is_classification:
            x = tanh(x)

        prediction = self.fc(x)
        return prediction


def tokenize(data_set, _tokenizer):
    ids = _tokenizer(data_set["text"], truncation=True)["input_ids"]
    return {"ids": ids}


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    print(f"\nStars Accuracy: {accuracy:.2f}")
    return accuracy


def get_regression_accuracy(predicted_values, true_values, tolerance):
    absolute_errors = torch.abs(true_values - predicted_values)

    # Determine if each prediction is within the tolerance for each target
    within_tolerance = absolute_errors <= tolerance

    # Calculate the "accuracy" for each target
    accuracies = torch.mean(within_tolerance.float(), dim=0)
    print(f"Ratings Accuracy: {accuracies.tolist()}")

    return accuracies.tolist()


def train(data_loader, _model, _classification_criterion, _regression_criterion, _optimizer, _device):
    _model.train()
    epoch_losses = []
    epoch_stars_accs = []
    epoch_ratings_accs = []
    for batch in tqdm(data_loader, desc="training..."):
        ids = batch[0].to(_device)
        star_labels = batch[1].to(_device)
        # ratings_values = batch[2].to(_device)

        for i in range(len(ids)):
            predictions = _model(ids[i])
            loss = _classification_criterion(predictions, star_labels[i] - 1)
            accuracy = get_accuracy(predictions, star_labels[i] - 1)

            # ratings_loss = _regression_criterion(ratings_predictions, ratings_values[i].float())
            # ratings_accuracy = get_regression_accuracy(ratings_predictions, ratings_values[i], 0.2)

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            epoch_losses.append(loss.item())
            epoch_stars_accs.append(accuracy.item())

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
                epoch_stars_accs.append(stars_accuracy.item())

                for rating in ratings_accuracy:
                    epoch_ratings_accs.append(rating.item())

    return np.mean(epoch_losses), np.mean(epoch_stars_accs), np.mean(epoch_ratings_accs)


if __name__ == "__main__":
    chunk_size = 350
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_name = 'distilbert/distilbert-base-uncased'
    print(torch.zeros(1).cuda())

    training_data = YelpDataset("train_data.json", chunk_size, transformer_name)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=3)

    test_data = YelpDataset("test_data.json", chunk_size, transformer_name)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=3)

    transformer = transformers.AutoModel.from_pretrained(transformer_name)
    model = SentimentNetwork(30522, 768, True)
    model.to(_device)
    print(torch.cuda.memory_summary())

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    classification_criterion = nn.CrossEntropyLoss()
    classification_criterion.to(_device)
    print(torch.cuda.memory_summary())

    regression_criterion = nn.MSELoss()
    regression_criterion.to(_device)

    n_epochs = 1
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
