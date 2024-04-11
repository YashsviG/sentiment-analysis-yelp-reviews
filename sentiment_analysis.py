import argparse
import numpy as np
import sklearn.metrics
import torch
import transformers
from transformers import AutoTokenizer
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from SentimentNetwork import SentimentNetwork
from YelpDataset import YelpDataset

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


def train(data_loader, _model, _classification_criterion, _regression_criterion, _optimizer, _device, _is_classification, ratings_index):
    _model.train()
    epoch_losses = []
    for batch in tqdm(data_loader, desc="training..."):
        ids = batch[0].to(_device)

        if _is_classification:
            labels = batch[1].to(_device)
        else:
            labels = batch[2].to(_device)

        for i in range(len(ids)):
            predictions = _model(ids[i])

            if _is_classification:
                loss = _classification_criterion(predictions, labels[i] - 1)
            else:
                loss = _regression_criterion(predictions, labels[i][:, ratings_index].unsqueeze(1).float())

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            epoch_losses.append(loss.item())

    return np.mean(epoch_losses)


def evaluate(data_loader, _model, _classification_criterion, _regression_criterion, _device, _is_classification, ratings_index, _task):
    _model.eval()
    epoch_losses = []
    predicted = []
    actual = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="evaluating..."):
            ids = batch[0].to(_device)
            if _is_classification:
                labels = batch[1].to(_device)
            else:
                labels = batch[2].to(_device)

            for i in range(len(ids)):
                predictions = _model(ids[i])

                if _is_classification:
                    labels = labels[i] - 1
                    loss = _classification_criterion(predictions, labels)
                else:
                    loss = _regression_criterion(predictions, labels[i][:, ratings_index].unsqueeze(1).float())

                epoch_losses.append(loss.item())
                
                predicted.append(predictions.cpu().detach().argmax(dim=-1).numpy())

                if _is_classification:
                    actual.append(labels.cpu().detach().numpy())
                else:
                    actual.append(np.array(labels[i][:, ratings_index].unsqueeze(1).float().cpu().detach()))

    if _is_classification:
        print(sklearn.metrics.classification_report(np.concatenate(predicted).ravel().tolist(), np.concatenate(actual).ravel().tolist(), target_names=["1", "2", "3", "4", "5"]))
    else:
        print(f"Root Mean Squared Error Score for {_task}: {sklearn.metrics.root_mean_squared_error(np.concatenate(predicted).ravel().tolist(), np.concatenate(actual).ravel().tolist())}")

    return np.mean(epoch_losses)



def run_neural_network(args):
    chunk_size = 400
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_name = 'google-bert/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if not args.trained_model:
        training_data = YelpDataset(args.training_file, chunk_size, tokenizer_name)
        train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=3)

    test_data = YelpDataset(args.test_file, chunk_size, tokenizer_name)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=3)

    transformer = transformers.AutoModel.from_pretrained(tokenizer_name)

    if args.task == "stars":
        classification = True
    else:
        classification = False

    _ratings_index = 0
    if args.task == "useful":
        _ratings_index = 0
    elif args.task == "funny":
        _ratings_index = 1
    elif args.task == "cool":
        _ratings_index = 2    


    model = SentimentNetwork(transformer.config.vocab_size, transformer.config.hidden_size, tokenizer.pad_token_id, classification, 100, [3, 5, 7], 0.25)
    if args.trained_model:
        model.load_state_dict(torch.load(args.trained_model))
         
    model.to(_device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    classification_criterion = nn.CrossEntropyLoss(ignore_index=-2)
    classification_criterion.to(_device)

    regression_criterion = nn.MSELoss()
    regression_criterion.to(_device)

    n_epochs = 1
    best_valid_loss = float("inf")

    if not args.trained_model:
        for epoch in range(n_epochs):
                train_loss = train(train_dataloader, model, classification_criterion,
                                                                regression_criterion, optimizer, _device, classification, _ratings_index)
                

                valid_loss = evaluate(test_dataloader, model, classification_criterion,
                                                                        regression_criterion, _device, classification, _ratings_index, args.task)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if not args.trained_model:
                        torch.save(model.state_dict(), f"{args.task}_{best_valid_loss}.pt")
                print(f"epoch: {epoch}")
                print(f"train_loss: {train_loss:.3f}")
                print(f"valid_loss: {valid_loss:.3f}")
    else:
        valid_loss = evaluate(test_dataloader, model, classification_criterion,
                                                                    regression_criterion, _device, classification, _ratings_index, args.task)
        print(f"valid_loss: {valid_loss:.3f}")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network")
    parser.add_argument("--task", type=str, help="stars, useful, funny, cool", required=True)
    parser.add_argument("--trained_model", type=str, help="Trained .pt model file.")
    parser.add_argument("--training_file", type=str, help=".jsonl file to to train with.")
    parser.add_argument("--test_file", type=str, help=".jsonl file to to test with.", required=True)
    parser.add_argument("--model", type=str, help="Model to use for Sentiment Analysis: nn, tbd, tbd", required=True)
    args = parser.parse_args()

    if args.model == "nn":
        run_neural_network(args)