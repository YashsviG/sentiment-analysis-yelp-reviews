import argparse
import os
import sys
import numpy as np
import sklearn.metrics
import torch
import transformers
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from SentimentNetwork import SentimentNetwork
from SVM import funny_cool_useful, stars, use_train_model
from YelpDataset import YelpDataset

import NaiveBayes


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


def train(
        data_loader,
        _model,
        _classification_criterion,
        _regression_criterion,
        _optimizer,
        _device,
        _is_classification,
        ratings_index,
):
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
                loss = _regression_criterion(
                    predictions, labels[i][:, ratings_index].unsqueeze(1).float()
                )

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            epoch_losses.append(loss.item())

    return np.mean(epoch_losses)


def evaluate(
        data_loader,
        _model,
        _classification_criterion,
        _regression_criterion,
        _device,
        _is_classification,
        ratings_index,
        _task,
):
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
                    loss = _regression_criterion(
                        predictions, labels[i][:, ratings_index].unsqueeze(1).float()
                    )

                epoch_losses.append(loss.item())

                predicted.append(predictions.cpu().detach().argmax(dim=-1).numpy())

                if _is_classification:
                    actual.append(labels.cpu().detach().numpy())
                else:
                    actual.append(
                        np.array(
                            labels[i][:, ratings_index]
                            .unsqueeze(1)
                            .float()
                            .cpu()
                            .detach()
                        )
                    )

    if _is_classification:
        print(
            sklearn.metrics.classification_report(
                np.concatenate(predicted).ravel().tolist(),
                np.concatenate(actual).ravel().tolist(),
                target_names=["1", "2", "3", "4", "5"],
            )
        )
    else:
        print(
            f"Root Mean Squared Error Score for {_task}: {sklearn.metrics.root_mean_squared_error(np.concatenate(predicted).ravel().tolist(), np.concatenate(actual).ravel().tolist())}"
        )

    return np.mean(epoch_losses)


def run_neural_network(args):
    chunk_size = 400
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if not args.trained_model:
        training_data = YelpDataset(args.training_file, chunk_size, tokenizer_name)
        train_dataloader = DataLoader(
            training_data, batch_size=1, shuffle=True, num_workers=3
        )

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

    model = SentimentNetwork(
        transformer.config.vocab_size,
        transformer.config.hidden_size,
        tokenizer.pad_token_id,
        classification,
        100,
        [3, 5, 7],
        0.25,
    )
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
            train_loss = train(
                train_dataloader,
                model,
                classification_criterion,
                regression_criterion,
                optimizer,
                _device,
                classification,
                _ratings_index,
            )

            valid_loss = evaluate(
                test_dataloader,
                model,
                classification_criterion,
                regression_criterion,
                _device,
                classification,
                _ratings_index,
                args.task,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if not args.trained_model:
                    torch.save(model.state_dict(), f"{args.task}_{best_valid_loss}.pt")
            print(f"epoch: {epoch}")
            print(f"train_loss: {train_loss:.3f}")
            print(f"valid_loss: {valid_loss:.3f}")
    else:
        valid_loss = evaluate(
            test_dataloader,
            model,
            classification_criterion,
            regression_criterion,
            _device,
            classification,
            _ratings_index,
            args.task,
        )
        print(f"valid_loss: {valid_loss:.3f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        # MEASURING TOOLS
        #            |Yashar Nesvaderani - A00984009
        #           0|       10|       20|       30|       40||      50|       60|       70|       80|
        description=("===================Sentiment Analysis For COMP8085 Project 2====================\n" +
                     "                                   Built By:                                    \n" +
                     "                          Yashsvi Girdhar - A01084035\n" +
                     "                             Aaron Moen - A01313456\n" +
                     "                         Yashar Nesvaderani - A00984009\n" +
                     "================================================================================\n" +
                     " Master Program for running all three sentiment analysis models in the project. \n"
                     ), formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--model", type=str, help="Model to use for Sentiment Analysis: nn, svm, nb", required=True)
    parser.add_argument("--test_file", type=str, help=".jsonl file to to test with. Required.", required=True)
    parser.add_argument("--training_file", type=str, help=".jsonl file to to train with. Must match model.")
    parser.add_argument("--trained_model", type=str, help="Trained pickled model file to load.")
    parser.add_argument("--pickle", action='store_true', help="Flag to export trained model as pickle file.")
    parser.add_argument("--out_file", type=str, help="Custom location to export pickle file to.",
                        default="pickled_models/new_model.pkl")
    parser.add_argument("--task", type=str, help="stars, useful, funny, cool, used for SVM.", default='stars')
    parser.add_argument("--experiment1", action="store_true", help="Runs with experiment 1, if svm selected.")
    parser.add_argument("--experiment2", action="store_true", help="Runs with experiment 2, if svm selected.")
    myArgs = parser.parse_args()

    # validate model
    if myArgs.model not in ("nn", "svm", "nb"):
        print('Model selection must be either "nn", "svm" or "nb"')
        sys.exit(0)

    # validate test_file
    if not os.path.isfile(myArgs.test_file):
        print('Unable to find training data file at ' + myArgs.test_file)
        print('Use -h for help')
        sys.exit(0)

    # validate training choice
    if not myArgs.trained_model and not myArgs.training_file:
        print('Either a trained model file or a training data file must be provided.')
        print('Use -h for help')
        sys.exit(0)

    # validate trained_model
    if myArgs.trained_model and not os.path.isfile(myArgs.trained_model):
        print('Unable to find trained model file at ' + myArgs.trained_model)
        print('Use -h for help')
        sys.exit(0)

    # validate training_file
    if myArgs.training_file and not os.path.isfile(myArgs.training_file):
        print('Unable to find training data file at ' + myArgs.training_file)
        sys.exit(0)

    # validate task
    if myArgs.task not in ("stars", "useful", "funny", "cool"):
        print('Task value must be either "stars", "useful", "funny", or "cool"')
        print('Use -h for help')
        sys.exit(0)

    # Validate impossible pickle
    if myArgs.pickle and not myArgs.trained_model:
        print('Cannot Pickle a pre-trained model.')
        print('Running tests with pre-trained model anyways.')
        myArgs.pickle = False

    return myArgs


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    if args.model == "nn":
        run_neural_network(args)

    elif args.model == "nb":
        NaiveBayes.NaiveBayes(args, True)

    elif args.model == "svm":
        if args.trained_model:
            use_train_model(args.test_file, args.task, args.trained_model)
        else:
            if args.task == "stars":
                stars(
                    experiment1=args.experiment1,
                    experiment2=args.experiment2,
                    training_file=args.training_file,
                    test_file=args.test_file,
                )
            else:
                funny_cool_useful(
                    experiment1=args.experiment1,
                    training_file=args.training_file,
                    test_file=args.test_file,
                )
