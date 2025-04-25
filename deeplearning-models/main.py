import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from shutil import copyfile
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from vocab import Vocab
from dataset import ViSMEC, collate_fn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scorers = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score
}

def train(epoch: int, model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    running_loss = 0.0
    with tqdm(desc=f'Epoch {epoch} - Training', unit='it', total=len(dataloader)) as pbar:
        for it, items in enumerate(dataloader):
            input_ids = items["input_ids"].to(device)
            labels = items["labels"].to(device)

            _, loss = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

def compute_scores(predictions: list, labels: list) -> dict:
    labels = np.array(labels)
    predictions = np.array(predictions)
    valid_indices = (labels >= 0) & (predictions >= 0)
    labels = labels[valid_indices]
    predictions = predictions[valid_indices]

    scores = {}
    for scorer_name, scorer in scorers.items():
        scores[scorer_name] = scorer(labels, predictions, average="macro")
    return scores

def evaluate_metrics(epoch: int, model: nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    all_labels = []
    all_predictions = []
    with tqdm(desc=f'Epoch {epoch} - Evaluating', unit='it', total=len(dataloader)) as pbar:
        for items in dataloader:
            input_ids = items["input_ids"].to(device)
            labels = items["labels"].to(device)
            with torch.no_grad():
                logits, _ = model(input_ids)

            predictions = logits.argmax(dim=-1).long()
            mask = labels != -1
            filtered_labels = labels[mask]
            filtered_predictions = predictions[mask]

            all_labels.extend(filtered_labels.cpu().numpy())
            all_predictions.extend(filtered_predictions.cpu().numpy())
            pbar.update()

    return compute_scores(all_predictions, all_labels)

def save_checkpoint(state: dict, checkpoint_path: str):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_path, "last_model.pth"))

def main(args):
    vocab = Vocab(args.train_path, args.dev_path, args.test_path)

    train_dataset = ViSMEC(args.train_path, vocab)
    dev_dataset = ViSMEC(args.dev_path, vocab)
    test_dataset = ViSMEC(args.test_path, vocab)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    epoch = 0
    best_score = 0
    patience = 0

    if args.model == "Luong":
        from LSTM_Luong_attn import LSTMModelLuong
        model = LSTMModelLuong(
            input_dim=args.input_dim,
            layer_dim=args.layer_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=len(vocab.stoi),
            dropout=args.dropout,
            output_dim=args.output_dim,
        ).to(device)
    elif args.model == "Bahdanau":
        from LSTM_Bahdanau_attn import LSTMModelBahdanau
        model = LSTMModelBahdanau(
            input_dim=args.input_dim,
            layer_dim=args.layer_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=len(vocab.stoi),
            dropout=args.dropout,
            output_dim=args.output_dim,
        ).to(device)
    elif args.model == "LSTM":
        from LSTM import LSTMModel
        model = LSTMModel(
            input_dim=args.input_dim,
            layer_dim=args.layer_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=len(vocab.stoi),
            dropout=args.dropout,
            output_dim=args.output_dim,
        ).to(device)
    elif args.model == "GRU":
        from gru_model import GRUModel
        model = GRUModel(
            input_dim=args.input_dim,
            layer_dim=args.layer_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=len(vocab.stoi),
            dropout=args.dropout,
            output_dim=args.output_dim,
        ).to(device)
    elif args.model == "RNN":
        from rnn_model import RNNModel
        model = RNNModel(
            input_dim=args.input_dim,
            layer_dim=args.layer_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=len(vocab.stoi),
            dropout=args.dropout,
            output_dim=args.output_dim,
        ).to(device)
    else :
        raise ValueError(f"Unsupported model type: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))

    with open(args.report_file, "w") as report_file:
        while True:
            train(epoch, model, train_dataloader, optimizer)
            scores = evaluate_metrics(epoch, model, dev_dataloader)
            print(f"Dev scores: {scores}")
            report_file.write(f"Epoch {epoch} - Dev scores: {scores}\n")

            score = scores["f1"]
            is_the_best_model = score > best_score

            if is_the_best_model:
                best_score = score
                patience = 0
            else:
                patience += 1

            save_checkpoint({
                "epoch": epoch,
                "best_score": best_score,
                "patience": patience,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, args.checkpoint_path)

            if is_the_best_model:
                copyfile(
                    os.path.join(args.checkpoint_path, "last_model.pth"),
                    os.path.join(args.checkpoint_path, "best_model.pth")
                )

            if patience == args.allowed_patience:
                break

            epoch += 1
        # Load the best model for testing
        best_model_path = os.path.join(args.checkpoint_path, "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path)["state_dict"])
        else:
            print("Best model not found. Using the last model for testing.")
        test_scores = evaluate_metrics(epoch, model, test_dataloader)
        print(f"Test scores: {test_scores}")
        report_file.write(f"Test scores: {test_scores}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence labeling model.")
    parser.add_argument("--input_dim", type=int, default=256, help="Input dimension size.")
    parser.add_argument("--layer_dim", type=int, default=3, help="Number of LSTM layers.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size.")
    parser.add_argument("--output_dim", type=int, default=7, help="Output dimension size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--allowed_patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--train_path", type=str, default='UIT-VSMEC/train.json', help="Path to training dataset.")
    parser.add_argument("--dev_path", type=str, default='UIT-VSMEC/valid.json', help="Path to validation dataset.")
    parser.add_argument("--test_path", type=str, default='UIT-VSMEC/test.json', help="Path to testing dataset.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save checkpoints.")
    parser.add_argument("--report_file", type=str, default="report.txt", help="Name of the report file.")
    parser.add_argument("--model", type=str, choices=["Luong", "Bahdanau","LSTM","GRU","RNN"], required=True, help="Model type to use.")

    args = parser.parse_args()
    main(args)
