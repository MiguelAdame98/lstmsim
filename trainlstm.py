import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from model import LSTMTextModel

import os
import pickle


# Device configuration
device = torch.device("mps")
seq_length = 70
batch_size = 25
embed_size = 200
hidden_size = 200
num_layers = 2
learning_rate = 0.001
num_epochs = 20

# Load WikiText-2 dataset using Hugging Face
def load_hf_dataset(max_vocab_size=10000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)  # Truncate to 512
    vocab = build_vocab(dataset["train"], tokenizer, max_vocab_size)
    return dataset, vocab, tokenizer

# Build vocabulary with a size cap
def build_vocab(dataset, tokenizer, max_vocab_size):
    tokens = [tokenizer.tokenize(preprocess_text(line)) for line in dataset["text"]]
    flat_tokens = [item for sublist in tokens for item in sublist]
    token_counts = Counter(flat_tokens)
    most_common = token_counts.most_common(max_vocab_size)
    vocab = {token: idx for idx, (token, _) in enumerate(most_common)}
    vocab["<unk>"] = len(vocab)  # Add unknown token
    return vocab

# Preprocess text during tokenization
def preprocess_text(line):
    return line.lower().strip()  # Lowercase and remove extra spaces

# Dataset class for Hugging Face text data
class TextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, seq_length, max_length=500):
        self.data = self.tokenize_and_encode(data, vocab, tokenizer, max_length)
        self.seq_length = seq_length

    def tokenize_and_encode(self, data, vocab, tokenizer, max_length):
        tokens = [
            vocab.get(token, vocab["<unk>"])
            for line in data["text"]
            for token in tokenizer.tokenize(preprocess_text(line))[:max_length]  # Explicit truncation
        ]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_length],
            self.data[idx + 1:idx + self.seq_length + 1],
        )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warnings
    # Load dataset
    dataset, vocab, tokenizer = load_hf_dataset(max_vocab_size=10000)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    train_dataset = TextDataset(dataset["train"], vocab, tokenizer, seq_length)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Disable multiprocessing
    )
    print(len(train_loader))
    # Model setup
    vocab_size = len(vocab)
    model = LSTMTextModel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler(enabled=(device.type in ["cuda", "mps"]))

    # Training loop
    def train_model():
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                hidden = model.init_hidden(inputs.size(0))

                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    outputs, hidden = model(inputs, hidden)
                    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                if batch_idx % 1000 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            scheduler.step()

    train_model()

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "embed_size": embed_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "vocab_size": vocab_size
}, "lstm_model4.pth")
print("Model saved successfully!")