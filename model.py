import torch
import torch.nn as nn
# Hyperparameters
seq_length = 35
batch_size = 230
embed_size = 200
hidden_size = 200
num_layers = 2
learning_rate = 0.001
num_epochs = 3
device = torch.device("mps")

class LSTMTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Store hyperparameters as attributes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
