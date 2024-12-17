import torch
from transformers import AutoTokenizer
from model import LSTMTextModel  # Import the model class
import pickle
import os

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
SEQ_LENGTH = 50
MAX_GEN_LENGTH = 500

def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1).item()

# Preprocess text during tokenization
def preprocess_text(line):
    """Lowercase and strip text."""
    return line.lower().strip()

# Text generation function
def generate_text(model, start_text, vocab, tokenizer, seq_length, max_len=500):
    """Generate text using the trained LSTM model."""
    model.eval()  # Set the model to evaluation mode
    vocab_reverse = {idx: token for token, idx in vocab.items()}  # Reverse the vocab mapping
    tokens = [vocab.get(token, vocab["<unk>"]) for token in tokenizer.tokenize(preprocess_text(start_text))]
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    hidden = model.init_hidden(1)  # Initialize hidden state for batch size 1
    generated_text = start_text
    with torch.no_grad():  # Disable gradient computation
        for _ in range(max_len):
            output, hidden = model(tokens[:, -seq_length:], hidden)
            if "<unk>" in vocab:
                output[:, -1, vocab["<unk>"]] = float('-inf')
            next_token = sample_with_temperature(output[:, -1, :].squeeze(), temperature=0.8)
            next_word = vocab_reverse.get(next_token, "<unk>")
            if next_word == "<unk>":  # Stop if <unk> token is generated repeatedly
                break
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
            generated_text += " " + next_word
    return generated_text
def generate_text2(model, start_text, vocab, tokenizer, seq_length, max_len=50, temperature=1.0):
    """Generate text using the trained LSTM model."""
    model.eval()
    vocab_reverse = {idx: token for token, idx in vocab.items()}  # Reverse vocab mapping
    tokens = [vocab.get(token, vocab["<unk>"]) for token in tokenizer.tokenize(preprocess_text(start_text))]
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    hidden = model.init_hidden(1)
    generated_text = start_text
    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(tokens[:, -seq_length:], hidden)
            
            # Sample next token with temperature scaling
            logits = output[:, -1, :].squeeze()
            logits = logits / temperature  # Apply temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            next_word = vocab_reverse.get(next_token, "<unk>")
            if next_word == "<unk>":
                continue  # Skip unknown tokens
            
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
            generated_text += " " + next_word
    return generated_text


# Main function for testing
def main():
    """Main testing loop."""
    # Ensure necessary files exist
    if not os.path.exists("lstm_model2.pth"):
        raise FileNotFoundError("Model file 'lstm_model.pth' not found. Train the model first.")
    if not os.path.exists("vocab.pkl"):
        raise FileNotFoundError("Vocabulary file 'vocab.pkl' not found. Save the vocab during training.")

    # Load model and vocabulary
    checkpoint = torch.load("lstm_model3.pth", map_location=device)
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Initialize the model
    model = LSTMTextModel(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Test text generation
    start_texts = [
        "Natural language",
        "Machine learning",
        "Deep learning",
        "Artificial intelligence",
        "I love my",
        "you are the",
        "Once upon a time",
        "In a world of",
        "The future of AI",
        "The best way to"
    ]

    for start_text in start_texts:
        print(f"Start Text: {start_text}")
        generated_text = generate_text2(model, start_text, vocab, tokenizer, SEQ_LENGTH, MAX_GEN_LENGTH)
        print(f"Generated Text: {generated_text}")
        print("-" * 50)

# Run the script
if __name__ == "__main__":
    main()
