import torch
from transformers import BertTokenizer
import torch.nn as nn

# Define LSTM model (same as used during training)
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TextLSTM(vocab_size=tokenizer.vocab_size, num_classes=5)
model.load_state_dict(torch.load("lstm_model.pt"))  # Load your saved weights
model.eval()

def classify_confidentiality(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        logits = model(input_ids)
        predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label

while True:
    print("LSTM Confidentiality Classifier")
    user_input = input("Enter a prompt (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    result = classify_confidentiality(user_input)
    print(f"Predicted Confidentiality Level: {result}")
