import torch
from transformers import BertTokenizer
import torch.nn as nn

# Define CNN model (same as used during training)
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv = nn.Conv1d(128, 100, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TextCNN(vocab_size=tokenizer.vocab_size, num_classes=5)
model.load_state_dict(torch.load("cnn_model.pt"))  # Load your saved weights
model.eval()

def classify_confidentiality(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        logits = model(input_ids)
        predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label

while True:
    print("CNN Confidentiality Classifier")
    user_input = input("Enter a prompt (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    result = classify_confidentiality(user_input)
    print(f"Predicted Confidentiality Level: {result}")
