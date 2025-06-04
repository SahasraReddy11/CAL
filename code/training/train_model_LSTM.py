import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Dataset
df = pd.read_excel("chatbot_confidentiality_dataset_v2.xlsx")
df = df.rename(columns={"Confidentiality Level": "label"})
df["label"] = df["label"].astype(int) - 1

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(labels.tolist())

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)

train_texts, test_texts, train_labels, test_labels = train_test_split(df["Prompt"], df["label"], test_size=0.2)
train_data = TextDataset(train_texts, train_labels)
test_data = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)

# LSTM Model
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

model = TextLSTM(vocab_size=tokenizer.vocab_size, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(3):
    model.train()
    for batch, labels in train_loader:
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids, batch["attention_mask"].to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
print("LSTM model training complete! ✅")
