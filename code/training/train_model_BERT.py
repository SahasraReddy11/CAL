from google.colab import drive
drive.mount('/content/drive')

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load and clean dataset
df = pd.read_excel("/content/drive/MyDrive/CAL/dataset.xlsx")
df = df.rename(columns={"Confidentiality Level": "label"})
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int) - 1
df = df[df["label"].between(0, 4)]
df = df.reset_index(drop=True)
print("Unique cleaned labels:", df["label"].unique())

assert df["label"].min() == 0
assert df["label"].max() == 4

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["Prompt"], padding="max_length", truncation=True)

# Dataset split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Prompt"], df["label"], test_size=0.2, random_state=42
)
train_dataset = Dataset.from_dict({"Prompt": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"Prompt": test_texts, "label": test_labels})

# Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load model
num_classes = df["label"].nunique()
print(f"Number of unique classes: {num_classes}")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Training args
print(TrainingArguments.__module__)
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/CAL/results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="/content/drive/MyDrive/CAL/logs"
)

# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train and save
trainer.train()
model.save_pretrained("/content/drive/MyDrive/CAL/bert_confidentiality")
tokenizer.save_pretrained("/content/drive/MyDrive/CAL/bert_confidentiality")

print("Model training complete! ✅")
