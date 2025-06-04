import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
df = pd.read_excel("chatbot_confidentiality_dataset_v2.xlsx")

# Rename column for consistency
df = df.rename(columns={"Confidentiality Level": "label"})  # Fix column name
df["label"] = df["label"].astype(int) - 1  # Ensure labels are integers

assert df["label"].min() == 0  # Smallest should be 0
assert df["label"].max() == 4  # Largest should be 4

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["Prompt"], padding="max_length", truncation=True)

# Convert to Hugging Face dataset format
train_texts, test_texts, train_labels, test_labels = train_test_split(df["Prompt"], df["label"], test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict({"Prompt": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"Prompt": test_texts, "label": test_labels})

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load BERT model

num_classes = df["label"].nunique()  # Should be 5 (0 to 4)
print(f"Number of unique classes: {num_classes}")  # Should be 5 in your case (0 to 4)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs"
)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("bert_confidentiality")
tokenizer.save_pretrained("bert_confidentiality")

print("Model training complete! ✅")
