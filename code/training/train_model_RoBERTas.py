import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
df = pd.read_excel("chatbot_confidentiality_dataset_v2.xlsx")
df = df.rename(columns={"Confidentiality Level": "label"})
df["label"] = df["label"].astype(int) - 1

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["Prompt"], padding="max_length", truncation=True)

train_texts, test_texts, train_labels, test_labels = train_test_split(df["Prompt"], df["label"], test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict({"Prompt": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"Prompt": test_texts, "label": test_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

# Training
training_args = TrainingArguments(
    output_dir="./roberta_results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./roberta_logs"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("roberta_confidentiality")
tokenizer.save_pretrained("roberta_confidentiality")
print("RoBERTa model training complete! ✅")
