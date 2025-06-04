import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load trained RoBERTa model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("roberta_confidentiality")
tokenizer = RobertaTokenizer.from_pretrained("roberta_confidentiality")
model.eval()

def classify_confidentiality(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_label

# Get user input
while True:
    print("RoBERTa Confidentiality Classifier")
    user_input = input("Enter a prompt (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    result = classify_confidentiality(user_input)
    print(f"Predicted Confidentiality Level: {result}")
