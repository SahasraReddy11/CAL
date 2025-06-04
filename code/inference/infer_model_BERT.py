import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_confidentiality")
tokenizer = BertTokenizer.from_pretrained("bert_confidentiality")
model.eval()  # Set model to evaluation mode

# Function to classify prompt
def classify_confidentiality(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_label

# Get user input
while True:
    print("Hello")
    user_input = input("Enter a prompt (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    result = classify_confidentiality(user_input)
    print(f"Predicted Confidentiality Level: {result}")
