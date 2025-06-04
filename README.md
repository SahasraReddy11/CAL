# Confidentiality Analysis Layer (CAL)

> A real-time security layer to protect AI chatbots from prompt injection and context-aware adversarial attacks.

## 🧠 Overview

Large Language Models (LLMs) are increasingly being targeted by **prompt injection attacks**, where malicious users manipulate inputs to extract unauthorized or sensitive information. Traditional keyword-based or rule-based filters are insufficient to detect such nuanced, context-aware threats.

This project introduces the **Confidentiality Analysis Layer (CAL)** — a fine-tuned BERT-based classifier that acts as a safeguard by analyzing and scoring chatbot responses based on their confidentiality. CAL operates in real time and helps ensure regulatory compliance and privacy protection for AI deployments.

## 🔐 Key Features

- **Real-Time Confidentiality Scoring**: Scores chatbot responses from 1 (least sensitive) to 5 (most sensitive).
- **BERT-Based Classifier**: Fine-tuned using a micro-labeled dataset aligned with the NIST 800-53 framework.
- **Model Comparisons**: Benchmarked against LSTM, CNN, and RoBERTa using precision, recall, F1-score, and accuracy.
- **High Accuracy**: BERT achieves >95% accuracy in detecting highly sensitive responses.
- **Low Latency**: Unlike RoBERTa, CAL with BERT provides high performance without compromising speed.
- **Regulation-Ready**: Supports GDPR and HIPAA compliance for enterprise-level deployment.

The codes present in the folder named "training" are each used to train a model with the given dataset. Please change the name of the dataset accordingly.

The codes present in the folder named "interference" will help use the generated models  to test out prompts of your own.
