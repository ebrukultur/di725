# Sentiment Classification with GPT2 & NanoGPT
This project explores the effectiveness of transformer-based models for multi-class sentiment analysis on customer service conversations. We compare fine-tuning a pre-trained GPT-2 model with training a custom NanoGPT-style model from scratch. Evaluation metrics, model performance, and training behavior are visualized using Weights & Biases.

## Highlights
    - Fine-tuned GPT-2 with a classification head
    
    - Custom NanoGPT-style model implemented from scratch
    
    - Evaluation metrics: Accuracy, F1-score, Confusion Matrix
    
    - Tokenization via tiktoken (GPT-2 BPE)
    
    - Support for mixed-precision training (AMP)
    
    - Training & evaluation logging with Weights & Biases
