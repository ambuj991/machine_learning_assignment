import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sample_data import get_sample_data
import model


# Define a function to get sentence embeddings
# •	Takes sentences, model, and tokenizer as input.
# •	Tokenizes sentences using the tokenizer (handles splitting, padding, truncation).
# •	Gets model outputs, focusing on the last hidden state from the final encoder layer.
# •	Performs mean pooling across tokens to create a single embedding vector per sentence.

def get_sentence_embeddings(sentences, model, tokenizer):
    
    # Tokenize the input sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        # Get model outputs
        outputs = model(**inputs)
    # Mean pooling of the token embeddings
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings


# Get sample data
sentences = get_sample_data()[0]

# Create model and tokenizer
model, tokenizer = model.create_model()

# Encode the sentences into embeddings
embeddings = get_sentence_embeddings(sentences, model, tokenizer)

# Print the obtained embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding.numpy()[:5]}\n")