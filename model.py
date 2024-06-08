from transformers import AutoModel, AutoTokenizer

# Function to create model and tokenizer
# Using BERT model and its tokenizer

def create_model():
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer