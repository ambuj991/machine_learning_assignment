


def get_sample_data():
    
    sentences = [
        "Sentence transformers are useful for various NLP tasks.",
        "They provide fixed-length embeddings for input sentences.",
        "This is a sample sentence to test the model.",
        "Transformers have revolutionized the field of natural language processing.",
        "The ability to capture long-term dependencies is a significant advantage.",
        "Bidirectional models provide context from both directions.",
        "Pre-trained models can be fine-tuned for specific tasks.",
        "Transfer learning has become a standard approach in NLP.",
        "This sentence is intended to test sentiment analysis.",
        "Another sentence for testing classification tasks.",
        "Machine learning models require a lot of data for training.",
        "The more data we have, the better the model performs.",
        "Hyperparameter tuning is an essential step in model optimization.",
        "Cross-validation helps in assessing the performance of the model.",
        "Model evaluation metrics include accuracy, precision, and recall.",
        "I love using transformers for my projects.",  # Positive sentiment
        "This model is incredibly efficient and accurate.",  # Positive sentiment
        "I had a terrible experience with the previous version of this model.",  # Negative sentiment
        "The results were disappointing and below expectations.",  # Negative sentiment
        "I'm very satisfied with the performance of this new update."  # Positive sentiment
    ]

    labels_task_a = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    labels_task_b = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]

    return sentences, labels_task_a, labels_task_b
