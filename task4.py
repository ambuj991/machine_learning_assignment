import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sample_data import get_sample_data
from model import create_model  

#  Multi-Task Learning Model
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_classes_task_a, num_classes_task_b):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        
        # Classifier for task A
        self.classifier_task_a = nn.Linear(base_model.config.hidden_size, num_classes_task_a)
        # Classifier for task B
        self.classifier_task_b = nn.Linear(base_model.config.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        # Get the hidden states from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Apply mean pooling to get a single vector per input
        pooled_output = torch.mean(outputs, dim=1)
        
        # Compute the softmax scores for task A
        task_a_output = F.softmax(self.classifier_task_a(pooled_output), dim=-1)
        # Compute the softmax scores for task B
        task_b_output = F.softmax(self.classifier_task_b(pooled_output), dim=-1)
        return {'classifier_task_a': task_a_output, 'classifier_task_b': task_b_output}

# Load sample data and create base model
sentences, labels_task_a, labels_task_b = get_sample_data()
base_model, tokenizer = create_model()

# The number of labels for each task 
# Task A :[0, 1, 2] -> 3                   [sentence classification]
# Task B :[0, 1]    -> 2                   [sentiment Analysis]
num_classes_task_a, num_classes_task_b = 3, 2

# Instantiate the multi-task model
multi_task_model = MultiTaskModel(base_model, num_classes_task_a, num_classes_task_b)

# Tokenize the input sentences
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Define loss functions for both tasks
loss_fn_task_a = nn.CrossEntropyLoss()
loss_fn_task_b = nn.CrossEntropyLoss()

# Implement layer-wise learning rates
param_optimizer = list(multi_task_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
base_model_layer_11 = "base_model.encoder.layer.11"

# Create a set to keep track of parameters already added
added_params = set()
optimizer_grouped_parameters = []

# Function to add parameter groups without duplication
def add_param_group(params, lr, weight_decay):
    global added_params
    params = [(n, p) for n, p in params if id(p) not in added_params]
    added_params.update(id(p) for _, p in params)
    if params:
        optimizer_grouped_parameters.append({'params': [p for _, p in params], 'lr': lr, 'weight_decay': weight_decay})

# Group parameters with specific learning rates and weight decay

# Below code configures distinct learning rates and weight decay values for different groups of model parameters, 
# for layer-wise learning rates. It allows various sections of the model to 
# learn at different rates hence increasing training efficiency. Lower learning rates are used to fine-tune the 
# parameters of the underlying model, whereas higher learning rates are utilized for newly added layers and task-specific 
# classifiers to learn faster. Weight decay, which helps prevent overfitting, is used judiciously, eliminating 
# specific parameters like as biases. This careful adjustment keeps the model stable while it learns task-specific properties.

add_param_group(((n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay) and base_model_layer_11 not in n), lr=1e-5, weight_decay=0.01)
add_param_group(((n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay) and base_model_layer_11 not in n), lr=1e-5, weight_decay=0.0)
add_param_group(((n, p) for n, p in param_optimizer if base_model_layer_11 in n and not any(nd in n for nd in no_decay)), lr=1e-4, weight_decay=0.01)
add_param_group(((n, p) for n, p in param_optimizer if base_model_layer_11 in n and any(nd in n for nd in no_decay)), lr=1e-4, weight_decay=0.0)
add_param_group(multi_task_model.classifier_task_a.named_parameters(), lr=1e-4, weight_decay=0.0)
add_param_group(multi_task_model.classifier_task_b.named_parameters(), lr=1e-4, weight_decay=0.0)

# Create the optimizer
optimizer = optim.AdamW(optimizer_grouped_parameters)

# Prepare inputs and labels for training
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Convert labels to PyTorch tensors
labels_task_a = torch.tensor(labels_task_a)
labels_task_b = torch.tensor(labels_task_b)

# Dummy training step
# Note: In practice, you'd train on a larger dataset with multiple epochs

for epoch in range(1):
    optimizer.zero_grad()  # Reset gradients
    outputs = multi_task_model(input_ids, attention_mask)  # Forward pass
    
    # Compute loss for task A
    loss_task_a = loss_fn_task_a(outputs['classifier_task_a'], labels_task_a) 
    # Compute loss for task B
    loss_task_b = loss_fn_task_b(outputs['classifier_task_b'], labels_task_b)
    
    # Sum the losses
    total_loss = loss_task_a + loss_task_b
    total_loss.backward()  # Backward pass
    optimizer.step()  # Update model parameters

# Test the model
outputs = multi_task_model(input_ids, attention_mask)

# Access outputs for each task (assuming labels are class indices)
predicted_label_a = torch.argmax(outputs['classifier_task_a'], dim=1)
predicted_label_b = torch.argmax(outputs['classifier_task_b'], dim=1)

# Print the obtained outputs for both tasks
print("Task A Outputs (Sentence Classification):")
print(outputs['classifier_task_a'])
print("\nTask B Outputs (Sentiment Analysis):")
print(outputs['classifier_task_b'])

predicted_label_a = torch.argmax(outputs['classifier_task_a'], dim=1)
predicted_label_b = torch.argmax(outputs['classifier_task_b'], dim=1)
# Print the  labels for both tasks
print("Task A labels (Sentence Classification):")
print(predicted_label_a)
print("\nTask B labels (Sentiment Analysis):")
print(predicted_label_b)