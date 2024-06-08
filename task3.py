import torch
import torch.nn as nn
import torch.optim as optim
from sample_data import get_sample_data
from model import create_model
import torch.nn.functional as F

# Defining Multi-Task Learning Model
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_classes_task_a, num_classes_task_b):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        
        # Define a classifier for Task A with output size equal to number of classes in Task A
        self.classifier_task_a = nn.Linear(base_model.config.hidden_size, num_classes_task_a)
        # Define a classifier for Task B with output size equal to number of classes in Task B
        self.classifier_task_b = nn.Linear(base_model.config.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        # Get the hidden states from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Apply mean pooling on the hidden states to get a single vector per input sequence
        pooled_output = torch.mean(outputs, dim=1)
        # Apply the classifier for Task A and use softmax to get probabilities
        task_a_output = F.softmax(self.classifier_task_a(pooled_output), dim=-1)
        # Apply the classifier for Task B and use softmax to get probabilities
        task_b_output = F.softmax(self.classifier_task_b(pooled_output), dim=-1)
        
        # Return the outputs for both tasks
        return {'classifier_task_a': task_a_output, 'classifier_task_b': task_b_output}

def freeze_layers(model, freeze_base=True, freeze_task_a=False, freeze_task_b=False):
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False
    if freeze_task_a:
        for param in model.classifier_task_a.parameters():
            param.requires_grad = False
    if freeze_task_b:
        for param in model.classifier_task_b.parameters():
            param.requires_grad = False

# Load sample data and the pre-trained model
sentences, labels_task_a, labels_task_b = get_sample_data()
base_model, tokenizer = create_model()

# Define the number of classes for each task
num_classes_task_a, num_classes_task_b = 3, 2

# Instantiate the multi-task model using the base model and the number of classes for each task
multi_task_model = MultiTaskModel(base_model, num_classes_task_a, num_classes_task_b)

# Check if a GPU is available and move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multi_task_model.to(device)

# Tokenize the input sentences using the tokenizer from the pre-trained model
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# Move the tokenized inputs to the appropriate device (GPU or CPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Remove unnecessary keys from inputs, keeping only 'input_ids' and 'attention_mask'
inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}

# Convert labels to PyTorch tensors and move them to the correct device
labels_task_a = torch.tensor(labels_task_a).to(device)
labels_task_b = torch.tensor(labels_task_b).to(device)

# Define the loss function for each task
loss_fn_task_a = nn.CrossEntropyLoss()
loss_fn_task_b = nn.CrossEntropyLoss()

# Define the optimizer, here AdamW is used with a learning rate of 1e-5
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, multi_task_model.parameters()), lr=1e-5)

# Scenario 1: Freeze the entire network (only train task-specific heads)
freeze_layers(multi_task_model, freeze_base=True, freeze_task_a=False, freeze_task_b=False)

# Scenario 2: Freeze only the transformer backbone
# freeze_layers(multi_task_model, freeze_base=True, freeze_task_a=False, freeze_task_b=False)

# Scenario 3: Freeze only one of the task-specific heads (e.g., Task A)
# freeze_layers(multi_task_model, freeze_base=False, freeze_task_a=True, freeze_task_b=False)



# Dummy training step 
multi_task_model.train()
for epoch in range(1):
    # Clear previous gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = multi_task_model(**inputs)  
    loss_task_a = loss_fn_task_a(outputs['classifier_task_a'], labels_task_a)  # Compute loss for Task A
    loss_task_b = loss_fn_task_b(outputs['classifier_task_b'], labels_task_b)  # Compute loss for Task B
    total_loss = loss_task_a + loss_task_b  

    # Backward pass to compute gradients
    total_loss.backward()  
    optimizer.step()  # Update model parameters

# Test the model
multi_task_model.eval()

# No need to compute gradients during testing
with torch.no_grad():  
    outputs = multi_task_model(**inputs)
    predictions_task_a = torch.argmax(outputs['classifier_task_a'], dim=1)  # Get predicted classes for Task A
    predictions_task_b = torch.argmax(outputs['classifier_task_b'], dim=1)  # Get predicted classes for Task B

# Print the obtained outputs and the mapped labels
print("Task A Outputs (Sentence Classification):")
print(predictions_task_a)
print("\nTask B Outputs (Sentiment Analysis):")
print(predictions_task_b)


# if the pre-trained model is really relevant:
# - Freeze the early layers (common low-level features).
# - Unfreeze the later layers (task-specific features).

# If the pre-trained model is less relevant, unfreeze additional layers to adapt to the new demands.
