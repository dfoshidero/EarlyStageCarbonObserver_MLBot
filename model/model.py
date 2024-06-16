import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from transformers import BertModel

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(current_dir, 'export')

# Load data
tokenized_features = np.load(os.path.join(export_dir, 'tokenized_features.npy'))
numerical_features = np.load(os.path.join(export_dir, 'numerical_features.npy'))
targets = np.load(os.path.join(export_dir, 'targets.npy'))

# Convert to tensors
tokenized_features = torch.tensor(tokenized_features, dtype=torch.long)
numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Combine tokenized and numerical features into a single dataset
dataset = TensorDataset(tokenized_features, numerical_features, targets)

# Split dataset into training and testing sets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for batching
batch_size = 32  # Adjust as necessary
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MultiModalModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_numerical_features=10, output_size=1):
        super(MultiModalModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.num_numerical_features = num_numerical_features
        
        # Define layers for numerical features
        self.fc1 = nn.Linear(num_numerical_features, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Define layers for combined features
        self.fc_combined = nn.Linear(self.bert.config.hidden_size + 64, 128)
        self.fc_output = nn.Linear(128, output_size)

    def forward(self, tokenized_input, numerical_input):
        # Process tokenized text through BERT
        bert_outputs = self.bert(input_ids=tokenized_input, attention_mask=(tokenized_input != 0))
        pooled_output = bert_outputs.pooler_output
        
        # Process numerical features
        x = F.relu(self.fc1(numerical_input))
        x = F.relu(self.fc2(x))
        
        # Combine features
        combined = torch.cat((pooled_output, x), dim=1)
        combined = F.relu(self.fc_combined(combined))
        output = self.fc_output(combined)
        
        return output

# Instantiate the model, define loss function and optimizer
model = MultiModalModel(num_numerical_features=numerical_features.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as necessary
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for tokenized_batch, numerical_batch, targets_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(tokenized_batch, numerical_batch)
        loss = criterion(outputs.squeeze(), targets_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

model.eval()
test_loss = 0.0
with torch.no_grad():
    for tokenized_batch, numerical_batch, targets_batch in test_loader:
        outputs = model(tokenized_batch, numerical_batch)
        loss = criterion(outputs.squeeze(), targets_batch)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader)}")