import json
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
import numpy as np

from pytorchnltk import tokenize, Stemming, bag_of_words
from pytorchmodel import NeuralNet

# Load intents.json
file_path = r"F:\All about Ai\Ai Projects\ChatBot AI\Trent Bot\intents.json"
with open(file_path, 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
x_y = []

# Process intents
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        x_y.append((w, tag))

ignore_words = ['?', "!", ",", "."]
all_words = [Stemming(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Prepare training data
x_train = []
y_train = []
for (pattern_sentence, tag) in x_y:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Define Dataset Class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = torch.tensor(x_train, dtype=torch.float32)  # Ensure correct data type
        self.y_data = torch.tensor(y_train, dtype=torch.long)  # Ensure correct label type

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  

    def __len__(self):
        return self.n_samples

# Create DataLoader
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers=0 for Windows

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device).float()  # Ensure words are float tensors
        labels = labels.to(device).long()  # Ensure labels are long tensors
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss={loss.item():.4f}')
        
print(f'Final Loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
