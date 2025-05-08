import os
import random
import json
import torch
from pytorchmodel import NeuralNet
from pytorchnltk import bag_of_words, tokenize

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents.json
file_path = r"F:\All about Ai\Ai Projects\ChatBot AI\Trent Bot\intents.json"
with open(file_path, 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE, map_location=device)

hidden_size = data["hidden_size"]
output_size = data["output_size"]
input_size = data["input_size"]
tags = data["tags"]
model_state = data["model_state"]
all_words = data["all_words"]

# Initialize and load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Set model to evaluation mode

bot_name = "Trent"
print("Let's talk! My Highness")
while True:
    sentence = input('You -> ')
    if sentence.lower() == "quit":
        break
    
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)  # Fix: Pass `x` to `from_numpy()`
    
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f'{bot_name} -> {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name} -> I do not understand...')
