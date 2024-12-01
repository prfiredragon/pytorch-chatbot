import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet, RNNModel


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 6000
batch_size = 100
#8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
#print(input_size, output_size)
n_iters = 3000

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)
print(f'Dataset size: {len(dataset)}')
print(f'sugested epochs size: {int(n_iters / (len(dataset) / batch_size))}')
#num_epochs = int(n_iters / (len(dataset) / batch_size))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = NeuralNet(input_size, hidden_size, output_size).to(device)
#num_words = torch.Size([8, 53])

for idx, (data, target) in enumerate(train_loader):
   print(f'Batch index: {idx}')
   print(f'Batch size: {data.size()}')
   #print(f'Batch size: {data}')
   #num_words = target.to(dtype=torch.long).to(device).size()
   #embedding = nn.Embedding(num_words, hidden_size)
   print(f'Batch label: {target}')
   print(f'Batch label size: {target.to(dtype=torch.long).to(device).size()}')
   break
#embedding = nn.Embedding(num_words, hidden_size)
model = RNNModel(input_size,hidden_size,layer_dim,output_size,device)
model.to(device)

#model = IntentRnnModel(
#   n_embeddings=input_size + 1,
#   n_embedding_dim=512,
#   padding_idx=23,
#   n_hidden_layer=3,
#   n_hidden_layer_neurons=512,
#   n_classes=output_size,
#).to(device)
#model.eval()
print(model)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

iter = 0
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        model.train()
        words = words.unsqueeze(1).to(device)
        labels = labels.to(dtype=torch.long).to(device)
        #embedding = nn.Embedding(words, hidden_size)
        #print(words.shape)
        #print(f'Batch size: {words.size()}')
        #print(labels)
        # Forward pass
        outputs = model(words)
        #print(outputs)
        #print(hn)
        # if y would be one-hot, we must apply
        #labels = torch.max(labels, 1)[1]
        #print(labels)
        loss = criterion(outputs, labels)
        #labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
       model.eval()
       correct = 0
       total = 0
       for words, labels in test_loader:
          words = words.unsqueeze(1).to(device)
          labels = labels.to(dtype=torch.long).to(device)
          outputs = model(words)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
       accuracy = 100 * correct / total
       print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, accuracy: {accuracy}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

model.eval()
scripted_model = torch.jit.script(model)
SMFILE = "chatbot/chatbot.pt"
torch.jit.save(scripted_model, SMFILE)
print(f'scripted_model export complete. file saved to {SMFILE}')
