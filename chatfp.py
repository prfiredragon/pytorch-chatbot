import random
import json

import torch

from model import NeuralNet, RNNModel
from nltk_utils import bag_of_words, tokenize
from tag_rules import rules_of_tags
from datetime import datetime

from pynode import Bridge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model = RNNModel(input_size,hidden_size,layer_dim,output_size,device).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
#print("Let's chat! (type 'quit' to exit)")
def home(request):
#while True:
    # sentence = "do you use credit cards?"
    #sentence = input("You: ")
    #if sentence == "quit":
    #    break
    print(request)
    message = request.split(":")
    sentence = tokenize(message[1])
    X = bag_of_words(sentence, all_words)
    #print(X.shape)
    X = X.reshape(1,1, X.shape[0])
    #print(X.shape)
    X = torch.from_numpy(X).to(device)
    #print(X.shape)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.84:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                complement = rules_of_tags(tag)
                print(f"Data for Debug : tag={tag} - prob={prob.item()} \n\n{bot_name}: {random.choice(intent['responses'])} {complement}")
                output = f"{random.choice(intent['responses'])} {complement}"
    else:
        output = f"I do not understand..."
    message[1] = output
    return message


class EchoWorker(Bridge):
    def __init__(self):
        super(EchoWorker, self).__init__()

        # Register for the echo event
        self.on('chat', self.handleEcho)

    def handleEcho(self, data):
        print("[python] Got:", repr(data))
        retval = home(data)
        self.emit('chat', retval)

# Create our worker
worker = EchoWorker()

# Loop, waiting for input
worker.loop()