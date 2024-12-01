import os
import random
import json
import torch

from .model import NeuralNet, RNNModel
from .nltk_utils import bag_of_words, tokenize
from .tag_rules import rules_of_tags
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, "intents.json")  # full path to text.

with open(file_path, "r") as json_data:
    intents = json.load(json_data)

FILE = os.path.join(module_dir, "data.pth")  # full path to text.

data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER

#model = NeuralNet(input_size, hidden_size, output_size).to(device)
model = RNNModel(input_size,hidden_size,layer_dim,output_size,device).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
def home(request):

    # get user input from request
    input = request.GET.get("input")
    print("Input:", input)
    output = "Let's chat! (type 'quit' to exit)"

    if input == "quit":
        output = "Goodbye, You Just Quited"
    
    # ------------------------------------------
    if input:

        sentence = tokenize(input)
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
                    output = f"{random.choice(intent['responses'])} {complement}"
        else:
            output = f"I do not understand..."


    context = {"input": input, "output": output}
    return render(request, "index.html", context)


def jsonhome(request):

    # get user input from request
    input = request.GET.get("input")
    print("Input:", input)
    output = ""

    if input == "quit":
        output = "Goodbye, You Just Quited"
    
    # ------------------------------------------
    if input:

        sentence = tokenize(input)
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
                    output = f"{random.choice(intent['responses'])} {complement}"
        else:
            output = f"I do not understand..."
        output = JsonResponse({"output": output})


    context = {"input": input, "output": output}
    return output
# Create your views here.
