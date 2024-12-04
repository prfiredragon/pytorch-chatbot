import io
import random
from flask import Flask, jsonify, request, send_from_directory, Response
from waitress import serve
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import random
import json


import torch

from model import NeuralNet, RNNModel
from nltk_utils import bag_of_words, tokenize
from tag_rules import rules_of_tags
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

app = Flask(__name__)


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
    sentence = tokenize(request)
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
                response = random.choice(intent['responses'])
                print(f"Data for Debug : tag={tag} - prob={prob.item()} \n\n{bot_name}: {response} {complement}")
                output = f"{response} {complement}"
                output = jsonify({'class_id': 'MESSAGE_NET_RRC', 'class_name': output, 'tag': tag})
    else:
        output = f"I do not understand..."
        output = jsonify({'class_id': 'MESSAGE_NET_RRC', 'class_name': output, 'tag': 'none'})
    return output


@app.route('/')
def hello():
    return "<h1>under Dev!</h1>"

@app.route('/result/<path:file>', defaults={'file': 'index.html'})
def serve_result(file):
    abort(404)

@app.route('/results/<path:file>')
def serve_results(file):
    print(file)
    if file == '':
       file = 'index.html'
    # Haven't used the secure way to send files yet
    return send_from_directory('results/', file)

@app.route('/predict', methods=['GET', 'POST'])
def predict():    
   if request.method == 'POST':
      print(request.json)
      mess = request.json['chat']
      retmess = home(mess)
      #f"Echo : {mess}"
      return  retmess
   else:
      print(request.args)
      print(request.args.get("chat"))
      if request.args:
         mess = request.args.get("chat")
      else:
         mess = 'Cat'
      return jsonify({'class_id': 'MESSAGE_NET_XXX', 'class_name': mess})

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig


if __name__ == '__main__':
   print("pres CTRL+C to stop the server")
   serve(app, host="0.0.0.0", port=5000)

