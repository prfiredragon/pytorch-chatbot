# Implementation of a Contextual Chatbot in PyTorch.  
## This Version 
This version is a modified version. This package is for reference. I found on the Internet many questioning if pytorch can be used on the web.
- using data from my self
- using data from another projects
- ussing RNN insted of NeuralNet
- Using rules for call functions like to get the date and time
- Using django for webchat UI
- Flask rest-api with test webchat UI

### Under Dev
- nodejs webchat ui version under dev



## Installation

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
$ mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate it
Mac / Linux:
```console
$ . venv/bin/activate
```
Windows:
```console
$ venv\Scripts\activate
```
### Install PyTorch and dependencies
Run
 ```console
$ pip install -r requirements.txt
 ```


For Information of PyTorch see [official website](https://pytorch.org/).


If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

### Install node dependencies
Run
 ```console
$ cd chatbot-js
$ npm install
$ cd ..
```

## Usage on console
Run
```console
$ python train.py
```
This will dump `data.pth` file. And then run
```console
$ python chat.py
```

## Usage on web with django
Run
```console
$ cp intents.json ./chatbot/
$ cp data.pth ./chatbot/
$ python manage.py runserver
```
on the webBrowser go to "http://localhost:8000/"


## Usage on web with node
Run
```console
$ node ./chatbot-js/index.js
```
[//]: # (## Usage on web with node)
[//]: # (Run)
[//]: # (```console)
[//]: # ($ node ./chatbot-js/index.js)
[//]: # (```)


## Usage on web with flask

go to the directory
```
$ cd chatbot-flask
```

install requirenmnts
```
$ pip install -r requirements.txt
```

run
```
$ python app.py
```




# Credits
## Intends.json
- Original Intends.json : [https://github.com/patrickloeber/pytorch-chatbot/blob/master/intents.json](https://github.com/patrickloeber/pytorch-chatbot/blob/master/intents.json)
- First Aid Recommendation Deep Learning ChatBot : [https://www.kaggle.com/code/therealsampat/first-aid-recommendation-deep-learning-chatbot/input](https://www.kaggle.com/code/therealsampat/first-aid-recommendation-deep-learning-chatbot/input)

## Code
- PYTorch with Dango : [https://github.com/Academy-Omen/torched-django](https://github.com/Academy-Omen/torched-django)
- WebUI used on nodejs implementation : [https://pusher.com/tutorials/live-chat-with-node-js-mysql-and-pusher-channels-part-2/#build-live-chat-with-pusher-channels](https://pusher.com/tutorials/live-chat-with-node-js-mysql-and-pusher-channels-part-2/#build-live-chat-with-pusher-channels)
- Flask rest-api : [https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- Webui used on the flask rest-api demo [https://github.com/patrickloeber/chatbot-deployment/tree/main/standalone-frontend](https://github.com/patrickloeber/chatbot-deployment/tree/main/standalone-frontend)


# Old pytorch-chat info
## Description

Simple chatbot implementation with PyTorch.

- The implementation should be easy to follow for beginners and provide a basic understanding of chatbots.
- The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
- Customization for your own use case is super easy. Just modify `intents.json` with possible patterns and responses and re-run the training (see below for more info).

The approach is inspired by this article and ported to PyTorch: [https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077).

## Watch the Tutorial
[![Alt text](https://img.youtube.com/vi/RpWeNzfSUHw/hqdefault.jpg)](https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg)

## Installation

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate it
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```
### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```
## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```
