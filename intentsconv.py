import json

with open('intents.json', 'r') as f:
    intentsj = json.load(f)

intents = []
for intent in intentsj['intents']:
    if intent['responses'][0] != ' ':    
        d = {
            "tag": f"{intent['tag']}",
            "patterns": intent['patterns'],
            "responses": intent['responses']}

        intents.append(d)

# Convert it to Dictionary and Save it as a JSON 
dic = {'intents':intents}

#json_object = json.loads(dic)

# Serializing json
json_formatted_str = json.dumps(dic, indent=2)

# Writing to sample.json
with open("intents_new.json", "w") as outfile:
    outfile.write(json_formatted_str)