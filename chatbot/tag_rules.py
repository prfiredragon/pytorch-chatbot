from datetime import datetime
import pytz

def rules_of_tags(tag):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    complement = ""
    match tag:
       case "date":
          complement = "{:%a %d %b %Y}".format(datetime.now(pytz.timezone('America/Puerto_Rico')))
       case "time":
          complement = "{:%I:%M%p}".format(datetime.now(pytz.timezone('America/Puerto_Rico')))
       case "PHP":
          print("You can become a backend developer")
       case "Solidity":
          print("You can become a Blockchain developer")
       case "Java":
          print("You can become a mobile app developer")
       case _:
          complement = ""

    return complement
