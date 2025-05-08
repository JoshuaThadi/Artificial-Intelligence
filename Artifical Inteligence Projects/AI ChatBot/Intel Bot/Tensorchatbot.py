import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
print("Python Version:", sys.version)

import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model  # type: ignore

# Load NLP tools
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')  # Ensure tokenization works

# Load necessary files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    # Return detected intents with probability
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results] if results else []

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Can you rephrase?"
    
    tag = intents_list[0]['intent']  # Fixed key name
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    
    return "I'm not sure how to respond to that."

print("Processing! Bot is running!")
while True:
    message = input("You -> ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Intel -> ", res)
