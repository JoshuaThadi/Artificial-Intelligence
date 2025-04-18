import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
print("Python Version:", sys.version)

import random
import json
import pickle
import numpy as np

import nltk 
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout  # type: ignore
from tensorflow.keras.optimizers import SGD  # type: ignore

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))  # Fixed the wrong variable

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]  # Fix incorrect reference
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)  # Fix NumPy slicing issue

train_x = np.array([entry[0] for entry in training])  # Fixed slicing
train_y = np.array([entry[1] for entry in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

stochastic_gradient_descent = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Fixed `lr`
model.compile(loss="categorical_crossentropy", optimizer=stochastic_gradient_descent, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras')
print("Done")

