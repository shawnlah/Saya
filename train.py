from pymongo import MongoClient
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import os
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

client = MongoClient(os.environ.get('CONNECTION_STRING'))
db = client.intents

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
# we might need apostrophe
words_to_ignore = ['?', '!']

for intent in db.intents.find():
    name = intent.get('name')
    for pattern in intent.get('patterns', []):
        # tokenize/split sentence
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent.get('name')))

        if name not in classes:
            classes.append(name)

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in words_to_ignore]
# remove duplicates and sort
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output = [0] * len(classes)

for document in documents:
    bag = []
    pattern_words = document[0]
    # lemmatizing the word means changing it to its base value, e.g: playing => play
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        if word in pattern_words:
            # we found which "tag" it belongs to
            bag.append(1)
            continue
        bag.append(0)

    output_row = list(output)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# SGD = Stochastic gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
