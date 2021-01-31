import random
import os
from datetime import datetime
import pickle
import nltk
import requests
import numpy as np
from flask import Flask, request
from pymongo import MongoClient
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

model = load_model('chatbot_model.h5')
client = MongoClient(os.environ.get('CONNECTION_STRING'))
db = client.intents
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()


app = Flask(__name__)

@app.route('/', methods=['POST'])
def chat():
    list_of_intents = db.intents.find()
    sentence = request.json['sentence']
    if not sentence:
        return 'You need to say something.'
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    p = (np.array(bag))
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for res in results:
        return_list.append({"intent": classes[res[0]], "probability": str(res[1])})
    tag = return_list[0]['intent']

    if tag == 'weather':
        response = requests.get('https://www.metaweather.com/api/location/1154781/')
        data = response.json()['consolidated_weather'][0]
        return f"It is currently {round(data['the_temp'], 1)} degree celsius with {data['humidity']}% humidity."

    if tag == 'time':
        return f"It is {datetime.now().strftime('%I:%M %p')}."

    if tag == 'date':
        return f"It is {datetime.now().strftime('%A, %d-%m-%Y')}"

    for intent in list_of_intents:
        if intent['name']== tag:
            result = random.choice(intent['responses'])
            return result
    return random.choice([
        "Unfortunately, my knowledge is limited.",
        "Sorry, but I can't understand you.",
        "Sorry, I'm not smart enough yet."
    ])
