import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer and load model and data
lemmatizer = WordNetLemmatizer()

# Load intents and model once
with open('D:\Chatbot\chatbot\Include\intents.json') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    # Convert to lower case and lemmatize
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            index = words.index(w)
            bag[index] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."

    tag = intents_list[0]['intent']
    responses = [i['responses'] for i in intents_json['intents'] if i['tag'] == tag]
    if responses:
        return random.choice(responses[0])
    return "Sorry, I didn't find a response."

print("Great: Bot is Running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

