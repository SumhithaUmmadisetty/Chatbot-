import random
import json
import pickle
import numpy as np 
import tensorflow as tf
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk 
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignoreletters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreletters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputempty = [0] * len(classes)

for document in documents:
    bag = []
    wordpatterns = document[0]
    wordpatterns = [lemmatizer.lemmatize(word.lower()) for word in wordpatterns]
    for word in words: 
        bag.append(1) if word in wordpatterns else bag.append(0)
    outputrow = list(outputempty)
    outputrow[classes.index(document[1])] = 1
    training.append(bag + outputrow) 

random.shuffle(training)
training = np.array(training)
trainX = training[:, :len(words)]  
trainY = training[:, len(words):]

model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

sgd = tf.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbot.h5', history)
print("Successfully Executed")
