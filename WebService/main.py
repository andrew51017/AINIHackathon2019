from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin
import os
import sys
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.initializers import Constant
from keras.models import model_from_json

import pickle

MAX_SEQUENCE_LENGTH = 200
MAX_NUM_WORDS       = 20000

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/classify", methods=["POST"])
@cross_origin()
def classifySample():
    texts = request.json['texts']
    
    testSequences = tokenizer.texts_to_sequences(texts)
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)    
    testDataProb = model.predict(testData)
    testDataClass = testDataProb.argmax(axis=-1)

    return jsonify(testDataClass.tolist())


if __name__ == '__main__':

    # Load the model

    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    model = loaded_model
    print("Loaded model from disk")

    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    app.run(host='0.0.0.0')    

