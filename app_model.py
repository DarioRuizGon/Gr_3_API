from flask import Flask, jsonify, request
import os
import pickle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods = ['GET'])
def hello():
    return """
    <h1>API de predicciones sobre el color de un vino</h1>
    <p>Para hacer una predicción utiliza el endpoint '/api/v1/predict' con los siguientes: 
    parámetros acidity ('volatile acidity'); chlorides; so2 ('total sulfur dioxide'); sulphates</p>
    """

@app.route('/api/v1/predict', methods = ['GET'])
def predict():
    model = joblib.load("my_model.joblib")

        
    acidity = request.args.get('acidity', None)
    chlorides = request.args.get('chlorides', None)
    so2 = request.args.get('so2', None)
    sulphates = request.args.get('sulphates', None)

    

    if acidity is None or chlorides is None or so2 is None or sulphates is None:
        return 'Faltan argunmenos, por favor revisa tu petición'
    else:
        prediction = model.predict([[float(acidity),float(chlorides),float(so2), float(sulphates)]])


    
    return jsonify({'prediction': int(prediction[0])})


# @app.route('/api/v1/predict', methods = ['GET'])
# def predict():

#     with open('wine_model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     acidity = request.args.get('acidity', None)
#     chlorides = request.args.get('chlorides', None)
#     so2 = request.args.get('so2', None)
#     sulphates = request.args.get('sulphates', None)

#     def prediction(X):
#         pred = int(model.predict(X)[0])
#         mapping = {0: "red", 1: "white"}
#         return jsonify({'prediction': mapping[pred]})

#     X = [acidity, chlorides, so2, sulphates]

#     def type_processing(X):
#         result = []
#         for n in X:
#             try:
#                 value = float(n)
#             except:
#                 value = None
#             result.append(value)
#         return result
    
#     X = np.array(type_processing(X)).reshape(-1, 4)

#     return prediction(X)

if __name__ == '__main__':
    app.run(debug=True)


