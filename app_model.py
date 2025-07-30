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
# import warnings
# warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods = ['GET'])
def hello():
    return """
    <h1>API de predicciones sobre el color de un vino</h1>
    <p>Usa /api/v1/predict con los parámetros acidity, chlorides, so, sulphates</p>
    """

# @app.route('/api/v1/predict', methods = ['GET'])
# def predict():
#     model = joblib.load("my_model.joblib")

        
#     acidity = request.args.get('acidity', None)
#     chlorides = request.args.get('chlorides', None)
#     so = request.args.get('so', None)
#     sulphates = request.args.get('sulphates', None)

    

#     if acidity is None or chlorides is None or so is None or sulphates is None:
#         return 'Args empty, not enough data to predict'
#     else:
#         prediction = model.predict([[float(acidity),float(chlorides),float(so), float(sulphates)]])


    
#     return jsonify({'prediction': int(prediction[0])})


@app.route('/api/v1/predict', methods = ['GET'])
def predict():

    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)

    acidity = request.args.get('acidity', None)
    chlorides = request.args.get('chlorides', None)
    so = request.args.get('so', None)
    sulphates = request.args.get('sulphates', None)

    def prediction(X):
        pred = model.predict(X)
        pred = int(pred)
        if pred == 0:
            pred = "red"
        else:
            pred = "white"
        return jsonify({'prediction': pred})

    X = [acidity, chlorides, so, sulphates]

    def type_processing(X):
        result = []
        for n in X:
            try:
                value = float(n)
            except:
                value = None
            result.append(value)
        return result
    
    X = np.array(type_processing(X)).reshape(-1, 4)

    return prediction(X)

if __name__ == '__main__':
    app.run(debug=True)


