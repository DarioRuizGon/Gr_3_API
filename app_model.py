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
    return '<h1>API de predicciones sobre el color de un vino</h1>'

@app.route('/api/v1/predict', methods = ['GET'])
def predict():
    model = joblib.load("my_model.joblib")

    # def prediction(X):
    #     pred = model.predict(X)
    #     return jsonify({f'prediction_{i}': pred[i].astype(int) for i in range(len(X))})

    
    acidity = request.args.get('acidity', None)
    chlorides = request.args.get('chlorides', None)
    so = request.args.get('so', None)
    sulphates = request.args.get('sulphates', None)

    

    if acidity is None or chlorides is None or so is None or sulphates is None:
        return 'Args empty, not enough data to predict'
    else:
        prediction = model.predict([[float(acidity),float(chlorides),float(so), float(sulphates)]])

    #return jsonify({'data': (float(acidity), float(chlorides), float(so), float(sulphates))})
    
    return jsonify({'predictions': prediction[0]})


    # X = [acidity, chlorides, so2, sulphates]

    # def type_processing(X):
    #     result = []
    #     for n in X:
    #         try:
    #             value = float(n)
    #         except:
    #             value = None
    #         result.append(value)
    #     return result
    
    # X = np.array(type_processing(X)).reshape(-1, 4)

    # return prediction(X)

if __name__ == '__main__':
    app.run(debug=True)


