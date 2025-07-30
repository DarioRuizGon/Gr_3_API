from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods = ["GET"])
def hello():
    return "<h1>API de predicciones sobre el color de un vino</h1>"

@app.route("/api/v1/predict", methods = ["GET"])
def predict():
    # with open("wine_model_2.pkl", "rb") as f:
    #     model = pickle.load(f)

    # def prediction(X):
    #     pred = model.predict(X)
    #     return jsonify({f"prediction_{i}": pred[i].astype(np.int8) for i in range(len(X))})

    
    acidity = request.args.get("acidity", None)
    chlorides = request.args.get("chlorides", None)
    so = request.args.get("so", None)
    sulphates = request.args.get("sulphates", None)

    return jsonify({"data": (acidity, type(acidity), chlorides, type(chlorides), so, type(so), sulphates, type(sulphates))})

    # if acidity is None or chlorides is None or so2 is None or sulphates is None:
    #     return "Args empty, not enough data to predict"
    # else:
    #     X_pred = pd.DataFrame([[float(acidity),float(chlorides),float(so2), float(sulphates)]], columns = [f"Column_{i}" for i in range(4)])
    #     prediction = model.predict(X_pred)
    
    # return jsonify({'predictions': prediction[0]})


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



