from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods = ["GET"])
def hello():
    return "<h1>API de predicciones sobre el color de un vino</h1>"

@app.route("/api/v1/predict", methods = ["GET"])
def predict():
    with open("wine_model.pkl", "rb") as f:
        model = pickle.load(f)

    def prediction(X):
        pred = model.predict(X)
        return jsonify({f"prediction_{i}": pred[i].astype(np.int8) for i in range(len(X))})

    acidity = request.args.get("acidity", None)
    chlorides = request.args.get("chlorides", None)
    so2 = request.args.get("so2", None)
    sulphates = request.args.get("sulphates", None)

    X = [float(acidity), float(chlorides), float(so2), float(sulphates)]

    if len(X[X == None]) >= 2:
        cont = input("La mitad o más de los valores son faltantes, si quieres continuar pulsa 's', si no 'n'").lower()
        if cont == "s":
            return prediction(X)
        else:
            return "Te faltaban muchos datos y has decidido no hacer una predicción"
        
    else:
        return prediction(X)



