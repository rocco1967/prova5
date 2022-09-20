# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:23:55 2022

@author: 39333
"""
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('model_catboost')
cols = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate','Body_Temp']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen,round=0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='calorie per givolong {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
        app.run(debug=True)