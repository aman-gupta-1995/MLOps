import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import sklearn

with open("classifier.pkl", "rb") as file:
    model = pickle.load(file)

api = Flask(__name__)

@api.route("/")
def base():
    return "Home Page."