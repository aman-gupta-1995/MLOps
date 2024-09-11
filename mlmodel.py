import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import pickle
import joblib

# model = load_model("model.h5")
model = joblib.load("model.joblib")
# with open("model.pkl", "rb") as file:
#     model = pickle.load(file)

df = pd.read_csv("iris.csv")
sl_max = df["sepal_length"].max()
sl_min = df["sepal_length"].min()
sl_avg = df["sepal_length"].mean()

sw_max = df["sepal_width"].max()
sw_min = df["sepal_width"].min()
sw_avg = df["sepal_width"].mean()

pl_max = df["petal_length"].max()
pl_min = df["petal_length"].min()
pl_avg = df["petal_length"].mean()

pw_max = df["petal_width"].max()
pw_min = df["petal_width"].min()
pw_avg = df["petal_width"].mean()

#SCALE THE INPUTS IF WE DID IN TRAINING ALSO. DO EVERYTHING ON INPUT WE DID IN TRAINING.
st.write("# IRIS Prediction\n")
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    sl_input = st.slider(label = "Sepal Length", min_value = sl_min, max_value= sl_max, 
                         value = sl_avg, key = "Sepal_Length")
with col2:
    sw_input = st.number_input(label = "Sepal Width", min_value= sw_min, max_value= sw_max, 
                    value = sw_avg, format = "%0.2f", key = "Sepal_Width")
with col3:
    pl_input = st.slider(label = "Petal Length", min_value = pl_min, max_value= pl_max, 
              value = pl_avg, key = "Petal_Length")
with col4:
    pw_input = st.number_input(label = "Petal Width", min_value= pw_min, max_value= pw_max, 
                    value = pw_avg, format = "%0.2f", key = "Petal_Width")

if sl_input > sl_max:
    st.error(f"Sepal Length Value cannot exceed {sl_max}. Please enter a value between {sl_min} and {sl_max}.")

input = np.array([[sl_input, sw_input, pl_input, pw_input]])

d = {0: "Setosa",
     1: "Versicolor",
     2: "Verginica"}

if st.button("Predict"):
    p = model.predict(input)
    v = np.argmax(p)
    result = d[v]
    confidence = np.around(np.max(p)*100,2)
    st.write(f"Category : {result} Confidence: {confidence}" )














# import streamlit as st
# import time

# # Cache to limit how often the function is executed
# @st.cache(allow_output_mutation=True)
# def get_value():
#     return {"value": 0}

# value_state = get_value()
# user_input = st.number_input("Enter a value:", min_value=0, max_value=100, value=value_state["value"])

# # Update cached value with a delay to simulate debouncing
# if st.button('Update Value'):
#     value_state["value"] = user_input
#     time.sleep(0.5)  # Add a delay to prevent rapid updates

# st.write(f"Entered value: {value_state['value']}")
