import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import sklearn
import catboost
from extra import car_enc, categories, column_name

model = joblib.load("cgbt.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")
mean_enc = np.mean(np.array(list(car_enc.values())))

api = Flask(__name__)

@api.route("/")
def base():
    return "Home Page."

@api.route("/<name>")
def user(name):
    return {"message": f"Hello {name}!"}

@api.route("/user/<name>")
def user1(name):
    return f"Hello {name}!"

@api.route("/predict", methods = ["POST"])
def prediction():
    data = request.get_json()
    # if not data:
    #     return jsonify({"error":"No Data Provided"}),400
    
    # return jsonify({
    #     "message": "data processed successfully",
    #     "data": data
    # }), 200
    # return {"year": data["year"]}
    df = pd.DataFrame(data, index = [0])

    df["company_name"] = df["company_name"].fillna(mean_enc)
    df["company_name"] = df["company_name"].map(car_enc)

    def manual_one_hot_encode(dff, column, categories):
        sorted_categories = sorted(categories)
        baseline_category = sorted_categories[0]

        for cat in sorted_categories[1:]:
            dff[f"{column}_{cat}"] = 0

        for i, j in dff[column].items():
            if j != baseline_category:
                dff.loc[i, f"{column}_{j}"] = 1
            
        return dff.drop(columns = [column])
    
    df = manual_one_hot_encode(df, "seller_type", categories["seller_type"])
    df = manual_one_hot_encode(df, "fuel_type", categories["fuel_type"])
    df = manual_one_hot_encode(df, "transmission_type", categories["transmission_type"])

    df = df[column_name]
    df = df.rename(columns = {"company_name":"company_enc"})

    df_scaled = scaler.transform(df)

    car_price_pred = model.predict(df_scaled)
    car_price_pred = np.around(car_price_pred[0],5)

    return {"Car Price": car_price_pred}

    

    




