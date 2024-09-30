import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df1 = pd.read_csv("MLFlow/car_price.csv")
X = df1.drop(columns = ["full_name", "company", "selling_price"])
y = df1["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, train_size = .8,
                                                    shuffle = True)

scaler = MinMaxScaler(feature_range = (0,1))
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)