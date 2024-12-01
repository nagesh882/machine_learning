import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("/home/ubuntu/Desktop/machine_learning/linearRegresssion/multivariant_regression/homeprices.csv")


df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
# print(df)


model = linear_model.LinearRegression()
model.fit(df.drop("price", axis="columns"), df.price)


p = model.predict(pd.DataFrame([[3000, 3, 40]], columns=["area", "bedrooms", "age"]))
print(f"Predicted Price of Home: {p[0]:.2f}")