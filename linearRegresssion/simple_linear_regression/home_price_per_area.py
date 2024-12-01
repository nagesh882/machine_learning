import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_excel("/home/ubuntu/Desktop/machine_learning/linear-regression/linear_regression_single_variable.ods", engine="odf")
# print(df)

# plt.figure(figsize=(8, 6))
# plt.xlabel("Area (sq ft)")
# plt.ylabel("Price (USD)")
# plt.title("Home Price(USD) Based on Area(sq. ft.)")
# df.columns = df.columns.str.strip().str.lower()
# plt.scatter(df["area"], df["price"], color="red", marker="o", label="Data point")
# plt.plot(df["area"], df["price"], color="blue", label="Regression line")
# plt.legend()
# plt.savefig("home_price_as_per_area_square.jpg", format="jpg", dpi=300)

df.columns = df.columns.str.strip().str.lower()
# Compute the best-fit line (linear regression)
# y = mx + b
x = df["area"]
y = df["price"]
# Calculate the coefficients
m, b = np.polyfit(x, y, 1)  # 1 indicates a linear fit
# Generate the line values
regression_line = m * x + b
# Plotting
plt.figure(figsize=(8, 6))
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (USD)")
plt.title("Home Price (USD) Based on Area (sq. ft.)")
plt.scatter(x, y, color="red", marker="o", label="Data points")
plt.plot(x, regression_line, color="blue", label=f"Regression line: y={m:.2f}x+{b:.2f}")
plt.plot(x, y, color="green", label=f"Data vary line")
plt.legend()
plt.savefig("home_price_as_per_area_square.jpg", format="jpg", dpi=300)


new_df = df.drop("price", axis="columns")
# print(new_df)

model = linear_model.LinearRegression()
model.fit(new_df, df.price)

predicted_area = int(input("Enter Area(sq): "))
predicted_area_price = model.predict(pd.DataFrame([[predicted_area]], columns=["area"]))
print(f"Predicted price for {predicted_area} sq ft area: ${predicted_area_price[0]:.2f}")


predicted_area_df = pd.read_csv("/home/ubuntu/Desktop/machine_learning/linear-regression/areas.csv")
predicted_area_df_prices = model.predict(predicted_area_df)
predicted_area_df["price"] = np.round(predicted_area_df_prices, 2)
# print(predicted_area_df)
predicted_area_df.to_csv("predicted_area_df_prices.csv")