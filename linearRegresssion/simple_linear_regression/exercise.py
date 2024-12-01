import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("/home/ubuntu/Desktop/machine_learning/linear-regression/canada_per_capita_income.csv")
# print(df)


# plt.figure(figsize=(8, 6))
# plt.xlabel("Year")
# plt.ylabel("Per Capita Income(USD)")
# plt.title("Canada Per Capita Income(USD)")
# df.columns = df.columns.str.strip().str.lower()
# plt.scatter(df["year"], df["per capita income (us$)"], color="red", marker="o", label="Data point")
# plt.plot(df["year"], df["per capita income (us$)"], color="blue", label="Regression line")
# plt.legend()
# plt.savefig("canada_per_capita_income.jpg", format="jpg", dpi=300)

df.columns = df.columns.str.strip().str.lower()
# Compute the best-fit line (linear regression)
# y = mx + b
x = df["year"]
y = df["per capita income (us$)"]
# Calculate the coefficients
m, b = np.polyfit(x, y, 1)  # 1 indicates a linear fit
# Generate the line values
regression_line = m * x + b
# Plotting
plt.figure(figsize=(8, 6))
plt.xlabel("Year")
plt.ylabel("Per Capita Income(USD)")
plt.title("Canada Per Capita Income(USD)")
plt.scatter(x, y, color="red", marker="o", label="Data points")
plt.plot(x, regression_line, color="blue", label=f"Regression line: y={m:.2f}x+{b:.2f}")
plt.plot(x, y, color="green", label=f"Data vary line")
plt.legend()
plt.savefig("canada_per_capita_income.jpg", format="jpg", dpi=300)



new_df = df.drop("per capita income (us$)", axis="columns")
# print(new_df)

model = linear_model.LinearRegression()
model.fit(new_df, df["per capita income (us$)"])


predict_year_capita_income = int(input("Enter Year: "))
predict_year_capita_income_df = model.predict(pd.DataFrame([[predict_year_capita_income]], columns=["year"]))
print(f"Predicted Year {predict_year_capita_income} for Capita Income: ${predict_year_capita_income_df[0]:.2f}")