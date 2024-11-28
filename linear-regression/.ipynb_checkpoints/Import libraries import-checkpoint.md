# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example dataset: Numbers from 1 to 40
X = np.array(range(1, 41)).reshape(-1, 1)  # Features (numbers 1 to 40)
y = np.array(range(2, 42))  # Target values (next number in the sequence)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict the next number for 40
next_number = model.predict([[40]])
print(f"Predicted next number after 40: {next_number[0]}")

# Visualization
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Current Number")
plt.ylabel("Next Number")
plt.title("Linear Regression to Predict the Next Number")
plt.legend()
plt.show()

