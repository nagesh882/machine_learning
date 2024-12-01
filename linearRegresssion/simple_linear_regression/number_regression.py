import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



x = np.array(range(1, 41)).reshape(-1, 1)
y = np.array(range(2, 42))


model = linear_model.LinearRegression()
model.fit(x, y)


predicted_number = int(input("Enter Number: "))
p = model.predict([[predicted_number]])
print(f"Predicted Next Number of => {predicted_number} is => {int(p[0])}")
