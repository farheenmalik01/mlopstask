# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Create a simple linear regression model
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')
