# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv("housing.csv")

# Features and target
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with multiple features!")
