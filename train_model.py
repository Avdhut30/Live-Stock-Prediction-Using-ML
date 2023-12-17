import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create synthetic data with 'Date' and 'Close' arrays having the same length
date_range = pd.date_range('2020-01-01', '2023-01-12', freq='D')
close_prices = np.random.rand(len(date_range)) * 100 + 100

# Create a dictionary with arrays of the same length
data = {
    'Date': date_range,
    'Close': close_prices
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Feature engineering (you may need to extract more features from your dataset)
df['Day'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Use 'Close' as the target variable
X = df[['Day', 'Month', 'Year']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Save the trained model for later use in the Flask app
import joblib
joblib.dump(model, 'stock_prediction_model.joblib')
