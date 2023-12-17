import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests

# Function to get live stock data from IEX Cloud API
def get_live_stock_data(stock_symbol):
    # Replace 'YOUR_IEX_CLOUD_API_KEY' with your actual IEX Cloud API key
    IEX_CLOUD_API_KEY = 'pk_b0ec6fdf52eb4455b40dad467a143467'

    # Fetch live stock data from IEX Cloud API
    url = f'https://cloud.iexapis.com/stable/stock/{stock_symbol}/chart/1y?token={IEX_CLOUD_API_KEY}'
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch live data. Status code: {response.status_code}")

    data = response.json()

    # Create a DataFrame with live stock data
    live_df = pd.DataFrame(data)
    live_df['date'] = pd.to_datetime(live_df['date'])
    live_df.set_index('date', inplace=True)
    return live_df

# Specify the stock symbol for which you want to train the model
stock_symbol = 'AAPL'

# Get live stock data
live_data = get_live_stock_data(stock_symbol)

# Feature engineering (you may need to extract more features from your dataset)
live_data['Day'] = live_data.index.dayofweek
live_data['Month'] = live_data.index.month
live_data['Year'] = live_data.index.year

# Use 'close' as the target variable
X = live_data[['Day', 'Month', 'Year']]
y = live_data['close']

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
joblib.dump(model, 'train_stock_model.joblib')
