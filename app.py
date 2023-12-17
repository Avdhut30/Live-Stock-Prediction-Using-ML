from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Replace 'YOUR_IEX_CLOUD_API_KEY' with your actual IEX Cloud API key
IEX_CLOUD_API_KEY = 'pk_b0ec6fdf52eb4455b40dad467a143467'
MODEL_PATH = 'stock_prediction_model.joblib'

# Load the trained model
model = joblib.load(MODEL_PATH)

def extract_features():
    # Implement feature extraction logic based on your model training features
    # For simplicity, using placeholders here
    day = pd.to_datetime('today').dayofweek
    month = pd.to_datetime('today').month
    year = pd.to_datetime('today').year
    return {'Day': day, 'Month': month, 'Year': year}

def get_historical_data(stock_symbol, days=30):
    # Fetch historical stock data from IEX Cloud API
    url = f'https://cloud.iexapis.com/stable/stock/{stock_symbol}/chart/{days}d?token={IEX_CLOUD_API_KEY}'
    response = requests.get(url)
    data = response.json()

    # Create a DataFrame with historical stock data
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def calculate_profit_loss(reference_price, current_price):
    return (current_price - reference_price) / reference_price * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    try:
        stock_symbol = request.form['stockSymbol']
        if not stock_symbol:
            return jsonify({'error': 'Please enter a stock symbol'})

        # Fetch live stock data from IEX Cloud API
        url = f'https://cloud.iexapis.com/stable/stock/{stock_symbol}/quote?token={IEX_CLOUD_API_KEY}'
        response = requests.get(url)
        data = response.json()

        # Extract relevant information (e.g., current stock price)
        stock_price = float(data['latestPrice'])
        
        # Get historical data for the last 30 days
        historical_data = get_historical_data(stock_symbol)

        # Calculate profit/loss
        reference_price = historical_data['close'].iloc[0]
        profit_loss = calculate_profit_loss(reference_price, stock_price)

        # Compare with the average closing price of the last 30 days
        avg_close_price_last_30_days = historical_data['close'].mean()
        status = 'Profit' if stock_price > avg_close_price_last_30_days else 'Loss'

        prediction_info = predict_stock_price(stock_price, profit_loss, status)  # Replace with your prediction logic

        return jsonify({'stockPrice': stock_price, 'prediction': prediction_info, 'status': status})

    except Exception as e:
        return jsonify({'error': str(e)})

def predict_stock_price(current_price, profit_loss, status):
    try:
        # Implement your stock price prediction logic using the loaded model
        # This is a placeholder, and a real implementation requires a trained model
        features = extract_features()  # Implement the feature extraction logic

        # Create a DataFrame with named features
        feature_df = pd.DataFrame([features])

        prediction = model.predict(feature_df)[0]

        if prediction > 500:
            recommendation = 'Buy'
        elif prediction < 500:
            recommendation = 'Sell'
        else:
            recommendation = 'Hold'

        # Print prediction information
        print("Raw Prediction Value:", prediction)
        print("Recommendation:", recommendation)
        print("Status:", status)

        # Add detailed prediction information
        if abs(prediction) > 0.5:  # You can adjust this threshold
            prediction_detail = 'Significant Change'
        else:
            prediction_detail = 'Minor Change'

        return {
            'direction': 'Up' if prediction > 0 else 'Down',
            'recommendation': recommendation,
            'detail': prediction_detail,
            'status': status
        }

    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True)
