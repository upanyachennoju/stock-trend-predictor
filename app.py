import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import tensorflow as tf

@st.cache_resource
def load_trained_model():
    model_path = "stock_predictor_model.h5"
    
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please upload 'stock_predictor_model.h5' to your app directory.")
        return None
    
    return tf.keras.models.load_model(model_path)


# Load Stock Data
def load_data(stock_symbol, start, end):
    df = yf.download(stock_symbol, start=start, end=end)
    return df[['Close']]

# Sentiment Analysis (Using API Instead of Scraping)
def get_sentiment(stock_symbol):
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}&apikey=YOUR_API_KEY"
        response = requests.get(url).json()
        sentiments = [article["overall_sentiment_score"] for article in response["feed"]]
        return np.mean(sentiments) if sentiments else 0
    except:
        return 0  # Return neutral sentiment if API fails

# Preprocess Data
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    X = np.array([data_scaled[i:i+time_step] for i in range(len(data_scaled) - time_step)])
    return X, scaler

# Predict Future Prices
def predict_future(model, data, scaler, days=7):
    future_prices = []
    last_60_days = data[-60:]

    for _ in range(days):
        input_data = last_60_days.reshape(1, 60, 1)
        predicted_price = model.predict(input_data)
        future_prices.append(predicted_price[0,0])
        last_60_days = np.append(last_60_days[1:], predicted_price)

    return scaler.inverse_transform(np.array(future_prices).reshape(-1,1))

# Streamlit UI
st.title("📈 Stock Market Trend Predictor")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
num_days = st.slider("Predict Future Days", 1, 10, 7)

if st.button("Predict"):
    with st.spinner("Fetching data..."):
        data = load_data(stock_symbol, '2020-01-01', '2024-01-01')
        sentiment = get_sentiment(stock_symbol)
        st.write(f"**Current Sentiment Score:** {sentiment:.2f}")

    with st.spinner(f"Predicting next {num_days} days..."):
        model = load_trained_model()  # Load the trained model
        X, scaler = preprocess_data(data.values)
        future_prices = predict_future(model, X, scaler, days=num_days)
        future_dates = pd.date_range(start=data.index[-1], periods=num_days+1)[1:]

    # Plot Predictions
    st.subheader(f"📉 Predicted Prices for Next {num_days} Days")
    fig, ax = plt.subplots()
    ax.plot(future_dates, future_prices, marker='o', linestyle='dashed', label="Predicted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)
