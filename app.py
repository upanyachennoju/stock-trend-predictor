import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

MODEL_PATH = "stock_predictor_model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Model file not found! Training a new model...")
        return train_and_save_model()

    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except OSError:
        st.error("🚨 Error loading model! The file may be corrupted. Re-training...")
        return train_and_save_model()

def train_and_save_model():
    # Dummy data for training if no previous model exists
    dummy_data = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)
    X, y, scaler = preprocess_data(dummy_data)

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on dummy data
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Save the model
    model.save(MODEL_PATH)
    st.success("✅ Model trained and saved successfully!")
    return model

# Load Stock Data
def load_data(stock_symbol, start, end):
    df = yf.download(stock_symbol, start=start, end=end)
    return df[['Close']]

# Sentiment Analysis
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
    if len(data) < time_step:
        st.error(f"⚠️ Not enough data available! At least {time_step} records are needed.")
        return None, None

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
        model = load_trained_model()
        if model is None:
            st.error("⚠️ Model could not be loaded. Please try again.")
        else:
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
