import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Stock Data
def load_data(stock_symbol, start, end):
    df = yf.download(stock_symbol, start=start, end=end)
    return df[['Close']]

# Sentiment Analysis
def get_sentiment(stock_symbol):
    url = f'https://finance.yahoo.com/quote/{stock_symbol}/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.text for h in soup.find_all('h3') if h.text]
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Preprocess Data
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(data_scaled) - time_step):
        X.append(data_scaled[i:i+time_step])
        y.append(data_scaled[i+time_step][0])
    return np.array(X), np.array(y), scaler

# Build Model
def build_model():
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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

    with st.spinner("Training Model..."):
        X, y, scaler = preprocess_data(data.values)
        model = build_model()
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    with st.spinner(f"Predicting next {num_days} days..."):
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
