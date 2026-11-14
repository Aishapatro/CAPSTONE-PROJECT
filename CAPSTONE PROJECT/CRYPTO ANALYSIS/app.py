
import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model_rf = pickle.load(file)


# Function to predict based on user inputs
def predict_btc_price(input_data):
    prediction = model_rf.predict(input_data)
    return prediction[0]


def main():
    # Title of your web app
    st.title('Predict BTC Close Price')

    # Sidebar for user inputs
    st.sidebar.title('Input Features')

    # Inputs for USDT, BNB closing prices and volumes
    usdt_close = st.sidebar.number_input(
        'USDT Close Price',
        min_value=0.0,
        format="%.2f"
    )

    usdt_volume = st.sidebar.number_input(
        'USDT Volume',
        min_value=0.0,
        format="%.2f"
    )

    bnb_close = st.sidebar.number_input(
        'BNB Close Price',
        min_value=0.0,
        format="%.2f"
    )

    bnb_volume = st.sidebar.number_input(
        'BNB Volume',
        min_value=0.0,
        format="%.2f"
    )

    # Create input dataframe
    input_data = pd.DataFrame({
        'USDT_Close': [usdt_close],
        'USDT_Volume': [usdt_volume],
        'BNB_Close': [bnb_close],
        'BNB_Volume': [bnb_volume]
    })

    # Prediction button
    if st.button("Predict BTC Close Price"):
        try:
            prediction = predict_btc_price(input_data)
            st.success(f"Predicted BTC Close Price: {prediction:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == '__main__':
    main()
