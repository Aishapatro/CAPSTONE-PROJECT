
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LSTM Power Prediction",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Household Power Consumption Prediction")
st.markdown(
    "### Time Series Analysis using LSTM Deep Learning"
)

st.write(
    "Upload your household power consumption dataset, "
    "train an LSTM model, and predict Global Active Power."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Model Settings")

epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=1,
    max_value=100,
    value=20
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64, 70, 128],
    index=3
)

train_ratio = st.sidebar.slider(
    "Training Data Percentage",
    min_value=60,
    max_value=90,
    value=80
)

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload household_power_consumption.txt or CSV file",
    type=["txt", "csv"]
)

if uploaded_file is None:
    st.info(
        "Please upload your household power consumption dataset."
    )

    st.markdown("""
    **Expected dataset columns:**

    - Global_active_power
    - Global_reactive_power
    - Voltage
    - Global_intensity
    - Sub_metering_1
    - Sub_metering_2
    - Sub_metering_3
    """)

    st.stop()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:
    if uploaded_file.name.endswith(".txt"):
        df = pd.read_csv(
            uploaded_file,
            sep=";",
            low_memory=False,
            na_values=["?", "nan"]
        )
    else:
        df = pd.read_csv(
            uploaded_file,
            low_memory=False,
            na_values=["?", "nan"]
        )

    # Convert Date and Time into datetime index
    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )
        df.drop(
            columns=["Date", "Time"],
            inplace=True
        )

        df.set_index("datetime", inplace=True)

    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )
        df.set_index("datetime", inplace=True)

    elif "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["DateTime"],
            errors="coerce"
        )
        df.set_index("DateTime", inplace=True)

    else:
        st.error(
            "Dataset must contain Date and Time columns "
            "or a datetime column."
        )
        st.stop()

    df.index.name = "datetime"

    # Convert columns into numeric format
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df.dropna(axis=0, how="all", inplace=True)

    df.sort_index(inplace=True)

    # Required features from notebook
    required_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing required columns: {missing_columns}"
        )
        st.stop()

    df = df[required_columns]

    # Fill missing values
    df = df.interpolate(method="time")
    df = df.ffill().bfill()

    if df.empty or df.isnull().values.any():
        st.error("Dataset is empty or contains missing values.")
        st.stop()

    st.success("Dataset uploaded and cleaned successfully!")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header("2. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric(
    "Missing Values",
    int(df.isnull().sum().sum())
)

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Statistical Description")
st.dataframe(df.describe(), use_container_width=True)

# --------------------------------------------------
# ORIGINAL DATA VISUALIZATION
# --------------------------------------------------

st.header("3. Data Visualization")

st.subheader("Global Active Power")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    df.index,
    df["Global_active_power"]
)
ax.set_xlabel("Date")
ax.set_ylabel("Global Active Power")
ax.set_title("Global Active Power Over Time")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# HOURLY RESAMPLING
# --------------------------------------------------

st.header("4. Hourly Resampling")

df_resample = df.resample("h").mean()

df_resample = df_resample.interpolate(
    method="time"
).ffill().bfill()

df_resample.dropna(inplace=True)

st.write("Hourly Resampled Dataset")
st.dataframe(
    df_resample.head(10),
    use_container_width=True
)

st.metric(
    "Hourly Records",
    f"{len(df_resample):,}"
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    df_resample.index,
    df_resample["Global_active_power"],
    color="red"
)

ax.set_title("Hourly Global Active Power")
ax.set_xlabel("Date")
ax.set_ylabel("Global Active Power")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# LSTM MODEL TRAINING
# --------------------------------------------------

st.header("5. LSTM Model Training")

st.write("""
**Model Architecture:**

- LSTM Layer: 100 neurons
- Dropout: 0.2
- Dense Layer: 1 neuron
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Time Step: Previous 1 hour
""")

if len(df_resample) < 100:
    st.warning(
        "Please upload at least 100 hourly records "
        "to train the model."
    )
    st.stop()

if st.button("🚀 Train LSTM Model", type="primary"):

    try:
        with st.spinner("Training LSTM model..."):

            # Convert data into NumPy array
            values = df_resample.values.astype("float32")

            # Chronological train-test split
            split_index = int(
                len(values) * train_ratio / 100
            )

            train_values = values[:split_index]
            test_values = values[split_index:]

            # Fit scaler only on training data
            scaler = MinMaxScaler(
                feature_range=(0, 1)
            )

            train_scaled = scaler.fit_transform(
                train_values
            )

            test_scaled = scaler.transform(
                test_values
            )

            # Create sequences using previous hour
            train_X = train_scaled[:-1, :]
            train_y = train_scaled[1:, 0]

            # First test prediction uses last training hour
            combined_scaled = np.vstack([
                train_scaled[-1:],
                test_scaled
            ])

            test_X = combined_scaled[:-1, :]
            test_y = combined_scaled[1:, 0]

            # Reshape into LSTM format
            train_X = train_X.reshape(
                train_X.shape[0], 1, train_X.shape[1]
            )

            test_X = test_X.reshape(
                test_X.shape[0], 1, test_X.shape[1]
            )

            # Build LSTM Model
            model = Sequential()

            model.add(
                LSTM(
                    100,
                    input_shape=(
                        train_X.shape[1],
                        train_X.shape[2]
                    )
                )
            )

            model.add(Dropout(0.2))
            model.add(Dense(1))

            model.compile(
                loss="mean_squared_error",
                optimizer="adam"
            )

            # Early stopping
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            # Train model
            history = model.fit(
                train_X,
                train_y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0,
                shuffle=False
            )

            # Predictions
            yhat = model.predict(
                test_X,
                verbose=0
            ).flatten()

            # Convert scaled values back to original values
            inv_yhat = (
                yhat * scaler.scale_[0]
                + scaler.min_[0]
            )

            inv_y = (
                test_y * scaler.scale_[0]
                + scaler.min_[0]
            )

            # Evaluation
            rmse = np.sqrt(
                mean_squared_error(inv_y, inv_yhat)
            )

            mae = mean_absolute_error(
                inv_y, inv_yhat
            )

            r2 = r2_score(
                inv_y, inv_yhat
            )

            # Store trained model and results
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["history"] = history.history
            st.session_state["actual"] = inv_y
            st.session_state["predicted"] = inv_yhat
            st.session_state["rmse"] = rmse
            st.session_state["mae"] = mae
            st.session_state["r2"] = r2
            st.session_state["test_index"] = (
                df_resample.index[split_index:]
            )

        st.success("LSTM Model Trained Successfully!")

    except Exception as e:
        st.error(f"Training error: {e}")

# --------------------------------------------------
# TRAINING RESULTS
# --------------------------------------------------

if "model" in st.session_state:

    st.header("6. Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "RMSE",
        f"{st.session_state['rmse']:.4f}"
    )

    col2.metric(
        "MAE",
        f"{st.session_state['mae']:.4f}"
    )

    col3.metric(
        "R2 Score",
        f"{st.session_state['r2']:.4f}"
    )

    # Loss graph
    st.subheader("Training and Validation Loss")

    history = st.session_state["history"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        history["loss"],
        label="Training Loss"
    )

    ax.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    ax.set_title("LSTM Model Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # Actual vs predicted
    st.header("7. Actual vs Predicted")

    actual = st.session_state["actual"]
    predicted = st.session_state["predicted"]
    test_index = st.session_state["test_index"]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        test_index,
        actual,
        label="Actual",
        marker="."
    )

    ax.plot(
        test_index,
        predicted,
        label="Predicted",
        color="red"
    )

    ax.set_title("Actual vs Predicted Global Active Power")
    ax.set_xlabel("Time")
    ax.set_ylabel("Global Active Power")
    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # First 200 hours
    st.subheader("First 200 Hours Prediction")

    fig, ax = plt.subplots(figsize=(12, 4))

    n = min(200, len(actual))

    ax.plot(
        range(n),
        actual[:n],
        label="Actual"
    )

    ax.plot(
        range(n),
        predicted[:n],
        label="Prediction"
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Global Active Power")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # Prediction table
    st.header("8. Prediction Results")

    results = pd.DataFrame({
        "DateTime": test_index,
        "Actual_Global_Active_Power": actual,
        "Predicted_Global_Active_Power": predicted
    })

    st.dataframe(
        results.head(100),
        use_container_width=True
    )

    # Download predictions
    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv_data,
        file_name="LSTM_Power_Predictions.csv",
        mime="text/csv"
    )

    # --------------------------------------------------
    # NEXT HOUR PREDICTION
    # --------------------------------------------------

    st.header("9. Next Hour Prediction")

    if st.button("🔮 Predict Next Hour"):

        # Last available hourly observation
        last_values = df_resample.iloc[-1:].values

        scaler = st.session_state["scaler"]

        last_scaled = scaler.transform(
            last_values
        )

        next_input = last_scaled.reshape(
            1, 1, len(df_resample.columns)
        )

        next_scaled = st.session_state[
            "model"
        ].predict(
            next_input,
            verbose=0
        )[0][0]

        next_prediction = (
            next_scaled - scaler.min_[0]
        ) / scaler.scale_[0]

        st.success(
            f"Predicted Global Active Power: "
            f"{next_prediction:.4f} kW"
        )

        st.caption(
            "Forecast for the hour immediately after the "
            "last available hourly observation."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "### ⚡ Household Power Consumption Prediction"
)

st.caption(
    "Developed using Python, Streamlit, "
    "Pandas, Scikit-learn and LSTM Deep Learning."
)
