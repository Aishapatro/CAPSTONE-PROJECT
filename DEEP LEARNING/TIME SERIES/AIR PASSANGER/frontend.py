
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.stattools import grangercausalitytests


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Air Passengers Time Series Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Air Passengers Time Series Analysis")
st.markdown(
    "### Time Series Data Analysis using Python"
)

st.markdown("""
This application performs time series analysis on the
Air Passengers dataset from 1949 to 1960.
""")

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

st.sidebar.info(
    "Upload your AirPassengers.csv dataset to begin."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv", "txt"]
)

st.sidebar.markdown("---")

analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    [
        "Dashboard",
        "Dataset Information",
        "Trend Analysis",
        "Seasonal Decomposition",
        "White Noise",
        "Detrending",
        "Deseasonalization",
        "Autocorrelation",
        "Lag Analysis",
        "Granger Causality"
    ]
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def prepare_data(data):

    data = data.copy()

    # Identify date column
    date_candidates = [
        "Date", "date", "Month", "month"
    ]

    date_col = next(
        (col for col in date_candidates if col in data.columns),
        None
    )

    # Identify passenger column
    passenger_candidates = [
        "Number of Passengers",
        "#Passengers",
        "Passengers",
        "passengers"
    ]

    passenger_col = next(
        (col for col in passenger_candidates if col in data.columns),
        None
    )

    if date_col is None:
        raise ValueError("Date or Month column not found.")

    if passenger_col is None:
        numeric_cols = data.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_cols:
            raise ValueError("Passenger column not found.")

        passenger_col = numeric_cols[0]

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce"
    )

    data[passenger_col] = pd.to_numeric(
        data[passenger_col],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_col, passenger_col]
    )

    data = data.sort_values(date_col)

    data = data.rename(
        columns={
            date_col: "Date",
            passenger_col: "Number of Passengers"
        }
    )

    data = data.set_index("Date")

    return data[["Number of Passengers"]]


if uploaded_file is not None:

    try:
        raw_df = load_data(uploaded_file)
        df = prepare_data(raw_df)

        st.sidebar.success("Dataset loaded successfully!")

    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()

else:
    st.info(
        "Please upload AirPassengers.csv using the sidebar."
    )
    st.stop()


# --------------------------------------------------
# COMMON VARIABLES
# --------------------------------------------------

series = df["Number of Passengers"]

st.sidebar.markdown("---")

period = st.sidebar.selectbox(
    "Seasonal Period",
    [12, 30],
    index=0,
    help="12 is suitable for monthly airline data. "
         "The original notebook uses period=30."
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if analysis_type == "Dashboard":

    st.header("📊 Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Average Passengers",
        f"{series.mean():.2f}"
    )

    col3.metric(
        "Maximum Passengers",
        int(series.max())
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Passenger Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Air Passengers Over Time")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        series,
        color="red",
        linewidth=2
    )

    ax.set_title("Air Passengers from 1949 to 1960")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Passengers")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

elif analysis_type == "Dataset Information":

    st.header("📋 Dataset Information")

    st.write("Dataset Shape:", df.shape)

    st.write("Column Names:")
    st.write(df.columns.tolist())

    st.subheader("Dataset Information")

    buffer = io.StringIO()
    df.info(buf=buffer)

    st.text(buffer.getvalue())

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().to_frame("Missing Values")
    )

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    st.subheader("Complete Dataset")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv().encode("utf-8")

    st.download_button(
        "Download Dataset",
        data=csv,
        file_name="AirPassengers_Processed.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# TREND ANALYSIS
# --------------------------------------------------

elif analysis_type == "Trend Analysis":

    st.header("📈 Trend and Seasonality Analysis")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        series,
        color="blue"
    )

    ax.set_title("Trend and Seasonality")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Passengers")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Two-Sided Time Series View")

    fig, ax = plt.subplots(figsize=(14, 5))

    x = df.index
    y = series.values

    ax.fill_between(
        x,
        y1=y,
        y2=-y,
        alpha=0.5,
        linewidth=2
    )

    ax.set_title("Air Passengers - Two Side View")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# SEASONAL DECOMPOSITION
# --------------------------------------------------

elif analysis_type == "Seasonal Decomposition":

    st.header("🔍 Seasonal Decomposition")

    if len(series) < 2 * period:
        st.warning(
            "At least two complete seasonal cycles are required."
        )
        st.stop()

    if (series <= 0).any():
        st.warning(
            "Multiplicative decomposition requires positive values."
        )
        st.stop()

    st.subheader("Multiplicative Decomposition")

    result_mul = seasonal_decompose(
        series,
        model="multiplicative",
        period=period
    )

    fig = result_mul.plot()
    fig.set_size_inches(14, 10)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Additive Decomposition")

    result_add = seasonal_decompose(
        series,
        model="additive",
        period=period
    )

    fig = result_add.plot()
    fig.set_size_inches(14, 10)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# WHITE NOISE
# --------------------------------------------------

elif analysis_type == "White Noise":

    st.header("📉 Random White Noise")

    rand_numbers = np.random.randn(1000)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        rand_numbers,
        color="blue"
    )

    ax.set_title("Random White Noise")
    ax.set_xlabel("Time")
    ax.set_ylabel("Random Values")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# DETRENDING
# --------------------------------------------------

elif analysis_type == "Detrending":

    st.header("📉 Time Series Detrending")

    st.subheader("1. Detrending using SciPy")

    detrended = signal.detrend(
        series.values
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        detrended
    )

    ax.set_title(
        "Air Passengers Detrended Using Least Squares Fit"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Detrended Values")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("2. Detrending using Seasonal Decomposition")

    if len(series) < 2 * period:
        st.warning("Not enough observations for decomposition.")
        st.stop()

    result = seasonal_decompose(
        series,
        model="multiplicative",
        period=period
    )

    detrended = series - result.trend

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        detrended
    )

    ax.set_title(
        "Detrending by Subtracting Trend Component"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Detrended Values")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# DESEASONALIZATION
# --------------------------------------------------

elif analysis_type == "Deseasonalization":

    st.header("📊 Air Passengers Deseasonalization")

    if len(series) < 2 * period:
        st.warning("Not enough observations for decomposition.")
        st.stop()

    result = seasonal_decompose(
        series,
        model="multiplicative",
        period=period
    )

    deseasonalized = series / result.seasonal

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        deseasonalized
    )

    ax.set_title("Air Passengers Deseasonalized")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deseasonalized Values")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# AUTOCORRELATION
# --------------------------------------------------

elif analysis_type == "Autocorrelation":

    st.header("📈 Autocorrelation Analysis")

    st.subheader("Autocorrelation Plot")

    fig, ax = plt.subplots(figsize=(12, 5))

    autocorrelation_plot(
        series.tolist(),
        ax=ax
    )

    ax.set_title("Air Passengers Autocorrelation")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("ACF and PACF")

    max_lags = min(50, len(series) // 2 - 1)

    if max_lags < 1:
        st.warning("Not enough observations.")
        st.stop()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 5)
    )

    plot_acf(
        series.tolist(),
        lags=max_lags,
        ax=axes[0]
    )

    plot_pacf(
        series.tolist(),
        lags=max_lags,
        ax=axes[1],
        method="ywm"
    )

    axes[0].set_title("ACF")
    axes[1].set_title("PACF")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# LAG ANALYSIS
# --------------------------------------------------

elif analysis_type == "Lag Analysis":

    st.header("📊 Lag Plot Analysis")

    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, 4)
    )

    for i, ax in enumerate(axes.flatten()):

        lag_plot(
            series,
            lag=i + 1,
            ax=ax
        )

        ax.set_title(f"Lag {i + 1}")

    fig.suptitle("Lag Plots of Air Passengers")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# GRANGER CAUSALITY
# --------------------------------------------------

elif analysis_type == "Granger Causality":

    st.header("🔬 Granger Causality Test")

    st.write(
        "Upload a separate dataset containing date and value columns. "
        "This analysis follows the separate dataset section in your notebook."
    )

    granger_file = st.file_uploader(
        "Upload Granger Dataset",
        type=["csv", "txt"],
        key="granger_upload"
    )

    if granger_file is not None:

        try:
            data = pd.read_csv(granger_file)

            if not {"date", "value"}.issubset(data.columns):
                st.error(
                    "Dataset must contain date and value columns."
                )
                st.stop()

            data["date"] = pd.to_datetime(
                data["date"],
                errors="coerce"
            )

            data["value"] = pd.to_numeric(
                data["value"],
                errors="coerce"
            )

            data = data.dropna(
                subset=["date", "value"]
            )

            data["month"] = data["date"].dt.month

            maxlag = st.slider(
                "Maximum Lag",
                min_value=1,
                max_value=5,
                value=2
            )

            if st.button("Run Granger Causality Test"):

                if len(data) <= maxlag + 2:
                    st.error("Not enough observations.")
                elif data["month"].nunique() <= 1:
                    st.error("Month column has no variation.")
                else:
                    results = grangercausalitytests(
                        data[["value", "month"]],
                        maxlag=maxlag,
                        verbose=False
                    )

                    st.success("Granger test completed.")

                    for lag, result in results.items():
                        st.write(f"Results for Lag {lag}")

                        f_stat, p_value, _, _ = result[
                            0
                        ]["ssr_ftest"]

                        st.write(
                            f"F-statistic: {f_stat:.4f}"
                        )

                        st.write(
                            f"P-value: {p_value:.6f}"
                        )

                        if p_value < 0.05:
                            st.write(
                                "Statistically significant result."
                            )
                        else:
                            st.write(
                                "No statistically significant result."
                            )

        except Exception as e:
            st.error(f"Granger test error: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center">
        <p>✈️ Air Passengers Time Series Analysis</p>
        <p>Developed using Python, Pandas, Matplotlib and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
