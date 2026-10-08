import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data import load_data
from src.indicators import add_indicators
from src.forecasting import train_model, forecast_future


st.set_page_config(
    page_title="Share Market Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Share Market Forecasting")
st.write(
    "An educational stock-market dashboard using technical indicators "
    "and machine-learning-based forecasting."
)

st.sidebar.header("Settings")

symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL"
).upper()

period = st.sidebar.selectbox(
    "Historical Period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=1,
    max_value=30,
    value=7
)

if st.sidebar.button("Load Forecast"):
    st.session_state["run_forecast"] = True

if "run_forecast" not in st.session_state:
    st.info("Enter a stock symbol and click 'Load Forecast' to begin.")
    st.stop()

try:
    with st.spinner("Downloading market data..."):
        data = load_data(symbol, period)

    if data.empty:
        st.error("No market data was found for this stock symbol.")
        st.stop()

    data = add_indicators(data)

    st.subheader(f"{symbol} Price Chart")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        )
    )

    if "SMA20" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA20"],
                mode="lines",
                name="SMA 20"
            )
        )

    if "SMA50" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA50"],
                mode="lines",
                name="SMA 50"
            )
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    latest_close = float(data["Close"].iloc[-1])

    col1.metric(
        "Latest Close",
        f"{latest_close:.2f}"
    )

    if "RSI" in data.columns:
        latest_rsi = float(data["RSI"].iloc[-1])
        col2.metric(
            "RSI",
            f"{latest_rsi:.2f}"
        )

    if "MACD" in data.columns:
        latest_macd = float(data["MACD"].iloc[-1])
        col3.metric(
            "MACD",
            f"{latest_macd:.2f}"
        )

    st.subheader("Technical Indicators")

    indicator_fig = go.Figure()

    if "RSI" in data.columns:
        indicator_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                mode="lines",
                name="RSI"
            )
        )

    indicator_fig.update_layout(
        height=350,
        yaxis_title="RSI"
    )

    st.plotly_chart(
        indicator_fig,
        use_container_width=True
    )

    st.subheader("Machine Learning Forecast")

    with st.spinner("Training forecasting model..."):
        model, X_test, y_test, mae, rmse = train_model(data)

    forecast = forecast_future(
        model,
        data,
        forecast_days
    )

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Test MAE",
        f"{mae:.2f}"
    )

    metric2.metric(
        "Test RMSE",
        f"{rmse:.2f}"
    )

    forecast_fig = go.Figure()

    forecast_fig.add_trace(
        go.Scatter(
            x=data.index[-60:],
            y=data["Close"].iloc[-60:],
            mode="lines",
            name="Historical Price"
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["Forecast"],
            mode="lines+markers",
            name="Forecast"
        )
    )

    forecast_fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

    st.subheader("Forecast Values")

    st.dataframe(
        forecast,
        use_container_width=True
    )

    csv = forecast.to_csv().encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        data=csv,
        file_name=f"{symbol}_forecast.csv",
        mime="text/csv"
    )

    st.warning(
        "Disclaimer: This application is for educational purposes only. "
        "Forecasts are estimates and should not be considered financial advice."
    )

except Exception as e:
    st.error("Something went wrong while loading the application.")
    st.exception(e)