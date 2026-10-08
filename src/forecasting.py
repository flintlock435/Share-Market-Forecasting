import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


WINDOW = 20


def train_model(data):
    prices = data["Close"].dropna().values

    X = []
    y = []

    for i in range(WINDOW, len(prices)):
        X.append(prices[i - WINDOW:i])
        y.append(prices[i])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    return model, X_test, y_test, mae, rmse


def forecast_future(model, data, days=7):
    prices = data["Close"].dropna().values

    window = list(prices[-WINDOW:])

    predictions = []

    for _ in range(days):
        X = np.array(window[-WINDOW:]).reshape(1, -1)

        prediction = model.predict(X)[0]

        predictions.append(prediction)

        window.append(prediction)

    last_date = data.index[-1]

    dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days
    )

    return pd.DataFrame(
        {"Forecast": predictions},
        index=dates
    )
