import yfinance as yf


def load_data(symbol, period="1y"):
    data = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return data

    # Handle newer yfinance MultiIndex columns
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    return data.dropna()