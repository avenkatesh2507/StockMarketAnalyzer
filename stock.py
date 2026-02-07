import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_close_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df.reset_index(inplace=True)
    return df[['Date','Close']]

def train_model(df):
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df[['Date_ordinal']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    last_day = df['Date'].max()
    next_day = last_day + pd.Timedelta(days=1)
    next_day_ordinal = [[next_day.toordinal()]]
    next_pred = model.predict(next_day_ordinal)[0]

    return model, next_pred

def plot_stock(df, model):
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    prediction = model.predict(df[['Date_ordinal']])

    plt.figure(figsize=(8,4))
    plt.plot(df['Date'], df['Close'], label='Actual Price')
    plt.plot(df['Date'], prediction, label='Predicted Price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    return plt
