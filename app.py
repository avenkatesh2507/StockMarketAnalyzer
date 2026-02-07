import streamlit as st
from stock import get_close_stock_data, train_model, plot_stock
import pandas as pd

st.set_page_config(page_title="Stock Market Analyzer", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #EBF4F6;
        color: #09637E;
        font-family: 'Arial', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        color: #09637E;
    }

    .stTextInput > div > input, .stSelectbox > div > div {
        background-color: #7AB2B2;
        color: #09637E;
        font-size: 18px;
    }

    div.stButton > button {
        background-color: #088395;
        color: #EBF4F6;
        font-size: 16px;
        font-weight: bold;
    }

    hr {
        border-top: 2px solid #09637E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stock Market Analyzer")
st.caption("Enter a stock ticker and analyze its price trend")

ticker = st.text_input("Stock Ticker", "AAPL")
period = st.selectbox("Select Time Period", ["6mo", "1y", "2y"], index=1)
analyze = st.button("Analyze Stock")

st.divider()

if analyze and ticker:
    try:
        df = get_close_stock_data(ticker)
        today = pd.Timestamp.now(tz=df['Date'].dt.tz)

        if period == "6mo":
            df = df[df['Date'] >= today - pd.DateOffset(months=6)]
        elif period == "1y":
            df = df[df['Date'] >= today - pd.DateOffset(years=1)]
        elif period == "2y":
            df = df[df['Date'] >= today - pd.DateOffset(years=2)]

        model, next_pred = train_model(df)

        last_close = df['Close'].iloc[-1]
        price_color = "#088395" if next_pred >= last_close else "#EB4F4F"

        st.markdown(
            f"<h2 style='color:{price_color}'>Predicted Next-Day Close: ${next_pred:.2f}</h2>",
            unsafe_allow_html=True
        )

        st.divider()
        st.subheader("Stock Price Trend")
        fig = plot_stock(df, model)
        st.pyplot(fig)

        st.divider()
        st.subheader("Recent Data")
        st.dataframe(df.tail())

    except Exception as e:
        st.error(f"Error analyzing stock: {e}")
