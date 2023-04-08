import pandas as pd
import datetime
import pickle
import streamlit as st
import yfinance as yf
st.header('This is a header')
ticker_symbol = 'AAPL'
ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(period = '1m',
                                start = '2019-01-01',
                                end = '2022-01-01')
st.dataframe(ticker_df)
st.line_chart(ticker_df.Volume)
with st.column(1):
   st.text("Hey")

with st.column(2):
   st.text("Ram")

with st.column(3):
   st.text("Ram!")
