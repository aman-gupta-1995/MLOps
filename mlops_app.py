import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(pd.__version__)
# print(np.__version__)
# print(plt.__version__)
import streamlit as st
import yfinance as yf
import streamlit as st
from datetime import datetime as dt
import plotly.express as px

st.image("/Users/family/Downloads/Main Download/Our/Study/Github/MLOps/download.png", width = 100)

st.write("""
# Stock Price Analyzer

Shown are the *Apple* Stock's **Closing Prices** and **Volume Traded**.
""")

# st.write({"key":"value"})
# st.write("""
# a^2 + b^2 = c^2
# """)
# a = np.array([[1,2,3],[4,5,6]])
# st.write(a)

ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)
data = ticker.history(start = "2024-01-01", interval = "1mo")

# st.empty()
st.write("")
# st.markdown("<br>", unsafe_allow_html = True)
st.dataframe(data)

# st.write("Above space")
# # st.empty()  # Creates an empty space
# st.write("")
# st.write("Below space")

# st.markdown("Above space")
# st.markdown("<br>", unsafe_allow_html=True)  # Adds a line break
# st.markdown("Below space")
data.reset_index(inplace = True)
data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%d %b")

# st.line_chart(data=data, x= "Date", y= "Close",
#                y_label= "Closing Amount", 
#                use_container_width=True)

# Create a Plotly line chart
fig = px.line(data, x="Date", y="Close", labels={"Close": "Closing Amount"})

# Update the y-axis range to start from 150
fig.update_yaxes(range=[150, 250])

st.write("""
# Closing Price
""")
# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

fig1 = px.line(data, x="Date", y="Volume", labels={"Volume": "Volume Traded"})

# Update the y-axis range to start from 150
# fig.update_yaxes(range=[150, 250])

st.write("""
# Volume Traded
""")
# Display the chart in Streamlit
st.plotly_chart(fig1, use_container_width=True)


