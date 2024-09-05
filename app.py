import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import yfinance as yf
import datetime as dt
import altair as alt
import plotly.express as px
import datetime as dt

col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
  ticker_symbol = st.selectbox("Select Organization", options = ["AAPL", "MSFT", "GOOG"])
with col2:
  start_date = st.date_input("Start Date", dt.date(2024, 1, 1))
with col3:
  end_date = st.date_input("End Date", dt.date.today())
with col4:
  interval = st.selectbox("Interval", options = ["1d", "1wk", "1mo"])

period = {"1d": "Daily",
          "1wk": "Weekly",
          "1mo": "Monthly"}

ticker = yf.Ticker(ticker_symbol)
data = ticker.history(start = f"{start_date}", end = f"{end_date}",interval = interval)

data.reset_index(inplace = True)
data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%d %b")

ticker_name = ticker.info["longName"]

image_path = {"AAPL": "apple.png",
              "MSFT": "microsoft.jpeg",
              "GOOG": "google.png"}

st.image(image_path[ticker_symbol], width = 100)
st.write(f"""
# {ticker_name}
Shown are *{ticker_name}* stock's **Closing Price** and **Volume Traded**.
         """)
# st.write("")
st.write("\n #### Year: 2024")

st.markdown(f"""
    <h3 style='text-align: center; color: orange; font-weight: bold; font-size: 16px;'>
        From <span style='color: white;'>{start_date}</span> To <span style='color: white;'>{end_date}</span> 
        ({period[interval]})
    </h3>
    """, unsafe_allow_html=True)
st.dataframe(data)

chart = alt.Chart(data).mark_line().encode(
    x=alt.X('Date:T', title='Date', axis=alt.Axis(labelAngle=45)),  # Rotated x-axis labels
    y=alt.Y('Close:Q', title='Closing Amount'),
    tooltip=['Date:T', 'Close:Q']  # Adding hover effects
).properties(
    width='container',  # Adjusting to use container width
    height=400
).interactive()  # Make the chart interactive

st.write("""
# Closing Price
""")
# Display the chart in Streamlit
st.altair_chart(chart, use_container_width = True)

fig = px.line(data, x="Date", y="Volume", labels={"Volume": "Volume Traded"})

# Update the y-axis range to start from 150
# fig.update_yaxes(range=[150, 250])

st.write("""
# Volume Traded
""")
# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)