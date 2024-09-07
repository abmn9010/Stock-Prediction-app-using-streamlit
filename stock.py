#import libraries

import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#set title
app_name = "Stock Market Analysis"
st.title(app_name)
st.subheader("Analyze the stock market data")

#add image for stock market representation
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSFEN0tFtL6FF-jchdN7bURYCpQ9g815e9nFw&s" , width = 600)

#Take input from user of about start and end date

st.sidebar.header("Input Parameters")

start = st.sidebar.date_input("Start", datetime(2020, 1, 1))
end = st.sidebar.date_input("End", datetime(2020, 1, 30))

# adding ticker symbol
ticker_list = ["AAPL", "AMZN", "TSLA", "GOOG", "MSFT","AGL","TQQQ","HWM","TSLA","NIO","TQQQ"]
ticker = st.sidebar.selectbox("Select the Company", ticker_list)
 
#fetych data from user input by using yfinance

data = yf.download(ticker, start=start, end=end)
#add Date as a column
data.insert(0, "Date", data.index, True)
data.reset_index(inplace=True, drop=True)

st.write('Data from',start, 'to', end)
st.write(data) 



#plot data
st.header("Data Visualization")
st.subheader("plot the data")
st.write('select your specific data from above table')
fig = px.line(data, x='Date', y=data.columns, title='closing price of the stock')
st.plotly_chart(fig)


# tO select column for analysis from box
column = st.selectbox("Select the Column seperately for Analysis", data.columns[1:])


#subset data
data1 = data[["Date", column]]
st.subheader("subset data")
st.write(data1)
  
 
#plot data
st.header("Subset Data Visualization ") 
fig = px.line(data1, x='Date', y=column, title='visualization of selected data')  
st.plotly_chart(fig)


#ADF test for data is stationary or not
#st.header("Test for Stationarity")
#st.write("if p-value is less than 0.05, the data is stationary")
#st.write("if p-value is greater than 0.05, the data is not stationary")
#st.write(adfuller(data1[column]) [1] < 0.05) 

st.write("Decomposition Data")
#decompose data
result = seasonal_decompose(data1[column], model='multiplicative', period=1)
fig = plt.figure()
fig = result.plot()
st.pyplot(fig)


#by plotly
st.plotly_chart(px.line(x=data1['Date'], y=result.trend, width=1000, height=500, title="trend", labels={"x": "Date", "y": "price"}))



#run the moodel

p = st.slider("Select the number of p", 0, 5, 2)
d = st.slider("Select the number of d", 0, 5, 1)
q = st.slider("Select the number of q", 0, 5, 2)
seasonal_order = st.number_input('Select the number of seasonal p', 0, 24, 12) 

#copy of data1
data2 = data1.copy()
# Initialize and fit the SARIMAX model
model = sm.tsa.statespace.SARIMAX(data1[column], order=(p, d, q), seasonal_order=(p, d, q, 12))
model = model.fit(disp=False)  # disp=False replaces disp=-1 for newer versions of statsmodels

# Display the results in Streamlit
st.write("Model Summary")
st.write(model.summary())


#forcasting

st.header("Forcasting")
forcast_period = st.number_input("Select the number of days for forcasting", value=10)

prediction = model.get_prediction(start = len(data1), end = len(data1) + forcast_period - 1)
prediction = prediction.predicted_mean
st.write(prediction)

#print  the  forcasting through daatframe

prediction.index = pd.date_range(start=end, periods=len(prediction), freq="D")
prediction = pd.DataFrame(prediction)

prediction.insert(0, "Date", prediction.index, True)
prediction.reset_index(inplace=True, drop=True)
st.write(prediction) 

st.write("## Actual Data")
st.write(data2)



#lets plot the data
fig = go.Figure()
#add actual data on the plot
fig.add_trace(go.Scatter(x=data2['Date'], y=data2[column], mode='lines', name='Actual Data'))
#add predicted data on the plot 
fig.add_trace(go.Scatter(x=prediction['Date'], y=prediction["predicted_mean"], mode='lines', name='Predicted Data'))
#set title and axis labels
fig.update_layout(title='Actual vs Predicted Data', xaxis_title='Date', yaxis_title='Price')
#disaplay the plot
st.plotly_chart(fig) 
