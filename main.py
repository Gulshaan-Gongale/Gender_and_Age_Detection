import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import pandas_datareader as pdr
import tensorflow as tf




start_date = "2010-01-01"
end_date = "2019-12-31"


st.title("Stock Price Predictor")
user_input = st.text_input("Enter Stock Ticker", "AAPL")
dfdata = pdr.DataReader(user_input, "yahoo", start_date, end_date)


st.subheader("Data from 2010 - 2019")
st.write(dfdata.describe())

#visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize= (8,7))
plt.plot(dfdata.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100ma")
ma100 = dfdata.Close.rolling(100).mean()
fig = plt.figure(figsize= (8,7))
plt.plot(ma100)
plt,plot(dfdata.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100ma & 200ma")
ma100 = dfdata.Close.rolling(100).mean()
ma200 = dfdata.Close.rolling(200).mean()
fig = plt.figure(figsize= (8,7))
plt.plot(ma100, "r")
plt.plot(ma200, "g")
plt,plot(dfdata.Close, "b")
st.pyplot(fig)

#spliting data into training and testing
data_training = pd.DataFrame(dfdata["Close"][0:int(len(dfdata)*0.70)])
data_testing = pd.DataFrame(dfdata["Close"][int(len(dfdata)*0.70): int(len(dfdata))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
data_training_sc = scaler.fit_transform(data_training)

#spliting data into x_train and y_train
x_train = []
y_train = []

for i in range(100, data_training_sc.shape[0]):
    x_train.append(data_training_sc[i-100: i])
    y_train.append(data_training_sc[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#load my model
model = load_model("keras_model.h5")
past_100_days = data_training.tail(100)
final_data = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_data)

#spliting data into x_train and y_train
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)    

## MAKING PREDICTIONS 
y_predict = model.predict(x_test)
scaler.scale_

scale_factor = 1/0.02123255
y_predict = y_predict * scale_factor
y_test = y_test * scale_factor