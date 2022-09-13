from __future__ import generator_stop
from audioop import avg
from cProfile import label
from json import load
from tkinter.filedialog import LoadFileDialog
import numpy as np 
import pandas as pd 
import streamlit as s
import matplotlib.pyplot as p
import pandas_datareader as data 
from tensorflow import keras
from keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler as ms
from pandas_datareader._utils import RemoteDataError


startDate ='2005-05-01'
endDate ='2022-08-30'

s.title('STOCK PREDICTER\n\nMade by Anvay')

userInput = s.text_input('Enter Ticker')
pull = data.DataReader(userInput, 'yahoo', startDate, endDate)
print(pull)

s.subheader('DATA (2005-2022)')
s.write(pull.describe())

s.subheader(f'Opening Price vs Time Chart | ID: {userInput}')
graph = p.figure(figsize =(12,6))
p.plot(pull.Open)
s.pyplot(graph)

s.subheader(f'Opening Price vs Time (500 Days mean avg) | ID: {userInput}')
avgOpen = pull.Open.rolling(500).mean()
graph = p.figure(figsize=(12,6))
p.plot(avgOpen)
p.plot(pull.Open)
p.xlabel('500 Days mean avg')
p.ylabel('Opening Price')
s.pyplot(graph)


dataT = pd.DataFrame(pull['Open'][0:int(len(pull)*0.6)])
dataTest = pd.DataFrame(pull['Open'][int(len(pull)*0.4): int(len(pull))])

scaler = ms(feature_range=(0,1))
dataT2 = scaler.fit_transform(dataT)

#predict

model = load_model('kerasModel.h5')

D500 = dataT.tail(500)
finalPull = D500.append(dataTest, ignore_index=True)

inputData = scaler.fit_transform(finalPull)

xTest = []
yTest = []

for j in range(500, inputData.shape[0]):
    xTest.append(inputData[j-500: j])
    yTest.append(inputData[j, 0])

xTest = np.array(xTest)
yTest = np.array(yTest)

yP = model.predict(xTest)

scaler = scaler.scale_
scaleFactor = 1/scaler[0]
yP = yP*scaleFactor
yTest = yTest*scaleFactor

s.subheader(f'Prediction vs Original | ID: {userInput}')
graphFinal = p.figure(figsize=(12,6))
p.plot(yTest, 'g', label='Original')
p.plot(yP, 'r', label='Predicted')
p.xlabel('Time')
p.ylabel('Price')
p.legend()
s.pyplot(graphFinal)