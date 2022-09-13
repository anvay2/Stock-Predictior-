from statistics import mode
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler as ms
import matplotlib.pyplot as p
import pandas_datareader as data 
from keras.layers import Dense, Dropout, LSTM 
from keras.models import Sequential 

startDate ='2005-05-01'
endDate ='2022-08-30'

pull = data.DataReader('GOOG', 'yahoo', startDate, endDate)
pull.head()
pull.tail()

pull = pull.reset_index()
pull.head()

pull = pull.drop(['Date', 'Adj Close'], axis =1)
pull.head()

avgOpen = pull.Open.rolling(500).mean()

p.figure(figsize = (12,6))
p.plot(pull.Open)

p.plot(avgOpen, color='red') 

print(pull.shape) #(4364, 5)

dataT = pd.DataFrame(pull['Open'][0:int(len(pull)*0.6)])
dataTest = pd.DataFrame(pull['Open'][int(len(pull)*0.4): int(len(pull))])

print(dataT.shape) #(2618, 1)
print(dataTest.shape) #(1746, 1)

scaler = ms(feature_range=(0,1))
dataT2 = scaler.fit_transform(dataT)
print(dataT2) #[[1.78311696e-04] [0.00000000e+00] [4.79686797e-03] ...[9.37711351e-01] [9.24466101e-01] [9.15606095e-01]]

xTrain = []
yTrain = []

for i in range(500, dataT2.shape[0]):
    xTrain.append(dataT2[i-500: i])
    yTrain.append(dataT2[i, 0])

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)


#layers
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences=True, input_shape= (xTrain.shape[1], 1)))
model.add(Dropout(0.2))

model = Sequential()
model.add(LSTM(units = 75, activation = 'relu', return_sequences=True))
model.add(Dropout(0.3))

model = Sequential()
model.add(LSTM(units = 100, activation = 'relu', return_sequences=True))
model.add(Dropout(0.4))

model = Sequential()
model.add(LSTM(units = 125, activation = 'relu'))
model.add(Dropout(0.5))


model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, epochs=50)

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

print(scaler.scale_)

scaleFactor = 1/0.0076607
yP = yP*scaleFactor
yTest = yTest*scaleFactor

p.figure(figsize=(12,6))
p.plot(yTest, label='Original')
p.plot(yP, 'r', label='Predicted')
p.xlabel('Time')
p.ylabel('Price')
p.legend()

p.show()


model.save('kerasModel.h5')




