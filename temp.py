import pandas as pd 
import matplotlib.pyplot as plt 
import datetime
from datetime import date, time
import yfinance as yf
import streamlit as st 
import numpy as np
from yahoofinancials import YahooFinancials
import pandas_ta as ta
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report,confusion_matrix 



classifier= RandomForestClassifier(n_estimators= 15, criterion="entropy")  
df = yf.download('ABNB',
                 start='2000-01-01',
                 end=date.today(),
                 progress=False,)
df['RSI(2)'] = ta.rsi(df['Close'],length=2)
df['RSI(7)'] = ta.rsi(df['Close'],length=7)
df['RSI(14)'] = ta.rsi(df['Close'],length=14)
df['CCI(30)'] = ta.cci(close=df['Close'],length=30,high=df['High'],low=df['Low'])
df['CCI(50)'] = ta.cci(close=df['Close'],length=50,high=df['High'],low=df['Low'])
df['CCI(100)'] = ta.cci(close=df['Close'],length=100,high=df['High'],low=df['Low'])
df['LABEL'] = np.where( df['Open'].shift(-2).gt(df['Open'].shift(-1)),"1","0")
df=df.dropna()
X = df[df.columns[6:-1]].values 
y = df['LABEL'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier.fit(X_train,y_train) 

predict_train = classifier.predict(X_train) 
predict_test = classifier.predict(X_test)
print(' Train Data Accuracy ')
print(classification_report(y_train,predict_train)) 

print( ' Testing Data Accuracy ' )
print( classification_report(y_test,predict_test) )

df['Prediction'] = np.append(predict_train,predict_test)
prediction = df.iloc[-1]['Prediction'] 
if prediction=="1":
  print("Today's return forecast: UP")
else: 
  print("Today's return forecast: DOWN")
  
print(pd.DataFrame(X))
print(df)