import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_test_splits(features_df,labels_df,shuffle=False):
    df = pd.DataFrame()
    df = df.append(features_df)
    df['time_stamp'] = labels_df['time_stamp'].values
    df['labels']=labels_df['temperature'].values

    trainingSet, testSet = train_test_split(df, test_size=0.2,shuffle=shuffle)
    train_y = trainingSet['labels']
    train_time = trainingSet['time_stamp']
    train_x = trainingSet.drop(columns=['labels','time_stamp'])
    test_y = testSet['labels']
    test_time = testSet['time_stamp']
    test_x = testSet.drop(columns=['labels','time_stamp'])
    return train_x, train_y, test_x, test_y , train_time, test_time

def linear_regression_prediction(train_x, train_y, test_x, test_y):
    regr = linear_model.LinearRegression()
    model = regr.fit(train_x.values, train_y.values)
    prediction = model.predict(test_x)
    r2 = r2_score(test_y,prediction)
    rmse = math.sqrt(mean_squared_error(test_y,prediction))
    return prediction, r2, rmse

def polynomial_regression_prediction(train_x, train_y, test_x, test_y):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=4, include_bias=False)),
        ('linreg', LinearRegression(normalize=True))
    ])
    model = pipeline.fit(train_x, train_y)
    prediction = model.predict(test_x)
    r2 = r2_score(test_y, prediction)
    rmse = math.sqrt(mean_squared_error(test_y, prediction))
    return prediction, r2, rmse

def random_forest_regression(train_x, train_y, test_x, test_y):
    rf = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=20)
    model = rf.fit(train_x, np.ravel(train_y))
    prediction = model.predict(test_x)
    r2 = r2_score(test_y, prediction)
    rmse = math.sqrt(mean_squared_error(test_y, prediction))
    return prediction, r2, rmse

def sc_1_neural_network(train_x, train_y, test_x, test_y):

    scaler_train_x,scaler_train_y,scaler_test_x,scaler_test_y = MinMaxScaler(),MinMaxScaler(),MinMaxScaler(),MinMaxScaler()
    scaler_train_x.fit(train_x)
    scaler_test_x.fit(test_x)
    scaler_train_y.fit(np.array(train_y).reshape(len(train_y),-1))
    scaler_test_y.fit(np.array(test_y).reshape(len(test_y),-1))
    sc_train_x= scaler_train_x.transform(train_x)
    sc_train_y = scaler_train_y.transform(np.array(train_y).reshape(len(train_y),-1))
    sc_test_x = scaler_test_x.transform(test_x)
    sc_test_y = scaler_test_y.transform(np.array(test_y).reshape(len(test_y),-1))
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(sc_train_x, sc_train_y, validation_data=(sc_test_x, sc_test_y), epochs=150, batch_size=10)
    prediction = model.predict(test_x)
    r2 = r2_score(test_y,prediction)
    rmse = math.sqrt(mean_squared_error(test_y,prediction))
    return prediction, r2, rmse

def sc_2_neural_network(train_x, train_y, test_x, test_y):

    scaler_train_x,scaler_train_y,scaler_test_x,scaler_test_y = MinMaxScaler(),MinMaxScaler(),MinMaxScaler(),MinMaxScaler()
    scaler_train_x.fit(train_x)
    scaler_test_x.fit(test_x)
    scaler_train_y.fit(np.array(train_y).reshape(len(train_y),-1))
    scaler_test_y.fit(np.array(test_y).reshape(len(test_y),-1))
    sc_train_x= scaler_train_x.transform(train_x)
    sc_train_y = scaler_train_y.transform(np.array(train_y).reshape(len(train_y),-1))
    sc_test_x = scaler_test_x.transform(test_x)
    sc_test_y = scaler_test_y.transform(np.array(test_y).reshape(len(test_y),-1))
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(sc_train_x, sc_train_y, validation_data=(sc_test_x, sc_test_y), epochs=150, batch_size=10)
    prediction = model.predict(test_x)
    r2 = r2_score(test_y,prediction)
    rmse = math.sqrt(mean_squared_error(test_y,prediction))
    return prediction, r2, rmse