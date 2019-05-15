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


data = pd.read_csv('sensor_nodes_data_in_2018_hourly.csv')
data['time_stamp'] = pd.to_datetime(data['time_stamp'])


data = data.drop('Unnamed: 0',axis=1)

sensor_list = data['sensor_code'].unique().tolist()

test_sensor_node = sensor_list[25]
labels_df = data[data['sensor_code'] == test_sensor_node].reset_index(drop=True)
time_list = labels_df['time_stamp'].values

selected_node_temperature_list = labels_df['temperature'].values.tolist()
print("{} sensor node chosen for test value".format(str(test_sensor_node)))
print(labels_df.describe())



fig1, ax1 = plt.subplots()
ax1.plot(time_list, selected_node_temperature_list, color="blue")

ax1.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
        title='Selected {} coded sensor node temperature in 2018,'.format(test_sensor_node))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.grid()
fig1.savefig("selected_sensor_node.png")
plt.show()

def get_closest_sensors(node):
    test_temp_list =node.split('.')
    test_x_cor = int(test_temp_list[1])
    test_y_cor = int(test_temp_list[2])
    df = pd.DataFrame(
        columns=['sensor_code', 'x_location', 'y_location', 'distance'])
    temp_list=[]
    for n in sensor_list:
        train_temp_list = n.split('.')
        train_x_cor = int(train_temp_list[1])
        train_y_cor = int(train_temp_list[2])
        distance = math.sqrt((test_x_cor-train_x_cor)**2 + (test_y_cor-train_y_cor)**2)
        temp_list.append(
            {'sensor_code': n, 'x_location' : train_x_cor, 'y_location': train_y_cor,
             'distance': distance})
    final_df = df.append(temp_list).sort_values(by='distance').reset_index(drop=True)
    a=final_df['distance'].unique().tolist()
    final_df = final_df[final_df['distance']==a[1]].reset_index(drop=True )
    closest_sensors_list = final_df['sensor_code'].values.tolist()
    return closest_sensors_list

def get_closest_sensors_data_as_features(all=False):
    if not all:
        input_sensor_code = random.choice(get_closest_sensors(test_sensor_node))
        print("{} sensor node chosen for input data".format(input_sensor_code))
        train_sensor_data = data[data['sensor_code'] == input_sensor_code].reset_index(drop=True)
        features_df = pd.DataFrame()
        features_df['input_{}'.format(input_sensor_code)] = train_sensor_data['temperature']
    else:
        input_sensor_list = get_closest_sensors(test_sensor_node)
        features_df = pd.DataFrame()
        for node in input_sensor_list:
            print("{} chosen as a input_{}".format(node,node))
            train_sensor_data = data[data['sensor_code'] == node].reset_index(drop=True)
            features_df['input_{}'.format(node)] = train_sensor_data['temperature']

    return features_df


#Machine Learning Methods

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

train_x, train_y, test_x, test_y, train_time, test_time = train_test_splits(get_closest_sensors_data_as_features(all=True),labels_df,shuffle=False)

print(train_x)
fig2, ax2 = plt.subplots()
ax2.plot(train_time, train_y,  color="blue")
ax2.plot(test_time, test_y,  color="blue",linestyle='dotted')
ax2.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
        title='Selected {} coded sensor node temperature in 2018'.format(test_sensor_node))
plt.gca().legend(('Training Temperature Data of {} Coded Sensor Node (°C)'.format(test_sensor_node), 'Testing Temperature Data of {} Coded Sensor Node (°C)'.format(test_sensor_node)))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.grid()
fig2.savefig("train_test_split_validation_data.png")
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(train_time, train_x,  color="crimson")
ax3.plot(test_time, test_x,  color="crimson",linestyle='dotted')
ax3.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
        title='Selected {} coded sensor node temperature in 2018'.format(train_x.columns.values[0].split('_')[1]))
plt.gca().legend(('Training Temperature Data of {} Coded Sensor Node (°C)'.format(train_x.columns.values[0].split('_')[1]), 'Testing Temperature Data of {} Coded Sensor Node (°C)'.format(train_x.columns.values[0].split('_')[1])))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.grid()
fig3.savefig("train_test_split_1_input_data.png")
plt.show()




def linear_regression_prediction(train_x,train_y):
    regr = linear_model.LinearRegression()
    model = regr.fit(train_x.values, train_y.values)
    prediction = model.predict(test_x)
    r2 = r2_score(test_y,prediction)
    rmse = math.sqrt(mean_squared_error(test_y,prediction))

    fig4, ax4 = plt.subplots()
    ax4.plot(train_time, train_y, color="blue")
    ax4.plot(test_time, test_y, color="blue")
    ax4.plot(test_time, prediction, color="crimson",linestyle="dotted")
    ax4.plot([],[],alpha=0)
    ax4.plot([], [],alpha=0)

    ax4.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
            title='Linear Regression Predictions Result')
    plt.gca().legend(('Actual Temperature Data of {} Coded Sensor Node (°C)'.format(
        test_sensor_node),'__nolegend__' ,'Predicticted Temperature (°C)'.format(
        prediction),"R2 score: {}".format(r2),"RMSE score: {}".format(rmse)))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.grid()
    fig4.savefig("linear_regression_results.png")
    plt.show()
    return prediction



def polynomial_regression_prediction(train_x,train_y):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=4, include_bias=False)),
        ('linreg', LinearRegression(normalize=True))
    ])
    model = pipeline.fit(train_x, train_y)
    prediction = model.predict(test_x)
    r2 = r2_score(test_y, prediction)
    rmse = math.sqrt(mean_squared_error(test_y, prediction))

    fig4, ax4 = plt.subplots()
    ax4.plot(train_time, train_y, color="blue")
    ax4.plot(test_time, test_y, color="blue")
    ax4.plot(test_time, prediction, color="crimson", linestyle="dotted")
    ax4.plot([], [], alpha=0)
    ax4.plot([], [], alpha=0)

    ax4.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
            title='Polynomial Regression Predictions Result')
    plt.gca().legend(('Actual Temperature Data of {} Coded Sensor Node (°C)'.format(
        test_sensor_node), '__nolegend__', 'Predicticted Temperature (°C)'.format(
        prediction), "R2 score: {}".format(r2), "RMSE score: {}".format(rmse)))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.grid()
    fig4.savefig("polynomial_regression_results.png")
    plt.show()

    return prediction

def random_forest_regression(train_x,train_y):
    rf = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=20)
    model = rf.fit(train_x, np.ravel(train_y))
    prediction = model.predict(test_x)

    r2 = r2_score(test_y, prediction)
    rmse = math.sqrt(mean_squared_error(test_y, prediction))

    fig4, ax4 = plt.subplots()
    ax4.plot(train_time, train_y, color="blue")
    ax4.plot(test_time, test_y, color="blue")
    ax4.plot(test_time, prediction, color="crimson", linestyle="dotted")
    ax4.plot([], [], alpha=0)
    ax4.plot([], [], alpha=0)

    ax4.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
            title='Random Forest Regression Predictions Result')
    plt.gca().legend(('Actual Temperature Data of {} Coded Sensor Node (°C)'.format(
        test_sensor_node), '__nolegend__', 'Predicticted Temperature (°C)'.format(
        prediction), "R2 score: {}".format(r2), "RMSE score: {}".format(rmse)))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.grid()
    fig4.savefig("random_forest_regression_result.png")
    plt.show()

    return prediction

linear_regression_prediction(train_x,train_y)
polynomial_regression_prediction(train_x,train_y)
random_forest_regression(train_x,train_y)
#Preparing Input Datas


