from predicting_functions import polynomial_regression_prediction, linear_regression_prediction, random_forest_regression,train_test_splits, neural_network
from plotting_functions import prediction_to_plot
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random

##############################################################################################
"""This scenario aims to estimate removed sensor nodes data from all of closest sensor node"""
##############################################################################################


#Reading created datas and converting into pandas dataframe
data = pd.read_csv('sensor_nodes_data_between_2017_2018_hourly.csv')
data['time_stamp'] = pd.to_datetime(data['time_stamp'])
data = data.drop('Unnamed: 0',axis=1)

#Choosing one sensor node to remove and selecting temperature data of this as a list
sensor_list = data['sensor_code'].unique().tolist()
removed_sensor_node = sensor_list[5]
labels_df = data[data['sensor_code'] == removed_sensor_node].reset_index(drop=True)
time_list = labels_df['time_stamp'].values

selected_node_temperature_list = labels_df['temperature'].values.tolist()
print("{} sensor node chosen for test value".format(str(removed_sensor_node)))
print(labels_df.describe())


#Plotting choosen sensor node's temperature data
fig1, ax1 = plt.subplots()
ax1.plot(time_list, selected_node_temperature_list, color="blue")

ax1.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
        title='Selected {} coded sensor node temperature between 2017 and 2018,'.format(removed_sensor_node))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.grid()
fig1.savefig("scenerio_1_randomly_selected_sensor_node.png")
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
        input_sensor_code = random.choice(get_closest_sensors(removed_sensor_node))
        print("{} sensor node chosen for input data".format(input_sensor_code))
        train_sensor_data = data[data['sensor_code'] == input_sensor_code].reset_index(drop=True)
        features_df = pd.DataFrame()
        features_df['input_{}'.format(input_sensor_code)] = train_sensor_data['temperature']
    else:
        input_sensor_list = get_closest_sensors(removed_sensor_node)
        features_df = pd.DataFrame()
        for node in input_sensor_list:
            print("{} chosen as a input_{}".format(node,node))
            train_sensor_data = data[data['sensor_code'] == node].reset_index(drop=True)
            features_df['input_{}'.format(node)] = train_sensor_data['temperature']

    return features_df

#Machine Learning Methods

train_x, train_y, test_x, test_y, train_time, test_time = train_test_splits(get_closest_sensors_data_as_features(all=True),labels_df,shuffle=False)


fig2, ax2 = plt.subplots()
ax2.plot(train_time, train_y,  color="blue")
ax2.plot(test_time, test_y,  color="blue",linestyle='dotted')
ax2.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
        title='Selected {} coded sensor node temperature in 2018'.format(removed_sensor_node))
plt.gca().legend(('Training Temperature Data of {} Coded Sensor Node (°C)'.format(removed_sensor_node), 'Testing Temperature Data of {} Coded Sensor Node (°C)'.format(removed_sensor_node)))
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

linear_regression_prediction(train_x,train_y,test_x,test_y)
prediction_to_plot(linear_regression_prediction, removed_sensor_node, train_x, train_y, test_x, test_y, train_time, test_time)

polynomial_regression_prediction(train_x,train_y,test_x,test_y)
prediction_to_plot(polynomial_regression_prediction, removed_sensor_node, train_x, train_y, test_x, test_y, train_time, test_time)

random_forest_regression(train_x,train_y,test_x,test_y)
prediction_to_plot(random_forest_regression, removed_sensor_node, train_x, train_y, test_x, test_y, train_time, test_time)

neural_network(train_x,train_y,test_x,test_y)
prediction_to_plot(neural_network, removed_sensor_node, train_x, train_y, test_x, test_y, train_time, test_time)