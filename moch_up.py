import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random

def mochup_data(maxx, maxy, number_of_sensor):

    #Reading CSV file
    df=pd.read_csv('braunschweig_outdoor_temperature_2017_2019.csv')
    braunschweig_outdoor_temperature=df['WERT'].values

    #Creating time series
    time_list=[]
    time = datetime.datetime(2017, 1, 1, 0, 0, 0)
    while True:
        tdelta=datetime.timedelta(hours=1)
        time_list.append(time)
        time += tdelta
        if time >= datetime.datetime(2019,1,1,0,0,0):
            break

    #Estimation of Indoor Temperature from Outdoor Temperature (2017-2019)
    temp_df = pd.DataFrame()
    temp_df['braunschweig_outdoor_temperature'] = braunschweig_outdoor_temperature
    max_outdoor_temperature = (temp_df['braunschweig_outdoor_temperature'].max())
    min_outdoor_temperature = (temp_df['braunschweig_outdoor_temperature'].min())
    max_avg_indoor_temperature=29
    min_avg_indoor_temperature=18
    temp_df['indoor_avg_temperature'] = max_avg_indoor_temperature-(max_outdoor_temperature-temp_df['braunschweig_outdoor_temperature'])*(max_avg_indoor_temperature-min_avg_indoor_temperature)/(max_outdoor_temperature-min_outdoor_temperature)
    braunschweig_outdoor_temperature = temp_df['braunschweig_outdoor_temperature'][:len(time_list)]

    #Creating Sensor Datas as dataframe (Each sensor has 1 hour delay)
    xlabel = np.linspace(0, maxx, number_of_sensor).tolist()
    ylabel = np.linspace(0, maxy, number_of_sensor).tolist()
    df = pd.DataFrame(
        columns=['time_stamp', 'sensor_code', 'x_location', 'y_location','temperature'])
    temp_list=[]
    i=0
    temp_nodes_temperature_list = temp_df['indoor_avg_temperature'].values.tolist()


    for x in xlabel:
        for y in ylabel:
            temp_nodes_temperature_list = np.asarray(temp_nodes_temperature_list)*np.random.uniform(0.9,1.1)

            temp_nodes_temperature_list= temp_nodes_temperature_list.tolist()
            del temp_nodes_temperature_list[0]

            for t in range(len(time_list)):
                temp_list.append({'time_stamp': time_list[t],'sensor_code': "S.{}.{}.".format(int(x), int(y)), 'x_location': x,
                                    'y_location': y,'temperature':temp_nodes_temperature_list[t]})
    final_df=df.append(temp_list)

    #Plotting Outdoor Temperature of Braunschweig between 2017-2018
    fig1, ax1 = plt.subplots()
    ax1.plot(time_list, braunschweig_outdoor_temperature, color="blue")
    ax1.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
           title='Braunschweig Outdoor Temperature Between 2017 and 2018 in °C',)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid()
    fig1.savefig("outdoor_temperature_between_2017_2018.png")
    plt.show()

    #Plotting Outdoor Temperature and Estimated Indoor Average Temperature
    fig2, ax2 = plt.subplots()
    ax2.plot(time_list, braunschweig_outdoor_temperature, color="blue")
    ax2.plot(time_list, temp_df['indoor_avg_temperature'].values.tolist()[:len(time_list)], color="crimson")
    ax2.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
            title='Indoor Average and Outdoor Temperature in °C', )
    plt.gca().legend(('Outdoor Temperature(°C)', 'Indoor Average Temperature (°C)'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.grid()
    fig2.savefig("indoor_and_outdoor_temperature_between_2017_2018.png")
    plt.show()


    #Choosing 4 random sensor nodes and plotting temperature datas of these nodes

    sensor_list=final_df['sensor_code'].unique().tolist()
    sensor_1 = sensor_list[random.choice(range(len(sensor_list)))]
    df1 = final_df[(final_df['sensor_code'] == sensor_1)]

    sensor_2 = sensor_list[random.choice(range(len(sensor_list)))]
    df2 = final_df[(final_df['sensor_code'] == sensor_2)]

    sensor_3 = sensor_list[random.choice(range(len(sensor_list)))]
    df3 = final_df[(final_df['sensor_code'] == sensor_3)]

    sensor_4 = sensor_list[random.choice(range(len(sensor_list)))]
    df4 = final_df[(final_df['sensor_code'] == sensor_4)]

    fig3, ax3 = plt.subplots()
    ax3.plot(time_list, df1['temperature'].values.tolist(), color="g")
    ax3.plot(time_list, df2['temperature'].values.tolist(), color="k")
    ax3.plot(time_list, df3['temperature'].values.tolist(), color="c")
    ax3.plot(time_list, df4['temperature'].values.tolist(), color="m")

    ax3.plot(time_list, temp_df['indoor_avg_temperature'].values.tolist()[:len(time_list)], color="crimson")
    ax3.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
            title='Temperature of Randomly Selected 4 Sensor Nodes (°C)')

    plt.gca().legend(('{} coded sensor nodes Temperature'.format(sensor_1),
                      '{} coded sensor nodes Temperature'.format(sensor_2),
                      '{} coded sensor nodes Temperature'.format(sensor_3),
                      '{} coded sensor nodes Temperature'.format(sensor_4),
                      'Indoor Average Temperature'))

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.grid()
    fig3.savefig("deneme.png")
    plt.show()
    return final_df.to_csv('sensor_nodes_data_between_2017_2018_hourly.csv')
mochup_data(60,60,3)