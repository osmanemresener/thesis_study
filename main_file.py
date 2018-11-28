import pandas as pd
import seaborn as sns
from scipy import interpolate
from matplotlib.widgets import Slider
import math
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
def heat_resource_location(x1,y1,x2,y2,x3,y3,x4,y4):
    return  x1,y1,x2,y2,x3,y3,x4,y4



def mochup_data(maxtime, maxx, maxy, number_of_sensor):
    """
    :param maxtime: Function will generate number of "maxtime" unique time as "0,1,2,3,4,5...."
    :param maxx: X label starts from "0" and ends at "maxx". It represents x coordination.
    :param maxy: Y label starts from "0" and ends at "maxy". It represents y coordination.
    :param number_of_sensor: Both X label and Y label, "number_of_sensors" will be number of sensor in a single line.
    :return: According to given inputs, function will return a pandas dataframe.

    Columns are "time_stamp" : '0' to 'maxtime'
                "x_location" : '0' to 'maxx'
                "y_location" : '0' to 'maxy'
                "Temperature": Randomly created float data between 25.0 and 35.0
                "is_there_any_sensor" : It represents there is sensor as "True".
    """
    time = np.arange(0, maxtime).tolist()
    node1=np.random.normal(35,0.1,maxtime)
    node2=np.random.normal(30,0.1,maxtime)
    node3=np.random.normal(32,0.1,maxtime)
    node4=np.random.normal(27,0.1,maxtime)
    xlabel = np.linspace(0, maxx, number_of_sensor).tolist()
    ylabel = np.linspace(0, maxy, number_of_sensor).tolist()
    x1=random.choice(xlabel)
    y1=random.choice(ylabel)
    while True:
        x2=random.choice(xlabel)
        y2=random.choice(ylabel)
        if x2 != x1 and y2 != y1:
            break
    while True:
        x3=random.choice(xlabel)
        y3=random.choice(ylabel)
        if x3 != x1 and y3 != y1 and x3 != x2 and y3 != y2:
            break
    while True:
        x4=random.choice(xlabel)
        y4=random.choice(ylabel)
        if x4 != x1 and y4 != y1 and x4 != x2 and y4 != y2 and x4 != x3 and y4 != y3:
            break
    print("x1",x1,"y1",y1,"x2",x2,"y2",y2,"x3",x3,"y3",y3,"x4",x4,"y4",y4,)

    sensor_list = []
    df = pd.DataFrame(columns=['time_stamp', 'x_location', 'y_location', 'temperature', 'is_there_any_sensor'])
    for t in time:
        for x in xlabel:
            for y in ylabel:
                if x == x1 and y == y1:
                    sensor_list.append({'time_stamp': t, 'x_location': x, 'y_location': y,
                                    'temperature': node1[t], 'is_there_any_sensor': "True"})
                elif x == x2 and y == y2:
                    sensor_list.append({'time_stamp': t, 'x_location': x, 'y_location': y,
                                        'temperature': node2[t], 'is_there_any_sensor': "True"})
                elif x == x3 and y == y3:
                    sensor_list.append({'time_stamp': t, 'x_location': x, 'y_location': y,
                                        'temperature': node3[t], 'is_there_any_sensor': "True"})
                elif x == x4 and y == y4:
                    sensor_list.append({'time_stamp': t, 'x_location': x, 'y_location': y,
                                        'temperature': node4[t], 'is_there_any_sensor': "True"})
                else:
                    sensor_list.append({'time_stamp': t, 'x_location': x, 'y_location': y,
                                    'temperature': np.random.uniform(23.5, 32.5), 'is_there_any_sensor': "True"})
    temperature_datas = df.append(sensor_list)
    return temperature_datas
temperature_datas = mochup_data(100,300,300,11)
mochup_data(100,300,300,11)
def select_time(t=0):
    """
    :param t: 't' value determines which time you want to work with.
    :return: Function returns 't' filtered dataframe.
    Columns are "time_stamp" : '0' to 'maxtime' "time_stamp" filtered according to value of 't'
                "x_location" : '0' to 'maxx'
                "y_location" : '0' to 'maxy'
                "Temperature": Randomly created float data between 25.0 and 35.0
                "is_there_any_sensor" : It represents there is sensor as "True".
    """
    time_list = temperature_datas["time_stamp"].unique().tolist()
    temperature_data = temperature_datas[temperature_datas["time_stamp"] == time_list[t]]
    return temperature_data

def bilinear_interpolation(i=0, t=0):
    """
    :param This function takes 'i' as integer parameter. Between two sequential sensor node's location, function is splitting this distance to 'i' pieces.
            Funtion applies splitting both x and y label. Creating upscaled new labels. Based on new label calculating new values for each coordinates.
    :return: Function returns a pandas dataframe:
                Indexs     : y_locations
                Columns    : x_locations
                Data_array : interpolated temperature values
    """
    start_time = time.monotonic()

    temperature_data = select_time(t)
    x_old = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(),
                        (len(temperature_data["x_location"].unique().tolist())))
    y_old = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(),
                        (len(temperature_data["y_location"].unique().tolist())))
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    if i == 1:
        x_n_points = (len(temperature_data["x_location"].unique().tolist()))
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist()))
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)
    else:
        x_n_points = (len(temperature_data["x_location"].unique().tolist())) + i * (
                    (len(temperature_data["x_location"].unique().tolist())) - 1)
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist())) + i * (
                    (len(temperature_data["y_location"].unique().tolist())) - 1)
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)

    f = interpolate.interp2d(xx, yy, sensors_data_array, kind='linear')
    interpolated_array = f(x_new, y_new)
    interpolated_df = pd.DataFrame(data=interpolated_array,
                                   index=y_new,
                                   columns=x_new)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    return interpolated_df
def bicubic_interpolation(i=0, t=0):
    """
    :param This function takes 'i' as integer parameter. Between two sequential sensor node's location, function is splitting this distance to 'i' pieces.
            Funtion applies splitting both x and y label. Creating upscaled new labels. Based on new label calculating new values for each coordinates.
    :return: Function returns a pandas dataframe:
                Indexs     : y_locations
                Columns    : x_locations
                Data_array : interpolated temperature values
    """
    start_time = time.monotonic()

    temperature_data = select_time(t)
    x_old = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(),
                        (len(temperature_data["x_location"].unique().tolist())))
    y_old = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(),
                        (len(temperature_data["y_location"].unique().tolist())))
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    if i == 1:
        x_n_points = (len(temperature_data["x_location"].unique().tolist()))
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist()))
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)
    else:
        x_n_points = (len(temperature_data["x_location"].unique().tolist())) + i * (
                    (len(temperature_data["x_location"].unique().tolist())) - 1)
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist())) + i * (
                    (len(temperature_data["y_location"].unique().tolist())) - 1)
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)

    f = interpolate.interp2d(xx, yy, sensors_data_array, kind='cubic')
    interpolated_array = f(x_new, y_new)
    interpolated_df = pd.DataFrame(data=interpolated_array,
                                   index=y_new,
                                   columns=x_new)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    return interpolated_df

def heatmap_with_seaborn(i=1):
    """

    :param i: This function take 'i' as input and using at interpolatin(i) function to create dataframe which is splitted according to 'i' value.
    :return: This function returns seaborn based heatmap.
    """
    z = interpolation(i)
    plt.figure(figsize=(15, 15))
    plt.xlabel('x_location', size=15)
    plt.ylabel('y_location', size=15)
    plt.title('Temperature Data of Factory ', size=15),
    ax = sns.heatmap(z, annot=False, linewidths=0, square=True, cmap='RdBu_r', center=30)
    ax.invert_yaxis()
    return plt.show()

def heatmap_with_matplotlib(i, t, f):
    """

    :param i: This function takes 'i' as input and using at interpolatin(i, t) function to create dataframe which is splitted according to 'i' value.
    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :return: This function returns matplotlib based heatmap. Which has time and interpolation slider on it.
    """
    xx = (f(i, t).columns.values.tolist())
    yy = (f(i, t).index.values.tolist())
    fig, ax = plt.subplots()
    img = plt.imread("overlay.png")

    im = ax.imshow(f(i, t).values, cmap="RdBu_r")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(xx)))
    ax.set_yticks(np.arange(len(yy)))
    ax.set_xticklabels(xx)
    ax.set_yticklabels(yy)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title("Temperature Data of Factory")
    fig.tight_layout()
    interpolation_slider = plt.axes((0.23, 0.02, 0.56, 0.03))
    int_slider_dep = Slider(interpolation_slider, 'Scaling Value', 0, 10, valinit=0)
    time_slider = plt.axes((0.23, 0.06, 0.56, 0.03))
    time_slider_dep = Slider(time_slider, 'Time Value', 0, 10, valinit=0)
    ax.imshow(img, origin="lower", extent=[-0.15, 10.17, -0.2, 10.12],alpha=0.2)

    def update_depth(val):
        t = int(round(time_slider_dep.val))
        i = int(round(int_slider_dep.val))
        im.set_data(f(i, t).values)

    int_slider_dep.on_changed(update_depth)
    time_slider_dep.on_changed(update_depth)
    return plt.show()

def ddd_plot(i,t):
    """

    :param i: This function takes 'i' as input and using at interpolatin(i, t) function to create dataframe which is splitted according to 'i' value.
    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :return: This function returns 3d plot of data.
    """
    xx = (interpolation(i, t).columns.values.tolist())
    yy = (interpolation(i, t).index.values.tolist())
    z = interpolation(i, t).values
    x,y = np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm,
                           rstride=1, cstride=1)
    ax.view_init(20, -120)
    ax.set_xlabel('xx')
    ax.set_ylabel('yy')
    ax.set_zlabel('zz')

    return plt.show()

def bilinear_interpolation(i=0,t=0):
    start_time = time.monotonic()
    temperature_data = select_time(t)
    x_old = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(),
                        (len(temperature_data["x_location"].unique().tolist())))
    y_old = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(),
                        (len(temperature_data["y_location"].unique().tolist())))
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    if i == 0:
        x_new = x_old
        y_new = y_old
    else:
        x_n_points = (len(temperature_data["x_location"].unique().tolist())) + (i) * (
                    (len(temperature_data["x_location"].unique().tolist())) - 1)
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist())) + (i) * (
                    (len(temperature_data["y_location"].unique().tolist())) - 1)
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)
    interpolated_array=[]
    x_seq_distance = (temperature_data["x_location"].max() - temperature_data["x_location"].min()) / (
            (len(temperature_data["x_location"].unique().tolist())) - 1)
    y_seq_distance = (temperature_data["y_location"].max() - temperature_data["y_location"].min()) / (
            (len(temperature_data["y_location"].unique().tolist())) - 1)
    for x in range(len(x_new)):
        temp_list = []
        for y in range(len(y_new)):

            x_cor = (x_new[x] / x_seq_distance)
            y_cor = (y_new[y] / y_seq_distance)

            x_predictor = int(x_new[x])
            y_predictor = int(y_new[y])

            x_above_index = int(math.ceil(x_cor))
            x_above_location = x_old[x_above_index]
            x_below_index = int(math.trunc(x_cor))
            x_below_location = x_old[x_below_index]

            y_above_index = int(math.ceil(y_cor))
            y_above_location = y_old[y_above_index]
            y_below_index = int(math.trunc(y_cor))
            y_below_location = y_old[y_below_index]

            if (x_new[x] in x_old) and (y_new[y] in y_old):
                temp_list.append(sensors_data_array[int(x_cor),int(y_cor)])
            elif (x_new[x] in x_old):
                p = ((y_above_location-y_predictor)/(y_above_location-y_below_location)*sensors_data_array[x_below_index,y_below_index])+((y_predictor-y_below_location)/(y_above_location-y_below_location))*sensors_data_array[x_below_index,y_above_index]
                temp_list.append(p)
            elif (y_new[y] in y_old):
                p = ((x_above_location - x_predictor) / (x_above_location - x_below_location) * sensors_data_array[x_below_index, y_below_index]) + ((x_predictor - x_below_location) / (x_above_location - x_below_location))*sensors_data_array[x_above_index, y_below_index]
                temp_list.append(p)
            else:
                r1=((x_above_location-x_predictor)/(x_above_location-x_below_location))*sensors_data_array[x_below_index,y_below_index]+((x_predictor-x_below_location)/(x_above_location-x_below_location))*sensors_data_array[x_above_index,y_below_index]
                r2=((x_above_location-x_predictor)/(x_above_location-x_below_location))*sensors_data_array[x_below_index,y_above_index]+((x_predictor-x_below_location)/(x_above_location-x_below_location))*sensors_data_array[x_above_index,y_above_index]
                p = ((y_above_location-y_predictor)/(y_above_location-y_below_location))*r1 + ((y_predictor-y_below_location)/(y_above_location-y_below_location))*r2
                temp_list.append(p)
        interpolated_array.append(temp_list)
    interpolated_df = pd.DataFrame(data=interpolated_array,
                                       index=y_new,
                                       columns=x_new)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    return (interpolated_df)

def linear_regression(i,t):
    temperature_data=select_time(t)
    predictors = temperature_data[['x_location',
            'y_location']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    temperature_prediction = temperature_data['temperature']
    x_old = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(),
                        (len(temperature_data["x_location"].unique().tolist())))
    y_old = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(),
                        (len(temperature_data["y_location"].unique().tolist())))
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    x_seq_distance = (temperature_data["x_location"].max() - temperature_data["x_location"].min()) / (
            (len(temperature_data["x_location"].unique().tolist())) - 1)
    y_seq_distance = (temperature_data["y_location"].max() - temperature_data["y_location"].min()) / (
            (len(temperature_data["y_location"].unique().tolist())) - 1)
    if i == 0:
        x_new = x_old
        y_new = y_old
    else:
        x_n_points = (len(temperature_data["x_location"].unique().tolist())) + (i) * (
                    (len(temperature_data["x_location"].unique().tolist())) - 1)
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist())) + (i) * (
                    (len(temperature_data["y_location"].unique().tolist())) - 1)
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)
    regr = linear_model.LinearRegression()
    regr.fit(predictors, temperature_prediction)
    lin_reg_array=[]
    for x in range(len(x_new)):
        temp_list = []
        for y in range(len(y_new)):
            x_cor = (x_new[x] / x_seq_distance)
            y_cor = (y_new[y] / y_seq_distance)
            if (x_new[x] in x_old) and (y_new[y] in y_old):
                temp_list.append(sensors_data_array[int(x_cor),int(y_cor)])
            else:
                temperature=regr.predict([[x_new[x],y_new[y]]])
                temp_list.append(temperature.item())
        lin_reg_array.append(temp_list)
    lin_reg_df = pd.DataFrame(data=lin_reg_array,
                                       index=y_new,
                                       columns=x_new)
    return lin_reg_df

def poly_regression(i,t):
    temperature_data=select_time(t)
    X = temperature_data[['x_location',
            'y_location']].values
    Z = temperature_data['temperature'].values
    x_old = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(),
                        (len(temperature_data["x_location"].unique().tolist())))
    y_old = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(),
                        (len(temperature_data["y_location"].unique().tolist())))
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    x_seq_distance = (temperature_data["x_location"].max() - temperature_data["x_location"].min()) / (
            (len(temperature_data["x_location"].unique().tolist())) - 1)
    y_seq_distance = (temperature_data["y_location"].max() - temperature_data["y_location"].min()) / (
            (len(temperature_data["y_location"].unique().tolist())) - 1)
    if i == 0:
        x_new = x_old
        y_new = y_old
    else:
        x_n_points = (len(temperature_data["x_location"].unique().tolist())) + (i) * (
                    (len(temperature_data["x_location"].unique().tolist())) - 1)
        x_new = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(), x_n_points)
        y_n_points = (len(temperature_data["y_location"].unique().tolist())) + (i) * (
                    (len(temperature_data["y_location"].unique().tolist())) - 1)
        y_new = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(), y_n_points)
    def grid_based_prediction():
        poly = PolynomialFeatures(degree=2)
        X_ = poly.fit_transform(X)
        clf = linear_model.LinearRegression()
        clf.fit(X_, Z)
        predict_x, predict_y = np.meshgrid(x_new,y_new)
        predict_z = np.concatenate((predict_x.reshape(-1, 1),
                                    predict_y.reshape(-1, 1)),
                                   axis=1)
        predict_x_ = poly.fit_transform(predict_z)
        predict_data = clf.predict(predict_x_)
        def threed():
            fig = plt.figure(figsize=(16, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            surf = ax1.plot_surface(predict_x, predict_y, predict_data.reshape(predict_x.shape),
                                    rstride=1, cstride=1, cmap=cm.jet, alpha=0.5)
            ax1.scatter(X[:, 0], X[:, 1], Z, c='b', marker='o')
            ax1.set_xlim((300, 0))
            ax1.set_ylim((0, 300))
            fig.colorbar(surf, ax=ax1)
        poly_reg_df = pd.DataFrame(data=predict_data.reshape(predict_x.shape),
                                       index=y_new,
                                       columns=x_new)
        return poly_reg_df
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=5, include_bias=False)),
        ('linreg', LinearRegression(normalize=True))
    ])
    pipeline.fit(X, Z)
    poly_reg_array = []

    for x in range(len(x_new)):
        temp_list = []
        for y in range(len(y_new)):
            x_cor = (x_new[x] / x_seq_distance)
            y_cor = (y_new[y] / y_seq_distance)
            if (x_new[x] in x_old) and (y_new[y] in y_old):
                temp_list.append(sensors_data_array[int(x_cor), int(y_cor)])
            else:
                temperature = pipeline.predict(np.array([[x_new[x], y_new[y]]]).reshape(1,-1))
                temp_list.append(temperature.item())
        poly_reg_array.append(temp_list)
    poly_reg_df = pd.DataFrame(data=poly_reg_array,
                                       index=y_new,
                                       columns=x_new)
    return poly_reg_df

def gaussian_regression(i,t):
    def kernel(a, b, param):
        """
        RBF Kernel
        """
        return


def multiple_heatmap_with_matplotlib(i, t, f1, f2, f3, f4):
    """

    :param i: This function takes 'i' as input and using at interpolatin(i, t) function to create dataframe which is splitted according to 'i' value.
    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :return: This function returns matplotlib based heatmap. Which has time and interpolation slider on it.
    """
    xx = (f1(i, t).columns.values.tolist())
    yy = (f1(i, t).index.values.tolist())
    img = plt.imread("overlay.png")
    fig=plt.figure()

    ### Figure 1 ###
    ax1=fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(f1(i, t).values, cmap="RdBu_r")
    ax1.invert_yaxis()
    ax1.set_xticks(np.arange(len(xx)))
    ax1.set_yticks(np.arange(len(yy)))
    ax1.set_xticklabels(xx)
    ax1.set_yticklabels(yy)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax1.set_title("{} - Temperature Data of Factory".format(f1.__name__))
    ax1.imshow(img, origin="lower", extent=[-0.15, 10.17, -0.2, 10.12],alpha=0.2)

    ### Figure 2 ###
    ax2=fig.add_subplot(2,2,2)
    im2 = ax2.imshow(f2(i, t).values, cmap="RdBu_r")
    ax2.invert_yaxis()
    ax2.set_xticks(np.arange(len(xx)))
    ax2.set_yticks(np.arange(len(yy)))
    ax2.set_xticklabels(xx)
    ax2.set_yticklabels(yy)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax2.set_title("{} - Temperature Data of Factory".format(f2.__name__))
    ax2.imshow(img, origin="lower", extent=[-0.15, 10.17, -0.2, 10.12], alpha=0.2)

    ### Figure 3 ###
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(f3(i, t).values, cmap="RdBu_r")
    ax3.invert_yaxis()
    ax3.set_xticks(np.arange(len(xx)))
    ax3.set_yticks(np.arange(len(yy)))
    ax3.set_xticklabels(xx)
    ax3.set_yticklabels(yy)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax3.set_title("{} - Temperature Data of Factory".format(f3.__name__))
    ax3.imshow(img, origin="lower", extent=[-0.15, 10.17, -0.2, 10.12], alpha=0.2)

    ### Figure 4 ###
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(f4(i, t).values, cmap="RdBu_r")
    ax4.invert_yaxis()
    ax4.set_xticks(np.arange(len(xx)))
    ax4.set_yticks(np.arange(len(yy)))
    ax4.set_xticklabels(xx)
    ax4.set_yticklabels(yy)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax4.set_title("{} - Temperature Data of Factory".format(f4.__name__))
    ax4.imshow(img, origin="lower", extent=[-0.15, 10.17, -0.2, 10.12], alpha=0.2)

    interpolation_slider = plt.axes((0.23, 0.02, 0.56, 0.03))
    int_slider_dep = Slider(interpolation_slider, 'Scaling Value', 0, 10, valinit=0)
    time_slider = plt.axes((0.23, 0.06, 0.56, 0.03))
    time_slider_dep = Slider(time_slider, 'Time Value', 0, 10, valinit=0)

    def update_depth(val):
        t = int(round(time_slider_dep.val))
        i = int(round(int_slider_dep.val))
        im1.set_data(f1(i, t).values)
        im2.set_data(f2(i, t).values)
        im3.set_data(f3(i, t).values)
        im4.set_data(f4(i, t).values)

    int_slider_dep.on_changed(update_depth)
    time_slider_dep.on_changed(update_depth)
    return plt.show()

multiple_heatmap_with_matplotlib(0,0,bilinear_interpolation,bicubic_interpolation,linear_regression,poly_regression)