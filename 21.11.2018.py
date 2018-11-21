import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import interpolate
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import math
import time
from datetime import timedelta



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
    xlabel = np.linspace(0, maxx, number_of_sensor).tolist()
    ylabel = np.linspace(0, maxy, number_of_sensor).tolist()
    sensor_list = []
    df = pd.DataFrame(columns=['time_stamp', 'x_location', 'y_location', 'temperature', 'is_there_any_sensor'])
    for t in time:
        for x in xlabel:
            for y in ylabel:
                sensor_list.append({'time_stamp': t, 'x_location': x, 'y_location': y,
                                    'temperature': np.random.uniform(25.0, 35.0), 'is_there_any_sensor': "True"})
    temperature_datas = df.append(sensor_list)
    return temperature_datas
temperature_datas = mochup_data(100,300,300,11)

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


# Splitting between two sensors distances with user's input value and creating more point get better resolotion
def interpolation(i=1, t=0):
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

# interactive_plot=interactive(heatmap_with_seaborn,i=(1,10,1))
# interactive_plot

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
    int_slider_dep = Slider(interpolation_slider, 'Interpolation Value', 0, 20, valinit=0)
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
#heatmap_with_matplotlib(1, 0)

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

def prediction_based_on_regression(t=0):
    """

    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :return: OLS based model results.
    """
    temperature_data = select_time(t)
    model = ols("temperature ~ x_location + y_location", temperature_data).fit()
    print(model.summary())

    print("\nRetrieving manually the parameter estimates:")
    print(model._results.params)
    anova_results = anova_lm(model)

    print('\nANOVA results')
    print(anova_results)

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
    for x in range(len(x_new)):
        temp_list = []
        for y in range(len(y_new)):
            x_seq_distance = (temperature_data["x_location"].max() - temperature_data["x_location"].min()) / (
                        (len(temperature_data["x_location"].unique().tolist())) - 1)
            y_seq_distance = (temperature_data["y_location"].max() - temperature_data["y_location"].min()) / (
                        (len(temperature_data["y_location"].unique().tolist())) - 1)
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
heatmap_with_matplotlib(0, 0, interpolation)
print(bilinear_interpolation(2))
print(interpolation(2))



