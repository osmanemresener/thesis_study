import pandas as pd
from scipy import interpolate
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF,ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#Creating random datas

def heat_resource_location(x1,y1,x2,y2,x3,y3,x4,y4):
    return  x1,y1,x2,y2,x3,y3,x4,y4

def mochup_data(maxtime, maxx, maxy, number_of_sensor):
    """
    :param maxtime: Function will generate number of "maxtime" unique time as "0,1,2,3,4,5...."
    :param maxx: X label starts from "0" and ends at "maxx". It represents x coordination.
    :param maxy: Y label starts from "0" and ends at "maxy". It represents y coordination.
    :param number_of_sensor: Both X label and Y label, "number_of_sensors" will be number of sensor in a single line.
    :return: According to given inputs, function will return a pandas dataframe.

    Columns are "sensor_code" : (Example : s.0.30)
                "time_stamp" : '0' to 'maxtime'
                "x_location" : '0' to 'maxx'
                "y_location" : '0' to 'maxy'
                "Temperature": Randomly created float data between 23.5 and 32.5. There is 4 heat source and their temperature has given below.
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
    df = pd.DataFrame(columns=['sensor_code','time_stamp', 'x_location', 'y_location', 'temperature', 'is_there_any_sensor'])
    for t in time:
        for x in xlabel:
            for y in ylabel:
                if x == x1 and y == y1:
                    sensor_list.append({'sensor_code': "S.{}.{}.".format(int(x),int(y)),'time_stamp': t, 'x_location': x, 'y_location': y,
                                    'temperature': node1[t], 'is_there_any_sensor': "True"})
                elif x == x2 and y == y2:
                    sensor_list.append({'sensor_code': "S.{}.{}.".format(int(x),int(y)),'time_stamp': t, 'x_location': x, 'y_location': y,
                                        'temperature': node2[t], 'is_there_any_sensor': "True"})
                elif x == x3 and y == y3:
                    sensor_list.append({'sensor_code': "S.{}.{}.".format(int(x),int(y)),'time_stamp': t, 'x_location': x, 'y_location': y,
                                        'temperature': node3[t], 'is_there_any_sensor': "True"})
                elif x == x4 and y == y4:
                    sensor_list.append({'sensor_code': "S.{}.{}.".format(int(x),int(y)),'time_stamp': t, 'x_location': x, 'y_location': y,
                                        'temperature': node4[t], 'is_there_any_sensor': "True"})
                else:
                    sensor_list.append({'sensor_code': "S.{}.{}.".format(int(x),int(y)),'time_stamp': t, 'x_location': x, 'y_location': y,
                                    'temperature': np.random.uniform(23.5, 32.5), 'is_there_any_sensor': "True"})
    temperature_datas = df.append(sensor_list)
    return temperature_datas
temperature_datas = mochup_data(100,300,300,11)
print(temperature_datas.head)
mochup_data(100,300,300,11)

#Functions for making grid and final array

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

def select_sensor(n=0):

    sensor_list = temperature_datas["sensor_code"].unique().tolist()
    sensor_code=sensor_list[n]
    return sensor_code

def create_grid(i,t):
    """
    :param i: This function gets 'i' as a parameter to create new grid.
    :param t: This function uses 't' at select_time(t) function to select and filter exactly this time.
    :return:
    temperature_data    : Function returns pandas dataframe which is filtered by given 't' value.
    x_old               : Function returns old x label.
    y_old               : Function returns old y label.
    x_new               : Function returns new x label.
    y_new               : Function returns new y label.
    """
    temperature_data = select_time(t)
    x_old = np.linspace(temperature_data["x_location"].min(), temperature_data["x_location"].max(),
                        (len(temperature_data["x_location"].unique().tolist())))
    y_old = np.linspace(temperature_data["y_location"].min(), temperature_data["y_location"].max(),
                        (len(temperature_data["y_location"].unique().tolist())))

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
    return temperature_data, x_old, y_old, x_new, y_new

def prediction_to_array(i,t,f,original=True):
    """

    :param i: Function takes 'i' as a input to use in create_grid(i,t) function.
    :param t: Function takes 't' as a input to use in create_grid(i,t) function.
    :param f: Function takes 'f' as a input to which model you want to use.
    :param original: Function takes "original" as a boolean input.  If original is "True", measured data will be same at their location.
                                                                    If original is "False" then value will taken from model.
    :return: Function returns an array.
    """
    if f==bilinear_interpolation or f==bicubic_interpolation:
        final_df=f(i,t)
    elif f==tf_regression:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        model=(f(i,t))

        temp_df= pd.DataFrame(dict(x_location=x_new,
                                   y_location=y_new))
        predict_input_func = tf.estimator.inputs.pandas_input_fn(
            x=temp_df,
            batch_size=10,
            num_epochs=1,
            shuffle=False)
        temperature_prediction=list(model.predict(predict_input_func))
        temperature_df=pd.DataFrame(dict(x_location=x_new,
                                         y_location=y_new,
                                         temperature=temperature_prediction))
        pivot_table=temperature_df.pivot('x_location','y_location','temperature')
        final_df = pd.DataFrame(data=pivot_table,
                                index=y_new,
                                columns=x_new)

    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
        sensors_data_array = pivot_table.values
        x_seq_distance = (temperature_data["x_location"].max() - temperature_data["x_location"].min()) / (
                (len(temperature_data["x_location"].unique().tolist())) - 1)
        y_seq_distance = (temperature_data["y_location"].max() - temperature_data["y_location"].min()) / (
                (len(temperature_data["y_location"].unique().tolist())) - 1)
        temp_array = []
        model = f(i,t)

        for x in range(len(x_new)):
            temp_list = []
            for y in range(len(y_new)):
                x_cor = (x_new[x] / x_seq_distance)
                y_cor = (y_new[y] / y_seq_distance)
                if original:
                    if (x_new[x] in x_old) and (y_new[y] in y_old):
                        temp_list.append(sensors_data_array[int(x_cor), int(y_cor)])
                    else:
                        temperature = model.predict(np.array([[x_new[x], y_new[y]]]).reshape(1, -1))
                        temp_list.append(temperature.item())
                else:
                    temperature = model.predict(np.array([[x_new[x], y_new[y]]]).reshape(1, -1))
                    temp_list.append(temperature.item())
            temp_array.append(temp_list)
        final_df = pd.DataFrame(data=temp_array,
                                    index=y_new,
                                    columns=x_new)
    return final_df

#Functions for predict unknown values

def bilinear_interpolation(i=0, t=0):
    """
    :param i,t : This function takes i and t as integer input to use in create_grid(i,t) function.
    :return: Function returns a pandas dataframe:
                Indexs     : y_locations
                Columns    : x_locations
                Data_array : Bilinear interpolated temperature values
    """
    temperature_data, x_old, y_old, x_new, y_new= create_grid(i,t)
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    f = interpolate.interp2d(xx, yy, sensors_data_array, kind='linear')
    interpolated_array = f(x_new, y_new)
    interpolated_df = pd.DataFrame(data=interpolated_array,
                                   index=y_new,
                                   columns=x_new)
    return interpolated_df

def bicubic_interpolation(i=0, t=0):
    """
    :param i,t : This function takes i and t as integer input to use in create_grid(i,t) function.
    :return: Function returns a pandas dataframe:
                Indexs     : y_locations
                Columns    : x_locations
                Data_array : Bicubic interpolated temperature values
    """
    temperature_data, x_old, y_old, x_new, y_new= create_grid(i,t)
    xx, yy = np.meshgrid(x_old, y_old)
    pivot_table = temperature_data.pivot('x_location', 'y_location', 'temperature')
    sensors_data_array = pivot_table.values
    f = interpolate.interp2d(xx, yy, sensors_data_array, kind='cubic')
    interpolated_array = f(x_new, y_new)
    interpolated_df = pd.DataFrame(data=interpolated_array,
                                   index=y_new,
                                   columns=x_new)
    return interpolated_df

def linear_regression(i,t,predictors=[],temperature_prediction=[]):
    """

    :param i, t: This function takes 'i' and 't' as an input to use in create_grid(i,t) function.
    :param predictors: This function takes "predictors" as an input to create linear regression model.
            Default of predictors is "[]". This means, it will taken from temperature data in this function.
    :param temperature_prediction: This function takes "temperature_prediction" as an input to create linear regression model.
            Default of temperature_prediction is "[]". This means, it will taken from temperature data in this function.
    :return: Linear regression fitted model.
    """

    if (np.array(predictors).shape[0]) != 0 and (np.array(temperature_prediction).shape[0]) != 0:
        predictors = predictors
        temperature_prediction = temperature_prediction
    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        predictors = temperature_data[['x_location',
                                       'y_location']].values
        temperature_prediction = temperature_data['temperature'].values
    regr = linear_model.LinearRegression()
    model = regr.fit(predictors, temperature_prediction)
    return model

def poly_regression(i,t,predictors=[],temperature_prediction=[]):
    """

    :param i, t: This function takes 'i' and 't' as an input to use in create_grid(i,t) function.
    :param predictors: This function takes "predictors" as an input to create polynomial regression model.
            Default of predictors is "[]". This means, it will taken from temperature data in this function.
    :param temperature_prediction: This function takes "temperature_prediction" as an input to create polynomial regression model.
            Default of temperature_prediction is "[]". This means, it will taken from temperature data in this function.
    :return: Polynomial regression fitted model.
    """
    if (np.array(predictors).shape[0]) != 0 and (np.array(temperature_prediction).shape[0]) != 0:
        predictors = predictors
        temperature_prediction = temperature_prediction
    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        predictors = temperature_data[['x_location',
                                       'y_location']].values
        temperature_prediction = temperature_data['temperature'].values
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=4, include_bias=False)),
        ('linreg', LinearRegression(normalize=True))
    ])
    model = pipeline.fit(predictors, temperature_prediction)
    return model

def alternative_polynomial_regression(i,t,predictors=[],temperature_prediction=[]):
    def PolynomialRegression(degree=2, **kwargs):
        return Pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    if (np.array(predictors).shape[0]) != 0 and (np.array(temperature_prediction).shape[0]) != 0:
        predictors = predictors
        temperature_prediction = temperature_prediction
    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        predictors = temperature_data[['x_location',
                                       'y_location']].values
        temperature_prediction = temperature_data['temperature'].values

    param_grid = {'polynomialfeatures__degree': np.arange(20),
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
    grid.fit(predictors, temperature_prediction)

    model = grid.best_estimator_

    return print(model)

def random_forest_regression(i,t,predictors=[],temperature_prediction=[]):
    """
    :param i, t: This function takes 'i' and 't' as an input to use in create_grid(i,t) function.
    :param predictors: This function takes "predictors" as an input to create random forest regression model.
            Default of predictors is "[]". This means, it will taken from temperature data in this function.
    :param temperature_prediction: This function takes "temperature_prediction" as an input to create random forest regression model.
            Default of temperature_prediction is "[]". This means, it will taken from temperature data in this function.
    :return: Random forest regression fitted model.
    """
    if (np.array(predictors).shape[0]) != 0 and (np.array(temperature_prediction).shape[0]) != 0:
        predictors = predictors
        temperature_prediction = temperature_prediction
    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        predictors = temperature_data[['x_location',
                                       'y_location']].values
        temperature_prediction = temperature_data['temperature'].values
    rf = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=20)
    model = rf.fit(predictors, np.ravel(temperature_prediction))
    return model

def gaussian_regression(i,t,predictors=[],temperature_prediction=[]):
    """
    :param i, t: This function takes 'i' and 't' as an input to use in create_grid(i,t) function.
    :param predictors: This function takes "predictors" as an input to create gaussian process regression model.
            Default of predictors is "[]". This means, it will taken from temperature data in this function.
    :param temperature_prediction: This function takes "temperature_prediction" as an input to create gaussian process regression model.
            Default of temperature_prediction is "[]". This means, it will taken from temperature data in this function.
    :return: Random gaussian process regression fitted model.
    """
    if (np.array(predictors).shape[0]) != 0 and (np.array(temperature_prediction).shape[0]) != 0:
        predictors = predictors
        temperature_prediction = temperature_prediction

    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        predictors = temperature_data[['x_location',
                                       'y_location']].values
        temperature_prediction = temperature_data['temperature'].values

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1,1e3))
    gr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    model = gr.fit(predictors, temperature_prediction)

    return model

#Plotting types/interfaces

def heatmap_with_matplotlib(i, t, f):
    """

    :param i: This function takes 'i' as input and using at prediction_to_array(i,t,f,original=True) function to create dataframe which is splitted according to 'i' value.
    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :param f: This function takes 'f' as an input to define which function/model you want to use.
    :return: This function returns matplotlib based heatmap. Which has time and interpolation slider on it.
    """
    xx = (prediction_to_array(i,t,f,original=True).columns.values.tolist())
    yy = prediction_to_array(i,t,f,original=True).index.values.tolist()
    fig, ax = plt.subplots()
    img = plt.imread("overlay.png")

    im = ax.imshow(prediction_to_array(i,t,f,original=True).values, cmap="coolwarm")
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
        im.set_data(prediction_to_array(i,t,f,original=True).values)


    int_slider_dep.on_changed(update_depth)
    time_slider_dep.on_changed(update_depth)
    return plt.show()

def multiple_heatmap_with_matplotlib(i, t, f1, f2, f3, f4,original):
    """

    :param i: This function takes 'i' as input and using at interpolatin(i, t) function to create dataframe which is splitted according to 'i' value.
    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :return: This function returns matplotlib based heatmap. Which has time and interpolation slider on it.
    """
    xx = (prediction_to_array(i,t,f1,original=True).columns.values.tolist())
    yy = (prediction_to_array(i,t,f1,original=True).index.values.tolist())
    img = plt.imread("overlay.png")
    fig=plt.figure()

    ### Figure 1 ###
    ax1=fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(prediction_to_array(i,t,f1,original=True).values, cmap="RdBu_r")
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
    im2 = ax2.imshow(prediction_to_array(i,t,f2,original=True).values, cmap="RdBu_r")
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
    im3 = ax3.imshow(prediction_to_array(i,t,f3,original=True).values, cmap="RdBu_r")
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
    im4 = ax4.imshow(prediction_to_array(i,t,f4,original=True), cmap="RdBu_r")
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
        im1.set_data(prediction_to_array(i,t,f1,original=True).values)
        im2.set_data(prediction_to_array(i,t,f2,original=True).values)
        im3.set_data(prediction_to_array(i,t,f3,original=True).values)
        im4.set_data(prediction_to_array(i,t,f4,original=True).values)

    int_slider_dep.on_changed(update_depth)
    time_slider_dep.on_changed(update_depth)
    return plt.show()

def interface():
    import sys
    from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    import matplotlib.pyplot as plt

    import random

    class Window(QDialog):
        def __init__(self, parent=None):
            super(Window, self).__init__(parent)

            # a figure instance to plot on
            self.figure = plt.figure()

            # this is the Canvas Widget that displays the `figure`
            # it takes the `figure` instance as a parameter to __init__
            self.canvas = FigureCanvas(self.figure)

            # this is the Navigation widget
            # it takes the Canvas widget and a parent
            self.toolbar = NavigationToolbar(self.canvas, self)

            # Just some button connected to `plot` method
            self.button = QPushButton('Plot')
            self.button.clicked.connect(self.plot)

            # set the layout
            layout = QVBoxLayout()
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
            layout.addWidget(self.button)
            self.setLayout(layout)

        def plot(self):
            xx = (bilinear_interpolation(0,0).columns.values.tolist())
            yy = (bilinear_interpolation(0,0).index.values.tolist())
            figure, ax = plt.subplots()
            img = plt.imread("overlay.png")

            im = ax.imshow((bilinear_interpolation(0,0).values))
            ax.invert_yaxis()
            ax.set_xticks(np.arange(len(xx)))
            ax.set_yticks(np.arange(len(yy)))
            ax.set_xticklabels(xx)
            ax.set_yticklabels(yy)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            ax.set_title("Temperature Data of Factory")
            figure.tight_layout()

            # instead of ax.hold(False)
            self.figure.clear()

            # create an axis
            ax = self.figure.add_subplot(111)

            # refresh canvas
            self.canvas.draw()

    if __name__ == '__main__':
        app = QApplication(sys.argv)

        main = Window()
        main.show()

        sys.exit(app.exec_())

def threed_plot(i,t,Original=True):
    """
    :param i: This function takes 'i' as input and using at interpolatin(i, t) function to create dataframe which is splitted according to 'i' value.
    :param t: This function takes 't' value as input and using at select_time(t) function to get 't' filtered dataframe.
    :return: This function returns 3d plot of data.
    """
    xx = (f(i, t).columns.values.tolist())
    yy = (f(i, t).index.values.tolist())
    z = f(i, t).values
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

#Machine Learning

def supervised_learning_test(i,t,f):
    temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
    train_data = temperature_data.sample(frac=0.8, random_state=200)
    test_Data = temperature_data.drop(train_data.index)
    train_location= train_data[['x_location', 'y_location']].values
    train_temperature= train_data[['temperature']].values

    test_location = test_Data[['x_location', 'y_location']].values
    test_temperature = test_Data[['temperature']].values

    model = f(i,t,predictors=train_location,temperature_prediction=train_temperature)
    r2 = r2_score(test_temperature, model.predict((test_location)))
    return r2

def check_with_reference(i,t,*args):
    temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
    train_data = temperature_data.sample(frac=0.9, random_state=200)
    test_data = temperature_data.drop(train_data.index)

    train_location = train_data[['x_location', 'y_location']].values
    train_temperature = train_data[['temperature']].values

    test_location = test_data[['x_location', 'y_location']].values
    test_temperature = test_data[['temperature']].values
    final_df = pd.DataFrame({'x_location':test_location[:,0],'y_location':test_location[:,1]})
    final_df['validation_temperature']= test_temperature
    for arg in args:
        model = arg(i, t, predictors=train_location, temperature_prediction=train_temperature)
        test_prediction = model.predict(test_location)
        mse = mean_squared_error(test_temperature, test_prediction)
        mae = mean_absolute_error(test_temperature, test_prediction)
        final_df['{}_mean_squared_error'.format(arg.__name__)] = mse

        final_df['{}_reference'.format(arg.__name__)] = test_prediction
        final_df['{}_error'.format(arg.__name__)] = (final_df.validation_temperature - final_df['{}_reference'.format(arg.__name__)]).abs()
        final_df['{}_error_percentage'.format(arg.__name__)] = final_df['{}_error'.format(arg.__name__)] / final_df.validation_temperature
    return final_df

def removing_sensor_nodes(i,t,f):
    temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
    df = pd.DataFrame(columns=['x_location', 'y_location', 'validation_temperature', '{}_prediction'.format(f.__name__), 'error_percentage'])
    sensor_list=[]
    for row in range(len(temperature_data.index)):
        test_data = temperature_data.iloc[[row]]
        train_data = temperature_data.drop([row])
        train_location = train_data[['x_location', 'y_location']].values
        train_temperature = train_data[['temperature']].values
        test_location = test_data[['x_location', 'y_location']].values
        test_temperature = test_data[['temperature']].values

        model = f(i, t, predictors=train_location, temperature_prediction=train_temperature)
        test_prediction = model.predict(test_location)
        error = np.absolute((test_temperature - test_prediction)/test_temperature)

        sensor_list.append({'x_location': test_location[:,0], 'y_location': test_location[:,1], 'validation_temperature': test_temperature,
                            '{}_prediction'.format(f.__name__): test_prediction, 'error_percentage': error})
    sensor_data = df.append(sensor_list)
    sorted_sensor_data =sensor_data.sort_values(by=['error_percentage'])
    return sorted_sensor_data

#Deep Learning

def tf_regression(i,t,predictors=[],temperature_prediction=[]):
    temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
    predictors = temperature_data[['x_location',
                                       'y_location']]
    temperature_prediction = temperature_data['temperature']

    scaler = MinMaxScaler()
    scaler.fit(predictors)
    scaled_predictors = pd.DataFrame(data=scaler.transform(predictors), columns=predictors.columns, index=predictors.index)


    x_cor_feature = tf.feature_column.numeric_column('x_location')
    y_cor_feature = tf.feature_column.numeric_column('y_location')
    feat_cols= [x_cor_feature,y_cor_feature]
    input_func = tf.estimator.inputs.pandas_input_fn(x=scaled_predictors, y=temperature_prediction, batch_size=10, num_epochs=1000,
                                                     shuffle=True)
    model = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)
    model.train(input_fn=input_func, steps=1000)

    return model





def keras_prediction(i,t,predictors=[],temperature_prediction=[]):

    if (np.array(predictors).shape[0]) != 0 and (np.array(temperature_prediction).shape[0]) != 0:
        predictors = predictors
        temperature_prediction = temperature_prediction
    else:
        temperature_data, x_old, y_old, x_new, y_new = create_grid(i, t)
        predictors = temperature_data[['x_location',
                                       'y_location']].values
        temperature_prediction = temperature_data['temperature'].values
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(predictors, temperature_prediction, epochs=150, batch_size=10, verbose=2)
    predictions = model.predict(predictors)

    return model

#Time Series Analsys

def timely_prediction_to_series(n,f):
    node_predictors,node_prediction,model=f(n)
    series=model.predict(node_predictors)
    return series

def timely_linear_regression(n,node_predictors=[],node_prediction=[]):

    if (np.array(node_predictors).shape[0]) != 0 and (np.array(node_prediction).shape[0]) != 0:
        node_predictors = node_predictors
        node_prediction = node_prediction
    else:
        sensor_code=select_sensor(n)
        temperature_data=temperature_datas.pivot('time_stamp','sensor_code','temperature')
        node_prediction= temperature_data[sensor_code].values
        node_predictors=temperature_data.drop(columns=['{}'.format(sensor_code)]).values

    regr=linear_model.LinearRegression()
    model=regr.fit(node_predictors,node_prediction)
    return node_predictors, node_prediction, model

def timely_random_forest_regression(n,node_predictors=[],node_prediction=[]):

    if (np.array(node_predictors).shape[0]) != 0 and (np.array(node_prediction).shape[0]) != 0:
        node_predictors = node_predictors
        node_prediction = node_prediction
    else:
        sensor_code=select_sensor(n)
        temperature_data=temperature_datas.pivot('time_stamp','sensor_code','temperature')
        node_prediction= temperature_data[sensor_code].values
        node_predictors=temperature_data.drop(columns=['{}'.format(sensor_code)]).values

    rf = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=20)
    model = rf.fit(node_predictors, np.ravel(node_prediction))
    return node_predictors, node_prediction, model

def timely_gaussian_regression(n,node_predictors=[],node_prediction=[]):

    if (np.array(node_predictors).shape[0]) != 0 and (np.array(node_prediction).shape[0]) != 0:
        node_predictors = node_predictors
        node_prediction = node_prediction
    else:
        sensor_code=select_sensor(n)
        temperature_data=temperature_datas.pivot('time_stamp','sensor_code','temperature')
        node_prediction= temperature_data[sensor_code].values
        node_predictors=temperature_data.drop(columns=['{}'.format(sensor_code)]).values

    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e3))
    gr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    model = gr.fit(node_predictors, node_prediction)
    return node_predictors, node_prediction, model

def timely_polynomial_regression(n,node_predictors=[],node_prediction=[]):

    if (np.array(node_predictors).shape[0]) != 0 and (np.array(node_prediction).shape[0]) != 0:
        node_predictors = node_predictors
        node_prediction = node_prediction
    else:
        sensor_code=select_sensor(n)
        temperature_data=temperature_datas.pivot('time_stamp','sensor_code','temperature')
        node_prediction= temperature_data[sensor_code].values
        node_predictors=temperature_data.drop(columns=['{}'.format(sensor_code)]).values

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=4, include_bias=False)),
        ('linreg', LinearRegression(normalize=True))
    ])
    model = pipeline.fit(node_predictors, node_prediction)
    return node_predictors, node_prediction, model

def timely_supervised_learning_test(n,*args):
    sensor_code = select_sensor(n)
    temperature_data = temperature_datas.pivot('time_stamp', 'sensor_code', 'temperature')
    train_data = temperature_data.sample(frac=0.8, random_state=200)
    test_Data = temperature_data.drop(train_data.index)
    node_prediction_train = train_data[sensor_code].values
    node_predictors_train=train_data.drop(columns=['{}'.format(sensor_code)]).values

    node_validation_test = test_Data[sensor_code].values
    node_predictors_test = test_Data.drop(columns=['{}'.format(sensor_code)]).values
    final_df=pd.DataFrame({'validation_temperature':node_validation_test})

    for arg in args:
        node_predictors, node_prediction, model = arg(n, node_predictors=node_predictors_train, node_prediction=node_prediction_train)
        predicted_temperatures=model.predict((node_predictors_test))
        final_df['{}_reference'.format(arg.__name__)] = predicted_temperatures
        final_df['{}_error'.format(arg.__name__)] = (final_df.validation_temperature - final_df['{}_reference'.format(arg.__name__)]).abs()
        final_df['{}_error_percentage'.format(arg.__name__)] = final_df['{}_error'.format(
            arg.__name__)] / final_df.validation_temperature
        final_df['{}_accuracy_percentage'.format(arg.__name__)] = 1-final_df['{}_error'.format(
            arg.__name__)] / final_df.validation_temperature
    return final_df

(timely_supervised_learning_test(0,timely_gaussian_regression,timely_linear_regression,timely_random_forest_regression))
