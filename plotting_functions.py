from predicting_functions import polynomial_regression_prediction, linear_regression_prediction, random_forest_regression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def prediction_to_plot(prediction_method, removed_sensor_node, train_x, train_y, test_x, test_y, train_time, test_time):
    prediction, r2, rmse = prediction_method(train_x, train_y, test_x, test_y)
    fig1, ax1 = plt.subplots()
    ax1.plot(train_time, train_y, color="blue")
    ax1.plot(test_time, test_y, color="blue")
    ax1.plot(test_time, prediction, color="crimson", linestyle="dotted")
    ax1.plot([], [], alpha=0)
    ax1.plot([], [], alpha=0)

    ax1.set(xlabel='Time Stamp (hourly)', ylabel='Temperature (°C)',
            title='{}_result'.format(prediction_method.__name__))
    plt.gca().legend(('Actual Temperature Data of {} Coded Sensor Node (°C)'.format(
        removed_sensor_node), '__nolegend__', 'Predicticted Temperature (°C)'.format(
        prediction), "R2 score: {}".format(r2), "RMSE score: {}".format(rmse)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid()
    fig1.savefig("{}_results.png".format(prediction_method.__name__))
    return plt.show()


