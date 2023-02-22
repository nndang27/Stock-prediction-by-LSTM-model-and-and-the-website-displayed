
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np


def read_file(file_name, main_feature, date_feature, stock_name):  # main_feature now is choosen <CloseFixed>
    df = pd.read_csv(file_name)
    date_ = df.filter([date_feature])
    data = df.filter([main_feature])
    dataset = data.values
    date_ = date_.values.flatten()
    date_data = [str(e)[0:4] + "-" + str(e)[4:6] + "-" + str(e)[6:8] for e in date_]
    stock_namee = df.filter([stock_name])
    return dataset.flatten(), date_data, stock_namee


def plotting_data(config, plot=False):
    # get the data from alpha vantage

    data_close_price, data_date, stockname = read_file(config["file_name"], config["main_feature"], config["date_feature"], config["stock_name"])

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points - 1]
    # print("Number data points:", num_data_points, display_date_range)

    if plot:
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
        xticks = [data_date[i] if ((i % config["plots"]["xticks_interval"] == 0 and (num_data_points - i) >
                                    config["plots"]["xticks_interval"]) or i == num_data_points - 1) else None for i in
                  range(num_data_points)]  # make x ticks nice
        x = np.arange(0, len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title("Daily close price for ")
        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.show()

    return data_date, data_close_price, num_data_points, display_date_range, stockname


# data_date, data_close_price, num_data_points, display_date_range = plotting_data(config, plot=False)
