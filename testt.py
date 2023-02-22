import os

import np as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils import plotting_data
from data_loading import Normalizer, TimeSeriesDataset, prepare_data
from model import LSTMModel
import numpy as np
from glob import glob
import torch
from matplotlib.pyplot import figure

config = {
    "file_name": "E:/GR1/dataset/excel_fpt.csv",
    "main_feature": '<OpenFixed>',
    "stock_name": '<Ticker>',
    "date_feature": '<DTYYYYMMDD>',
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}
scaler = Normalizer()


def data_handling(file_csv_path, main_feature):
    global config
    config["file_name"] = str(file_csv_path)
    config["main_feature"] = str(main_feature)
    data_date, data_close_price, num_data_points, display_date_range, stockname = plotting_data(config, plot=False)
    # scaler = Normalizer()
    # print(display_date_range, data_close_price)
    normalized_data_close_price = scaler.fit_transform(data_close_price)
    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(scaler, data_date,
                                                                                                  num_data_points,
                                                                                                  normalized_data_close_price,
                                                                                                  config,
                                                                                                  plot=False)
    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen


def predict_n_days(n, checkpoint_path, data_last_days):
    model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                      num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
    DEVICE = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    results = []
    for i in range(n):
        data = data_last_days[len(data_last_days) - n:]
        x = torch.tensor(data).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
        prediction = model(x)
        prediction = prediction.cpu().detach().numpy()
        prediction_before_transform = prediction
        prediction = scaler.inverse_transform(prediction)[0]
        results.append(str(round(prediction, 3)))
        data_last_days = np.append(data_last_days, prediction_before_transform[0])
    return results


def predict_all(num_days, file_data_path, folder_checkpoint):
    predict_open = []
    predict_high = []
    predict_low = []
    predict_close = []
    predict_volume = []
    for checkpoint_path in glob(folder_checkpoint + '/*.pth'):
        filename = os.path.basename(checkpoint_path)
        feature = "<" + filename[filename.find("_") + 1:filename.find(".")] + ">"
        split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_last_days = data_handling(
            str(file_data_path), str(feature))
        prediction = predict_n_days(num_days, checkpoint_path, data_last_days)
        if filename[filename.find("_") + 1:filename.find(".")] == "OpenFixed":
            predict_open[:] = prediction
        if filename[filename.find("_") + 1:filename.find(".")] == "HighFixed":
            predict_high[:] = prediction
        if filename[filename.find("_") + 1:filename.find(".")] == "LowFixed":
            predict_low[:] = prediction
        if filename[filename.find("_") + 1:filename.find(".")] == "CloseFixed":
            predict_close[:] = prediction
        if filename[filename.find("_") + 1:filename.find(".")] == "Volume":
            predict_volume[:] = prediction
    return predict_open, predict_high, predict_low, predict_close, predict_volume


if __name__ == "__main__":

    data_date, data_close_price, num_data_points, display_date_range, stockname = plotting_data(config, plot=False)
    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_last_days = data_handling(config["file_name"],
                                                                                                    config[
                                                                                                        "main_feature"])
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    # predict on the training data, to see how well the model managed to learn and memorize
    model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                      num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
    DEVICE = 'cpu'
    checkpoint = torch.load(r"E:\GR1\checkpoint\excel_fpt\FPT_OpenFixed.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))
    # predict on the validation data, to see how the model does
    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # if config["plots"]["show_plots"]:
    #     # prepare data for plotting, show predicted prices
    #
    #     to_plot_data_y_train_pred = np.zeros(num_data_points)
    #     to_plot_data_y_val_pred = np.zeros(num_data_points)
    #
    #     to_plot_data_y_train_pred[
    #     config["data"]["window_size"]:split_index + config["data"]["window_size"]] = scaler.inverse_transform(
    #         predicted_train)
    #     to_plot_data_y_val_pred[split_index + config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)
    #
    #     to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    #     to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    #
    #     # plots
    #
    #     fig = figure(figsize=(25, 5), dpi=80)
    #     fig.patch.set_facecolor((1.0, 1.0, 1.0))
    #     plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
    #     plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)",
    #              color=config["plots"]["color_pred_train"])
    #     plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)",
    #              color=config["plots"]["color_pred_val"])
    #     plt.title("Compare predicted open prices to actual open prices of FPT")
    #     xticks = [data_date[i] if ((i % config["plots"]["xticks_interval"] == 0 and (num_data_points - i) >
    #                                 config["plots"]["xticks_interval"]) or i == num_data_points - 1) else None for i in
    #               range(num_data_points)]  # make x ticks nice
    #     x = np.arange(0, len(xticks))
    #     plt.xticks(x, xticks, rotation='vertical')
    #     plt.grid(b=None, which='major', axis='y', linestyle='--')
    #     plt.legend()
    #     plt.show()
# predict_open, predict_high, predict_low, predict_close, predict_volume = predict_all("E:/GR1/dataset/excel_aaa.csv", "E:/GR1/checkpoint")
    x = torch.tensor(data_last_days).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(
        2)  # this is the data type and shape required, [batch, sequence, feature]
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()
    prediction = scaler.inverse_transform(prediction)[0]

    if config["plots"]["show_plots"]:
        # prepare plots

        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range - 1] = scaler.inverse_transform(data_y_val)[-plot_range + 1:]
        to_plot_data_y_val_pred[:plot_range - 1] = scaler.inverse_transform(predicted_val)[-plot_range + 1:]

        to_plot_data_y_test_pred[plot_range - 1] = prediction

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # plot

        plot_date_test = data_date[-plot_range + 1:]
        plot_date_test.append("next trading day")

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10,
                 color=config["plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10,
                 color=config["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".",
                 markersize=20, color=config["plots"]["color_pred_test"])
        plt.title("Predicted close price of the next trading day")
        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()