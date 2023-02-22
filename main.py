from matplotlib import pyplot as plt

from train import run_epoch, save_model
from utils import plotting_data
from data_loading import Normalizer, TimeSeriesDataset, prepare_data
from model import LSTMModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

config = {
    "file_name": "E:/GR1/dataset/excel_fpt.csv",
    "main_feature": '<LowFixed>',
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
data_date, data_close_price, num_data_points, display_date_range, stockname = plotting_data(config, plot=False)
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)
split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(scaler, data_date,
                                                                                              num_data_points,
                                                                                              normalized_data_close_price,
                                                                                              config,
                                                                                              plot=False)

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

# create `DataLoader`
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

# define optimizer, scheduler and loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

# begin training
best_valid_loss = float('inf')
checkpoint_name = stockname + "_"+config["main_feature"][1:len(config["main_feature"])-1] + ".pth"
all_loss_train = []
all_loss_val = []
for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(config, scheduler, optimizer, criterion, model, train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(config, scheduler, optimizer, criterion, model, val_dataloader)
    all_loss_train.append(loss_train)
    all_loss_val.append(loss_val)
    scheduler.step()

    print('\nEpoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
    if loss_val < best_valid_loss:
        save_model(epoch, model, optimizer, criterion, checkpoint_name="FPT_LowFixed.pth")
# Loss plots.
# plt.figure(figsize=(10, 7))
# plt.plot(
#     all_loss_train, color='orange', linestyle='-',
#     label='train loss'
# )
# plt.plot(
#     all_loss_val, color='red', linestyle='-',
#     label='validataion loss'
# )
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig(f"./checkpoint/excel_aaa/lossVolumeAAA.png")