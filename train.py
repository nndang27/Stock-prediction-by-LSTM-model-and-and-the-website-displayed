from model import LSTMModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_epoch(config, scheduler, optimizer, criterion, model, dataloader, is_training=False):
    epoch_loss = 0
    if is_training:
        model.train()
    else:
        model.eval()
    for idx, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if is_training:
            optimizer.zero_grad()
        batchsize = x.shape[0]
        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])
        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())
        if is_training:
            loss.backward()
            optimizer.step()
        epoch_loss += (loss.detach().item() / batchsize)
    lr = scheduler.get_last_lr()[0]
    return epoch_loss, lr


def save_model(epochs, model, optimizer, criterion, checkpoint_name):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"E:/GR1/checkpoint/excel_fpt/" + str(checkpoint_name))
    print("Saved sucessfully!!")
