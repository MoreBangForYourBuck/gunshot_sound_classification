from tqdm import tqdm
import torch
from torch import nn
import torchmetrics as tm
import torch
from torch.utils.data import DataLoader
from helpers.preprocessing import cross_entropy_weights, get_distribution, normalize_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List
from copy import deepcopy
import numpy as np
import torchvision

def model_output_to_classes(model_output:torch.Tensor) -> torch.Tensor:
    return torch.max(model_output, 1)[1] # Indices of max values


def stats(model:nn.Module, dataloader:torch.utils.data.DataLoader,num_classes) -> float:
    precisions = []
    recalls = []
    f_ones = []
    precision = tm.Precision(task="multiclass", average='macro', num_classes=num_classes)
    recall = tm.Recall(task="multiclass", average='macro', num_classes=num_classes)
    f_one = tm.F1Score(task="multiclass", num_classes=num_classes)

    for (X, y) in tqdm(dataloader):
        model.eval()
        with torch.no_grad():
            y_p = model_output_to_classes(model(X))

            recalls.append((recall(y_p,y)).item())
            precisions.append((precision(y_p,y)).item())
            f_ones.append((f_one(y_p,y)).item())
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f_ones)/len(recalls)

def accuracy(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader) -> float:
        sum = 0
        length = 0
        for (X, y) in tqdm(dataloader):
            model.eval()
            with torch.no_grad():
                y_p = model(X)
                sum += torch.sum(y == torch.round(y_p)).item()
                length += len(y_p)
        return sum/length

def plot_loss(train_loss_history:List[float], val_loss_history:List[float]) -> None:
        plt.figure()
        plt.title('Loss curve')
        plt.plot(range(len(train_loss_history)), train_loss_history, label='train loss')
        plt.plot(range(len(val_loss_history)), val_loss_history, label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()