import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os

def plot_cm(cm):
    fig, ax = plt.subplots(figsize=(18, 16)) 
    ax = sns.heatmap(
        cm, 
        annot=True, 
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )


class ResNetModel(nn.Module):
    def __init__(self,x_shape, BATCH_SIZE):
        super(ResNetModel, self).__init__()
        self.batch = BATCH_SIZE
        self.input = nn.Linear(1000, 512)
        self.relu = nn.ReLU() # Activation function
        self.length = x_shape
        self.hidden_layer = nn.Linear(512, x_shape)
        self.output = nn.Sigmoid()
        self.rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
    def forward(self, x):
        x = x.reshape((self.batch,3,self.length,683)) # x = x.reshape((self.batch,3,self.length,43))
        x = self.relu(self.input(self.rn(x)))
        return self.output(self.hidden_layer(x))

class Dataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray):
        self.X = X.copy()
        self.y = y.copy()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx:int):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]).to(torch.float32)
        )
        
def calc_acc(y, y_p):
    return torch.sum((y_p > 0.5).int() == y) / (y.shape[0] * y.shape[1])

def zeros_and_ones(t):
    ones = torch.tensor((t*2),dtype=torch.long).sum().detach().item()
    total = len(t.detach().numpy())
    return (total-ones)/total*100,ones/total*100

def class_counts(y):
    return [torch.sum(y == 0), torch.sum(y == 1)]

def distribution(y):
    frac_ones = np.sum(y) / (y.shape[0] * y.shape[1])
    return torch.tensor([frac_ones, 1 - frac_ones])

if __name__ == '__main__':
    print('CUDA' if torch.cuda.is_available() else 'CPU')

    X = np.load('dataset/TestDataNpz/spectrograms.npz')
    y = np.load('dataset/TestDataNpz/labels.npz')

    X = np.array(X['a']).reshape((1000,56,2049))
    y = np.array(y['a']).reshape((1000,56))
        
    # Define train/val split
    num_samples = X.shape[0]

    BATCH_SIZE = 10
    model = ResNetModel(56,BATCH_SIZE) #(10,3,68,10)

    model  = joblib.load('lstm_torch_train0.2711311876773834_val0.218733012676239.joblib') 

    
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCELoss()
    
    test_dataloader = DataLoader(Dataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    cms0 = []
    cms1 = []
    cms2 = []
    cms3 = []
    for X, y in test_dataloader:
        print('=',sep='',end='')
        y_p = model(X)
        loss = criterion(y_p, y)
        accuracies.append(calc_acc(y, y_p).item())
        precisions.append(precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
        recalls.append(recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
        f1s.append(f1_score(y.flatten().detach(),y_p.flatten().detach().round()))
        cms0.append(np.array(confusion_matrix(y.flatten().detach(), y_p.flatten().detach().round()))[0][0])
        cms1.append(np.array(confusion_matrix(y.flatten().detach(), y_p.flatten().detach().round()))[0][1])
        cms2.append(np.array(confusion_matrix(y.flatten().detach(), y_p.flatten().detach().round()))[1][0])
        cms3.append(np.array(confusion_matrix(y.flatten().detach(), y_p.flatten().detach().round()))[1][1])
        plot_cm(confusion_matrix(y.flatten().detach(), y_p.flatten().detach().round()))
    print()
    print("Accuracy",np.mean(accuracies))
    print("Precision",np.mean(precisions))
    print("Recall",np.mean(recalls))
    print("F1",np.mean(f1s))
    plot_cm(np.array([sum(cms0),sum(cms1),sum(cms2),sum(cms3)]))


    