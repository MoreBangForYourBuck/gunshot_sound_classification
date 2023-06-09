import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import joblib
from matplotlib import pyplot as plt
from tqdm import tqdm
from pprint import pprint
from torchmetrics import F1Score


class LSTMModel(nn.Module):
    def __init__(self, X_shape:Tuple[int, int, int]):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(X_shape[1], hidden_size=256, num_layers=1, dropout=0.2, bidirectional=True,
                            batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        #h = h.view(h.shape[1], -1)
        x = self.relu(self.fc1(h[0]))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.flatten()


class LSTMDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray, window_size:int=10):
        self.window_size = window_size
        self.X = X.copy()
        self.y = y.copy()
    
    def __len__(self):
        return self.X.shape[0] - self.window_size
    
    def __getitem__(self, idx:int):
        return (
            torch.tensor(self.X[idx:idx+self.window_size, :]),
            torch.tensor(self.y[idx+self.window_size]).to(torch.float32)
        )


def calc_acc(y, y_p):
    return torch.sum((y_p > 0.5).int() == y) / (y.shape[0])

def zeros_and_ones(t):
    ones = torch.tensor((t*2),dtype=torch.long).sum().detach().item()
    total = len(t.detach().numpy())
    return (total-ones)/total*100,ones/total*100

def class_counts(y):
    return [torch.sum(y == 0), torch.sum(y == 1)]

def distribution(y):
    frac_ones = np.sum(y) / (y.shape[0])
    return torch.tensor([frac_ones, 1 - frac_ones])

if __name__ == '__main__':
    print('CUDA' if torch.cuda.is_available() else 'CPU')

    # Load X and y
    # X = np.load('dataset/spectrograms.npy')
    # y = np.load('dataset/labels.npy')
    X = np.load('dataset/TrainDataNpz/spectrograms.npz')
    y = np.load('dataset/TrainDataNpz/labels.npz')
    X = np.array(X['a'])
    y = np.array(y['a'])
    # Assert that X and y have the same number of samples
    assert X.shape[0] == y.shape[0]

    # Define train/val split
    num_samples = X.shape[0]
    train_ratio = .99
    split_index = int(num_samples*train_ratio)

    # Split X and y
    X_train = X[:10000]
    X_val = X[split_index:]
    y_train = y[:10000]
    y_val = y[split_index:]
    # Assert that no samples are lost
    # assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
    # assert y_train.shape[0] + y_val.shape[0] == y.shape[0]

    model = LSTMModel(X_train.shape)

    EPOCHS = 25
    BATCH_SIZE = 10
    # criterion = nn.CrossEntropyLoss(weight=distribution(y_train))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_dataloader = DataLoader(LSTMDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(LSTMDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    f1_score = F1Score('binary', threshold=0.5, average='macro')
    
    best_validation_loss = 1
    
    def history():
        return {
            'train': {
                'loss': [],
                'acc': [],
                'f1': [],
                'precision': [],
                'recall': []
            },
            'val': {
                'loss': [],
                'acc': [],
                'f1': [],
                'precision': [],
                'recall': []
            }
        }
    
    epoch_history = history()
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        batch_history = history()

        model.train()
        for X, y in tqdm(batch_dataloader):
            
            y_p = model(X)
            loss = criterion(y_p, y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print(f1_score(y_p, y))
            
            batch_history['train']['loss'].append(loss.item())
            batch_history['train']['acc'].append(calc_acc(y, y_p).item())
            # batch_history['train']['f1'].append(f1_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['train']['precision'].append(
            #     precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['train']['recall'].append(recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
            
        epoch_history['train']['loss'].append(np.mean(batch_history['train']['loss']))
        epoch_history['train']['acc'].append(np.mean(batch_history['train']['acc']))
        # epoch_history['train']['f1'].append(np.mean(batch_history['train']['f1']))
        # epoch_history['train']['precision'].append(np.mean(batch_history['train']['precision']))
        # epoch_history['train']['recall'].append(np.mean(batch_history['train']['recall']))

        
        model.eval()
        for X, y in tqdm(val_dataloader):
            with torch.no_grad():
                y_p = model(X)
                loss = criterion(y_p, y)
                
            batch_history['val']['loss'].append(loss.item())
            batch_history['val']['acc'].append(calc_acc(y, y_p).item())
            # batch_history['val']['f1'].append(f1_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['val']['precision'].append(
            #     precision_score(y.flatten().detach(),y_p.flatten().detach().round()))
            # batch_history['val']['recall'].append(recall_score(y.flatten().detach(),y_p.flatten().detach().round()))
            
        epoch_history['val']['loss'].append(np.mean(batch_history['val']['loss']))
        epoch_history['val']['acc'].append(np.mean(batch_history['val']['acc']))
        # epoch_history['val']['f1'].append(np.mean(batch_history['val']['f1']))
        # epoch_history['val']['precision'].append(np.mean(batch_history['val']['precision']))
        # epoch_history['val']['recall'].append(np.mean(batch_history['val']['recall']))
        
        pprint(epoch_history)
        if np.mean(epoch_history['val']['loss']) < best_validation_loss:
            best_validation_loss = np.mean(epoch_history['val']['loss'])
            joblib.dump(model, f"bilstm_torch_train{epoch_history['train']['loss'][-1]}_val{epoch_history['val']['loss'][-1]}.joblib")
    plt.figure()
    plt.title('Loss curve')
    plt.plot(range(len(epoch_history['train']['loss'])), epoch_history['train']['loss'], label='train loss')
    plt.plot(range(len(epoch_history['val']['loss'])), epoch_history['val']['loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()