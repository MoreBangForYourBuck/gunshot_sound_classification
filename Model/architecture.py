from torch import nn
import torch
import torchvision

from helpers.AudioSampler import AudioSampler

class MLPModel(nn.Module):
    def __init__(self, hyperparams:dict):
        super(MLPModel, self).__init__()
        self.input = nn.Linear(hyperparams['window_size'], 20)
        self.relu = nn.ReLU() # Activation function
        self.hidden_layer = nn.Linear(20, hyperparams['num_classes'])
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.hidden_layer(x)
        return self.output(x)
    
class ResNetModel(nn.Module):
    def __init__(self, hyperparams:dict):
        super(ResNetModel, self).__init__()
        self.input = nn.Linear(1000, 512)
        self.relu = nn.ReLU() # Activation function
        self.hidden_layer = nn.Linear(512, hyperparams['num_classes'])
        self.output = nn.Sigmoid()
        self.rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
    def forward(self, x):
        x = self.relu(self.input(self.rn(x)))
        return self.output(self.hidden_layer(x))

class GunshotDataset(torch.utils.data.Dataset):
    sampler = AudioSampler()

    def __init__(self,hyperparams,size):
        self.window_size = hyperparams['window_size']
        if hyperparams['model_type'] == 'mlp':
            self.X,self.y = GunshotDataset.sampler.sample_array(size,self.window_size, convert_to_mono=True)
        else:
            self.X,self.y = GunshotDataset.sampler.sample_array(size,self.window_size, convert_to_mono=True,output_spectrogram=True)
        self.X = torch.tensor(self.X,requires_grad=True)
        self.y = torch.tensor(self.y,requires_grad=True)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
