from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torchsummary import summary
import torch
from torch.utils.data import DataLoader

from helpers.eval import accuracy, plot_loss
from helpers.preprocessing import get_model_params
from architecture import GunshotDataset, MLPModel,ResNetModel
from helpers.preprocessing import cross_entropy_weights, get_distribution, normalize_data

if __name__ == '__main__':
    hyperparams = get_model_params()

    train = True #set to train and save, or load and eval
    reload = False

    if train == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if reload == True:
            model = joblib.load(hyperparams['model_type']+'.joblib') 
        else:
            if hyperparams['model_type'] =='mlp':
                model = MLPModel(hyperparams).to(device)
            elif hyperparams['model_type'] == 'resnet':
                model = ResNetModel(hyperparams).to(device)
        #summary(model,(32,100))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        criterion = torch.nn.MSELoss()
        
        train_loss_history = []
        val_loss_history = []
        val_acc = None
        train_acc = None
    
        train_generator = DataLoader(GunshotDataset(hyperparams,8),hyperparams['batch_size'])
        val_generator = DataLoader(GunshotDataset(hyperparams,2),hyperparams['batch_size'])
        for epoch in range(1, hyperparams['epochs'] + 1):
            print(f'Epoch {epoch}')
            
            # # Normalization (optionally)
            # if hyperparams['normalize']['run']:
            #     X_train, scaler = normalize_data(X_train, method=hyperparams['normalize']['method'])
            #     if scaler: # None if method is 'mean'
            #         X_val = scaler.transform(X_val)
            
            # Batch train
            model.train()
            batch_train_loss_history = []
            for (X, y) in tqdm(train_generator):
                optimizer.zero_grad()
                
                y_p = model(X)
                loss = criterion(y_p.squeeze(),y)

                loss.backward()
                optimizer.step()
                batch_train_loss_history.append(loss.item())
            
            # Batch validation
            model.eval()
            batch_val_loss_history = []
            for (X, y) in tqdm(val_generator):
                with torch.no_grad():
                    y_p = model(X)
                
                loss = criterion(y_p.squeeze(), y)
                batch_val_loss_history.append(loss.item())
            
            # Batch average loss
            epoch_train_loss = sum(batch_train_loss_history) / len(batch_train_loss_history)
            epoch_val_loss = sum(batch_val_loss_history) / len(batch_val_loss_history)
            print(f'Train loss: {epoch_train_loss:.4f}\nVal loss: {epoch_val_loss:.4f}')
            
            # Append batch loss to epoch loss list
            train_loss_history.append(epoch_train_loss)
            val_loss_history.append(epoch_val_loss)
        
        # Calculate accuracy
        #train_acc = accuracy(model, train_generator)
        #val_acc = accuracy(model, val_generator)


        joblib.dump(model, model_type+'.joblib') #save model
    else:
        model = joblib.load(model_type+'.joblib') 

    model_accuracy = accuracy(model,val_generator)
    print('Model accuracy: {0}%\n'.format(model_accuracy))
    plot_loss(train_loss_history, val_loss_history)