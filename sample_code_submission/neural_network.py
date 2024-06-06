import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import wandb
from pytorch_lightning.loggers import WandbLogger



class NeuralNetwork(nn.Module):
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier

    """

    def __init__(self, train_data, input_shape=None, epoch=0):
        # since we are using an NN that may be loaded from a file, we need to know the input shape which may have changed if FE added new features
        self.input_shape = train_data.shape[1] if input_shape is None else input_shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print("Using device: ", self.device)
        super().__init__()
        
        
        self.in1 = nn.Linear(self.input_shape, 500)
        self.h1 = nn.Linear(500, 500)
        self.h2 = nn.Linear(500, 500)
        self.h3= nn.Linear(500, 500)
        self.out = nn.Linear(500, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        self.scaler = StandardScaler()
        self.predictions = []
        self.real_labels = []
        self.accuracy = 0
        self.epoch = epoch

        self.to(self.device)
    
    def forward(self, x):
        if x.shape[1] != self.input_shape:
            # if the model uses less features than the input data, only keep the features the model uses
            x = x.split(self.input_shape, dim=1)[1] 
        x = self.relu(self.in1(x))
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        x = self.relu(self.h3(x))
        x = self.out(x)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.forward(x)
        self.predictions.extend(y_hat.argmax(dim=-1).squeeze(0).cpu().numpy())
        self.real_labels.extend(y.cpu().numpy())
        loss = self.loss_fn(y_hat, y)
        wandb.log({"Loss": loss})
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def fit(self, train_data, y_train, weights_train=None, epochs=1):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        
        train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
        wandb.login()
        wandb.init(project="higgsml")

        for epoch in range(self.epoch, self.epoch+epochs):

            for batch in train_dl:

                self.optimizer.zero_grad()
                loss = self.training_step(batch, 0)
                loss.backward()
                self.optimizer.step()
            print("Epoch: ", epoch)
            wandb.log({"Epoch": epoch})

        self.epoch += epochs
        
        preds = np.array(self.predictions)
        labels = np.array(self.real_labels)
        self.accuracy = np.mean(labels == preds)
        print("Training Accuracy: ", np.mean(labels == preds))
        wandb.log({"Training Accuracy": np.mean(labels == preds)})
        self.save_model(f"../ckpts/model-{self.epoch}.pth")


    def predict(self, test_data):
        x = self.scaler.transform(test_data)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        dl = DataLoader(TensorDataset(x), batch_size=512, shuffle=False)

        pred = []

        with torch.no_grad():
            for batch in dl:
                x = batch[0]
                pred.extend(self.forward(x).cpu().numpy())
            pred = np.array(pred)
            pred = pred[:, 1]
        return pred
    
    def save_model(self, path):
        print(f"Saving model to {path} with accuracy: {self.accuracy} at epoch {self.epoch}")
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_shape': self.input_shape,
            'accuracy': self.accuracy,
            'epoch': self.epoch
        }, path)

        self.predictions = []
        self.real_labels = []
    
    def load_model(self, path):
        print("Loading model from ", path)
        checkpoint = torch.load(path)
        self.input_shape = checkpoint['input_shape']
        print(f"Using {self.input_shape} features")
        self.in1 = nn.Linear(self.input_shape, 500)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.accuracy = checkpoint['accuracy']
        print("loaded model with accuracy: ", self.accuracy)
        self.epoch = checkpoint['epoch']
        print("Starting from epoch: ", self.epoch)
        self.to(self.device)
        self.eval()
        print("Finished loading model")


# [[d1], [d2], [d3], [d4], [d5], [d6], [d7], [d8], [d9], [d10]]
# [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.0, 1.0]]
# [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]