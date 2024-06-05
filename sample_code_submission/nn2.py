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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier

    """

    def __init__(self, train_data):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        
        
        self.in1 = nn.Linear(train_data.shape[1], 500)
        self.h1 = nn.Linear(500, 500)
        self.h2 = nn.Linear(500, 500)
        self.out = nn.Linear(500, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        self.scaler = StandardScaler()
        self.predictions = []
        self.real_labels = []

        self.to(self.device)
    
    def forward(self, x):
        x = self.relu(self.in1(x))
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
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

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        
        train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
        wandb.login()
        wandb.init(project="higgsml")

        epochs = 1
        for epoch in range(epochs):
            for batch in train_dl:

                self.optimizer.zero_grad()
                loss = self.training_step(batch, 0)
                loss.backward()
                self.optimizer.step()
            print("Epoch: ", epoch)
            wandb.log({"Epoch": epoch})
        
        preds = np.array(self.predictions)
        labels = np.array(self.real_labels)
        print("Training Accuracy: ", np.mean(labels == preds))
        wandb.log({"Training Accuracy": np.mean(labels == preds)})

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self(test_data).argmax(dim = 1).cpu().numpy()
        
        print("PREDS", pred.shape)
        return pred


# [[d1], [d2], [d3], [d4], [d5], [d6], [d7], [d8], [d9], [d10]]
# [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.0, 1.0]]
# [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]