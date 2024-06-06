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
from tqdm.auto import tqdm  # Importer tqdm pour la barre de progression


class NeuralNetwork(pl.LightningModule):
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def _init_(self, train_data):
        
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super()._init_()
        self.model = nn.Sequential(
            nn.Linear(train_data.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax()
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        self.scaler = StandardScaler()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        #y = y.type(torch.LongTensor).to(self.device)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def fit(self, train_data, y_train, weights_train=None):
    
        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
    
        train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=128*4, shuffle=True)
    
        # Utilisation de PyTorch Lightning avec Wandb
        wandb.login()
        wandb.init(project="higgs")
        wb_logger = WandbLogger(project="higgs")
        trainer = pl.Trainer(max_epochs=20, accelerator='auto', enable_progress_bar=False, logger=wb_logger)
        trainer.fit(self, train_dataloaders =train_dl)


    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        with torch.no_grad():  # Désactive le calcul des gradients
            output = self.model(test_data)
        return output.argmax(dim=1).numpy()  # Obtenir les classes prédites
