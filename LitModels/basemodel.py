import torch
import lightning as L
from utils.metrics import Metrics
import numpy as np
from torch import nn

class BaseModel(L.LightningModule):
    def __init__(self, 
                 model:nn.Module=nn.Identity(), 
                 # training
                 lr=0.0001, 
                 eta_min=0.0, 
                 max_epoch=10, 
                 steps_per_epoch=100, 
                 loss_type:str="MAE",
                 metrics:Metrics=Metrics(),
                 # testing
                 muti_steps:int=1):
        
        super().__init__()
        self.model = model
        self.lr = lr
        self.eta_min = eta_min
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = (steps_per_epoch+1)*max_epoch
        self.loss_type = loss_type
        self.choose_loss(loss_type)
        self.metrics = metrics
        # ============ Test ============
        self.muti_steps = muti_steps
        self.test_metrics = {}
        # ==============================
        self.example_input_array = torch.Tensor(1, 69, 128, 256)
        self.save_hyperparameters(ignore=['model', 'metrics', 'muti_steps_reader']) # too big too save


    def mae_loss(self, pred, tar):
        return torch.mean(torch.abs(pred-tar))
    
    def mse_loss(self, pred, tar):
        return torch.mean((pred-tar)**2)

    def choose_loss(self, type):
        if type == "MAE":
            self.loss = self.mae_loss
        elif type == "MSE":
            self.loss = self.mse_loss


    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        lr_now = self.the_optimizer.param_groups[0]['lr']
        log_dict = {"train_loss": loss, "lr": lr_now}
        self.log_dict(log_dict, prog_bar=True)
        return loss
    
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        rmse = self.metrics.WRMSE(y_hat, y)
        log_dict = {"val_loss": val_loss, "val_RMSE_z500": rmse[11]}
        self.log_dict(log_dict, prog_bar=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.9))
        self.the_optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_steps, eta_min=self.eta_min)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }
        
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler_dict}
  