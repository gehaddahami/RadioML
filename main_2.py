#%%
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn 
import torch.nn as nn

# Importing the model and the data loader 
from data.data_loading import Radioml_18
from model.model import QuantizedRadioml
from utils.loops import train_loop, test_loop, display_loss



#%% Loading the data and preparing the dataloaders
dataset_path =  "/home/student/Downloads/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
dataset = Radioml_18(dataset_path)

batchsize = 1024 
train_loader = DataLoader(dataset, batch_size= batchsize, sampler= dataset.train_sampler) 
validation_loader = DataLoader(dataset, batch_size= batchsize, sampler= dataset.validation_sampler)
test_loader = DataLoader(dataset, batch_size= batchsize, sampler= dataset.test_sampler)


#%% creating the model instance
# Parameters of the model
model_config = {
            "input_length": 2,
            "hidden_layers": [64] * 7 + [128] * 2,
            "output_length": 24,
            "input_bitwidth": 8,
            "hidden_bitwidth": 8,
            "output_bitwidth": 8,
            "input_fanin": 3, 
            "conv_fanin": 2, 
            "hidden_fanin": 50,
            "output_fanin": 128
        } 

model = QuantizedRadioml(model_config=model_config)
model


#%% Main loop 
num_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)  # Example scheduler

train_data = train_loader
validation_data = validation_loader

running_loss = []
accuracy = []
for epoch in range(num_epochs):
    # Set the model to training mode
    loss_epoch = train_loop(model, train_data, optimizer, criterion)
    test_acc, predictions, labels = test_loop(model, validation_data)
    print("Epoch %d: Training loss = %f, validation accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
    running_loss.append(loss_epoch)
    accuracy.append(test_acc)
    
    # Step the scheduler
    scheduler.step()

# Optionally, plot the running loss and accuracy
display_loss(running_loss)