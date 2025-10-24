import pdb
import itertools
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
import random
import copy
from tqdm import tqdm

sys.path.append('../')
from meta_fusion.utils import *


# Train the cooperative learning method for binary classification
def train_coop_bc(model1, model2, model1_optim, model2_optim,
               criterion, train_loader, val_loader, 
               niter=2, epoch = 5,
               progress=True, verbose=True, random_state=0):
    set_random_seed(random_state)
    model1.train()
    model2.train()
    iterator = tqdm(range(niter)) if progress else range(niter)
    
    # Assuming train_loader is your DataLoader
    mod1 = []
    mod2 = []
    target = []

    for batch_mod1, batch_mod2, batch_target in train_loader:
        mod1.append(batch_mod1)
        mod2.append(batch_mod2)
        target.append(batch_target)

    mod1 = torch.cat(mod1)
    mod2 = torch.cat(mod2)
    target = torch.cat(target)

    val_mod1 = []
    val_mod2 = []
    val_target = []
    
    for batch_mod1, batch_mod2, batch_target in val_loader:
        val_mod1.append(batch_mod1)
        val_mod2.append(batch_mod2)
        val_target.append(batch_target)

    val_mod1 = torch.cat(val_mod1)
    val_mod2 = torch.cat(val_mod2)
    val_target = torch.cat(val_target)

    if verbose:
        print("Training cooperative learning model...")

    best_val_mis = float('inf')
    best_model1 = None
    best_model2 = None

    alphalist = [0.0, 0.5, 1.0, 3.0, 5.0, 9.0, 15.0, 20.0]
    for i, alpha in enumerate(alphalist):
        if verbose:
            print("Training with agreement penalty:", alpha,".")
        
        # Deepcopy the models to start from scratch
        model1_copy = copy.deepcopy(model1)
        model2_copy = copy.deepcopy(model2)
        model1_optim_copy = copy.deepcopy(model1_optim)
        model2_optim_copy = copy.deepcopy(model2_optim)

        f2 = torch.zeros(len(target))
        for _ in range(niter):
            r1 = target / (1 + alpha) - (1 - alpha) * f2 / (1 + alpha)
            for _ in range(epoch):
                model1_optim_copy.zero_grad()
                outputs = model1_copy(mod1)
                model1_loss = criterion(outputs, r1)
                model1_loss.backward()
                model1_optim_copy.step()

            with torch.no_grad():
                f1 = model1_copy(mod1)

            r2 = target / (1 + alpha) - (1 - alpha) * f1 / (1 + alpha)
            for _ in range(epoch):
                model2_optim_copy.zero_grad()
                outputs = model2_copy(mod2)
                model2_loss = criterion(outputs, r2)
                model2_loss.backward()
                model2_optim_copy.step()

            with torch.no_grad():
                f2 = model2_copy(mod2)
        
        with torch.no_grad():
            yhat_val = model1_copy(val_mod1) + model2_copy(val_mod2)
            yhatc_val = (yhat_val > torch.mean(yhat_val)).int()
            val_mis = torch.mean((yhatc_val != val_target).float()).item()

        # Track the best models based on validation misclassification
        if val_mis < best_val_mis:
            best_val_mis = val_mis
            best_model1 = copy.deepcopy(model1_copy)
            best_model2 = copy.deepcopy(model2_copy)

        if verbose:
            print(f"Best validation misclassification: {best_val_mis}")

        return best_model1, best_model2, best_val_mis




# Train the cooperative learning method for regression
def train_coop(model1, model2, model1_optim, model2_optim,
               train_loader, val_loader, 
               niter=2, epoch = 5,
               progress=True, verbose=True, random_state=0):
    
    # Save the initial state of the models
    initial_state_model1 = copy.deepcopy(model1.state_dict())
    initial_state_model2 = copy.deepcopy(model2.state_dict())
    
    initial_state_optim1 = copy.deepcopy(model1.optim.state_dict())
    initial_state_optim2 = copy.deepcopy(model2.optim.state_dict())

    alphalist = [0.0, 0.5, 1.0, 3.0, 5.0, 9.0, 15.0, 20.0]
    criterion = nn.MSELoss()

    set_random_seed(random_state)
    model1.train()
    model2.train()
    iterator = tqdm(range(niter)) if progress else range(niter)

    mod1 = []
    mod2 = []
    target = []

    for batch_mod1, batch_mod2, batch_target in train_loader:
        mod1.append(batch_mod1)
        mod2.append(batch_mod2)
        target.append(batch_target)

    # Prepare the data
    mod1 = torch.cat([batch_mod1 for batch_mod1, _, _ in train_loader])
    mod2 = torch.cat([batch_mod2 for _, batch_mod2, _ in train_loader])
    target = torch.cat([batch_target for _, _, batch_target in train_loader])

    val_mod1 = torch.cat([batch_mod1 for batch_mod1, _, _ in val_loader])
    val_mod2 = torch.cat([batch_mod2 for _, batch_mod2, _ in val_loader])
    val_target = torch.cat([batch_target for _, _, batch_target in val_loader])


    if verbose:
        print("Training cooperative learning model...")


    val_mse_list = []
    best_val_mse = float('inf')
    best_state_model1 = None
    best_state_model2 = None

    print("iterations:", niter, "epochs:", epoch)
    for i, alpha in enumerate(alphalist):
        if verbose:
            print("Training with agreement penalty:", alpha,".")
        
        # Load the initial state to reset the models
        model1.load_state_dict(initial_state_model1)
        model2.load_state_dict(initial_state_model2)
        
        model1_optim = load_state_dict(initial_state_optim1)
        model2_optim = load_state_dict(initial_state_optim2)

        f2 = torch.zeros(len(target)).reshape(-1,1)
        for iteration in range(niter):
            if verbose:
                print(f"Training with the {iteration}th iteration...")
            
            r1 = target / (1 + alpha) - (1 - alpha) * f2 / (1 + alpha)
            
            for epoch_num in range(epoch):
                model1_optim.zero_grad()
                outputs = model1(mod1)
                model1_loss = criterion(outputs, r1)
                model1_loss.backward()
                model1_optim.step()

            with torch.no_grad():
                f1 = model1(mod1)

            r2 = target / (1 + alpha) - (1 - alpha) * f1 / (1 + alpha)
            for epoch_num in range(epoch):
                model2_optim.zero_grad()
                outputs = model2(mod2)
                model2_loss = criterion(outputs, r2)
                model2_loss.backward()
                model2_optim.step()

            with torch.no_grad():
                f2 = model2(mod2)
        
        with torch.no_grad():
            yhat_val = model1(val_mod1) + model2(val_mod2)
            val_mse = criterion(yhat_val, val_target)
        
        val_mse_list.append(val_mse)

        # Track the best models based on validation misclassification
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state_model1 = copy.deepcopy(model1.state_dict())  
            best_state_model2 = copy.deepcopy(model2.state_dict())

    if verbose:
        print(f"Best validation MSE: {best_val_mse}")
        print(f"Validation MSE history: {val_mse_list}")
    
    # Load the best model states before returning them
    model1.load_state_dict(best_state_model1)
    model2.load_state_dict(best_state_model2)

    return model1, model2, alpha, val_mse_list










