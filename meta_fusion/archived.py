import torch.nn as nn
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
import random
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import os
import time
import copy


sys.path.append('../')
from meta_fusion.utils import *
from meta_fusion.models import MLP_Net


def get_combined_nets(mod1_outs, mod2_outs, output_dim):
    # Create the combined nets, including cases where one modality is None
    combined_nets = []

    # Case where both mod1_out and mod2_out are used
    for mod1_out, mod2_out in itertools.product(mod1_outs, mod2_outs):
        input_dim = mod1_out + mod2_out
        hidden_dim = int((mod1_out + mod2_out) / 2)
        combined_nets.append(MLP_Net(input_dim, [hidden_dim], output_dim))

    # Case where only mod1_out is used
    for mod1_out in mod1_outs:
        input_dim = mod1_out
        hidden_dim = int(mod1_out / 2)
        combined_nets.append(MLP_Net(input_dim, [hidden_dim], output_dim))

    # Case where only mod2_out is used
    for mod2_out in mod2_outs:
        input_dim = mod2_out
        hidden_dim = int(mod2_out / 2)
        combined_nets.append(MLP_Net(input_dim, [hidden_dim], output_dim))

    return combined_nets



class MetaExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dims):
        super(MetaExtractor, self).__init__()
        
        # Ensure hidden_dims and output_dims have the same length
        assert len(hidden_dims) == len(output_dims), "hidden_dims and output_dims must have the same length"
        
        # Create multiple hidden layers and corresponding output layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for hidden_dim in hidden_dims
        ])
        
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for hidden_dim, output_dim in zip(hidden_dims, output_dims)
        ])
        
    def forward(self, x):
        outputs = []
        for hidden_layer, output_layer in zip(self.hidden_layers, self.output_layers):
            hidden_output = F.relu(hidden_layer(x))
            output = output_layer(hidden_output)
            outputs.append(output)
        return outputs



class MetaJoint(nn.Module):
    def __init__(self, meta_extractor_mod1, meta_extractor_mod2, combined_nets):
        super(MetaJoint, self).__init__()
        self.meta_extractor_mod1 = meta_extractor_mod1
        self.meta_extractor_mod2 = meta_extractor_mod2
        self.combined_nets = nn.ModuleList(combined_nets)
        self.dimension_info = []

    def forward(self, mod1, mod2):
        mod1_feature_list = self.meta_extractor_mod1(mod1)
        mod2_feature_list = self.meta_extractor_mod2(mod2)
        
        outputs = []
        dimension_info = []

        # Generate all possible combinations of features
        feature_pairs = (
            [(m1, m2) for m1, m2 in itertools.product(mod1_feature_list, mod2_feature_list)] +
            [(m1, None) for m1 in mod1_feature_list] +
            [(None, m2) for m2 in mod2_feature_list]
        )

        for i, (mod1_features, mod2_features) in enumerate(feature_pairs):
            if mod1_features is not None and mod2_features is not None:
                mod1_dim = mod1_features.shape[1]
                mod2_dim = mod2_features.shape[1]
                combined_input = torch.cat((mod1_features, mod2_features), dim=1)
            elif mod1_features is not None:
                mod1_dim = mod1_features.shape[1]
                mod2_dim = 0
                combined_input = mod1_features
            elif mod2_features is not None:
                mod1_dim = 0
                mod2_dim = mod2_features.shape[1]
                combined_input = mod2_features

            output = self.combined_nets[i](combined_input)
            outputs.append(output)
            self.dimension_info.append((mod1_dim, mod2_dim))

        return outputs









#-------------------------------------#
#------old actor critic models--------#
#-------------------------------------#

# Training the neural network
def train_actors_critic(actor_model, critic_model, train_loader, criterion, 
                        actor_optim, critic_optim, epochs=50, progress=True, verbose=True, radnom_state=0):
    set_random_seed(random_state)
    actor_model.train()
    critic_model.train()
    iterator = tqdm(range(epochs)) if progress else range(epochs)

    # Train the actor models
    for epoch in iterator:
        total_actor_loss = 0.0
        for mod1, mod2, target in train_loader:
            
            if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
                mod1 = mod1.view(mod1.size(0), -1)

            # Update actor models
            actor_optim.zero_grad()
            actor_outputs = actor_model(mod1, mod2)
            losses = []
            for output in actor_outputs:
                loss = criterion(output, target)
                losses.append(loss)
                
            # Calculate total loss
            actor_loss = sum(losses)
            actor_loss.backward()
            actor_optim.step()
            total_actor_loss += actor_loss.item()

            # # Update the critic model
            # critic_optim.zero_grad()
            # with torch.no_grad():
            #     combined_prob = torch.cat(actor_outputs, dim=1)
            # critic_output = critic_model(combined_prob)
            # critic_loss = criterion(critic_output, target)
            # critic_loss.backward()
            # critic_optim.step()
            # total_critic_loss += critic_loss.item()
        if verbose:
            avg_actor_loss = total_actor_loss / (len(train_loader) * len(losses))
            print(f'Epoch {epoch+1}/{epochs}, Average Actor Loss: {avg_actor_loss:.4f}')

        # Train the meta-learning critic model
    for epoch in iterator:
        total_critic_loss = 0.0
        for mod1, mod2, target in train_loader:

            if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
                mod1 = mod1.view(mod1.size(0), -1)

            # Update the critic model
            critic_optim.zero_grad()
            with torch.no_grad():
                actor_outputs = actor_model(mod1, mod2)
                combined_prob = torch.cat(actor_outputs, dim=1)
            critic_output = critic_model(combined_prob)
            critic_loss = criterion(critic_output, target)
            critic_loss.backward()
            critic_optim.step()
            total_critic_loss += critic_loss.item()
        if verbose:
            avg_actor_loss = total_actor_loss / (len(train_loader) * len(losses))
            avg_critic_loss = total_critic_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Average Actor Loss: {avg_actor_loss:.4f}')
            print(f'Epoch {epoch+1}/{epochs}, Average Critic Loss: {avg_critic_loss:.4f}')
            


# # Train the meta-fusion actor models
# def train_actors(actor_model, actor_optim, 
#                  train_loader, criterion, 
#                  epochs=50, progress=True, 
#                  verbose=True, random_state=0):

#     set_random_seed(random_state)
#     actor_model.train()

#     if verbose:
#         print("Training actor models...")

#     iterator = tqdm(range(epochs)) if progress else range(epochs)

#     # Train the actor models
#     for epoch in iterator:
#         total_actor_loss = 0.0
#         for mod1, mod2, target in train_loader:
            
#             if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
#                 mod1 = mod1.view(mod1.size(0), -1)

#             # Update actor models
#             actor_optim.zero_grad()
#             actor_outputs = actor_model(mod1, mod2)
#             losses = []
#             for output in actor_outputs:
#                 loss = criterion(output, target)
#                 losses.append(loss)
                
#             # Calculate total loss
#             actor_loss = sum(losses)
#             actor_loss.backward()
#             actor_optim.step()
#             total_actor_loss += actor_loss.item()

#         if verbose:
#             avg_actor_loss = total_actor_loss / (len(train_loader) * len(losses))
#             print(f'Epoch {epoch+1}/{epochs}, Average Actor Loss: {avg_actor_loss:.4f}')

#     if verbose:
#         print("Traning complete!")



def train_actors(actor_model, actor_optim, 
                 train_loader, val_loader, criterion, 
                 epochs=50, progress=True, 
                 verbose=True, random_state=0):

    set_random_seed(random_state)
    actor_model.train()

    if verbose:
        print("Training actor models...")

    iterator = tqdm(range(epochs)) if progress else range(epochs)

    best_val_loss = float('inf')
    best_model_state = None
    
    training_losses = []
    validation_losses = []

    for epoch in iterator:   
        # Training phase
        total_actor_loss = 0.0
        actor_model.train()  
        for mod1, mod2, target in train_loader:             
            if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
                mod1 = mod1.view(mod1.size(0), -1)

            # Update actor models
            actor_optim.zero_grad()
            actor_outputs = actor_model(mod1, mod2)
            losses = []
            for output in actor_outputs:
                loss = criterion(output, target)
                losses.append(loss)
                
            # Calculate total loss
            actor_loss = sum(losses)
            actor_loss.backward()
            actor_optim.step()
            total_actor_loss += actor_loss.item()

        # Calculate average loss for the epoch
        avg_actor_loss = total_actor_loss / (len(train_loader) * len(losses))
        training_losses.append(avg_actor_loss)

        # Validation phase
        actor_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mod1, mod2, target in val_loader:
                if mod1.dim() > 2:
                    mod1 = mod1.view(mod1.size(0), -1)

                actor_outputs = actor_model(mod1, mod2)
                losses = []
                for output in actor_outputs:
                    loss = criterion(output, target)
                    losses.append(loss)

                val_loss += sum(losses).item()

        # Calculate average validation loss
        avg_val_loss = val_loss / (len(val_loader) * len(losses))
        validation_losses.append(avg_val_loss)
        
        # Check if this is the best validation performance so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = actor_model.state_dict()  # Save the best model's state


        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Average Actor Loss: {avg_actor_loss:.4f}, '
                  f'Validation Loss: {avg_val_loss:.4f}')

    # Restore the best model state
    actor_model.load_state_dict(best_model_state)

    if verbose:
        print("Training complete! Best validation loss: {:.4f}".format(best_val_loss))

    return actor_model, training_losses, validation_losses


            
def train_critic(actor_model, critic_model, critic_optim, 
                 train_loader, val_loader, criterion, 
                 epochs=50, progress=True, 
                 verbose=True, random_state=0):

    set_random_seed(random_state)
    actor_model.eval()
    critic_model.train()
    iterator = tqdm(range(epochs)) if progress else range(epochs)

    best_val_loss = float('inf')
    best_model_state = None

    training_losses = []
    validation_losses = []

    if verbose:
        print("Training meta-learner critic...")

    # Train the meta-learning critic model
    for epoch in iterator:
        
        # Training phase
        critic_model.train()
        total_critic_loss = 0.0
        for mod1, mod2, target in train_loader:
            
            if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
                mod1 = mod1.view(mod1.size(0), -1)

            # Update the critic model
            critic_optim.zero_grad()
            with torch.no_grad():
                actor_outputs = actor_model(mod1, mod2)
                combined_features = torch.cat(actor_outputs, dim=1)  # Combine outputs from the actor model

            critic_output = critic_model(combined_features)
            critic_loss = criterion(critic_output, target)
            critic_loss.backward()
            critic_optim.step()
            total_critic_loss += critic_loss.item()

        avg_critic_loss = total_critic_loss / len(train_loader)
        training_losses.append(avg_critic_loss)

        # Validation phase
        critic_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mod1, mod2, target in val_loader:
                if mod1.dim() > 2:
                    mod1 = mod1.view(mod1.size(0), -1)

                actor_outputs = actor_model(mod1, mod2)
                combined_features = torch.cat(actor_outputs, dim=1)  # Combine outputs from the actor model
                critic_output = critic_model(combined_features)
                loss = criterion(critic_output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Check if this is the best validation performance so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = critic_model.state_dict()  # Save the best model's state

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Average Critic Loss: {avg_critic_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Restore the best model state
    critic_model.load_state_dict(best_model_state)

    if verbose:
        print("Training complete! Best validation loss: {:.4f}".format(best_val_loss))

    return critic_model, training_losses, validation_losses



# Evaluating the neural network
def test_actors_critic(actor_model, n_actors, critic_methods, test_loader, 
                     verbose=True, **kwargs):
    actor_model.eval()
    with torch.no_grad():
        method_accuracies = {critic_method: 0 for critic_method in critic_methods}
        actor_correct = np.zeros(n_actors, dtype=int)

        for mod1, mod2, target in test_loader:
            
            if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
                mod1 = mod1.view(mod1.size(0), -1)

            # Evaluate the actor model
            actor_outputs = actor_model(mod1, mod2)

            for i, output in enumerate(actor_outputs):
                actor_pred = output.argmax(dim=1, keepdim=True)
                actor_correct[i] += actor_pred.eq(target.view_as(actor_pred)).sum().item()


            # Evaluate the critic models
            for critic_method in critic_methods:
                if critic_method == "Meta-learner":
                    # Expecting 'meta_learner' to have a model passed via kwargs
                    meta_model = kwargs.get('meta_learner_model')
                    combined_prob = torch.cat(actor_outputs, dim=1)
                    critic_output = meta_model(combined_prob)
                    critic_pred = critic_output.argmax(dim=1, keepdim=True)

                elif critic_method == "Majority-vote":
                    # Majority vote: Each base learner votes for a class, the majority vote wins
                    votes = torch.stack([output.argmax(dim=1) for output in actor_outputs], dim=1)
                    critic_pred = torch.mode(votes, dim=1)[0].view(-1, 1)

                elif critic_method == "Simple-averaging":
                    # Simple averaging: Average the probabilities across all base learners
                    average_prob = torch.mean(torch.stack(actor_outputs, dim=1), dim=1)
                    critic_pred = average_prob.argmax(dim=1, keepdim=True)
                
                method_accuracies[critic_method] += critic_pred.eq(target.view_as(critic_pred)).sum().item()

        method_accuracies = {key: value / len(test_loader.dataset)  for key, value in method_accuracies.items()}
        actor_accuracy = actor_correct / len(test_loader.dataset)
        method_accuracies["Actor models"] = actor_accuracy
        if verbose:
            for key, value in method_accuracies.items():
                print(f'Method: ({key}), Test Accuracy: {value}')
        return method_accuracies



# Evaluating the neural network for regression
def test_actors_critic_regression(actor_model, n_actors, critic_methods, test_loader, 
                                  criterion, verbose=True, **kwargs):
    actor_model.eval()
    with torch.no_grad():
        method_errors = {critic_method: 0.0 for critic_method in critic_methods}
        actor_errors = np.zeros(n_actors, dtype=float)

        for mod1, mod2, target in test_loader:
            
            if mod1.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
                mod1 = mod1.view(mod1.size(0), -1)

            # Evaluate the actor model
            actor_outputs = actor_model(mod1, mod2)

            # Calculate errors for each actor
            for i, output in enumerate(actor_outputs):
                actor_errors[i] += criterion(output, target).item()

            # Evaluate the critic models
            for critic_method in critic_methods:
                if critic_method == "Meta-learner":
                    # Expecting 'meta_learner' to have a model passed via kwargs
                    meta_model = kwargs.get('meta_learner_model')
                    combined_features = torch.cat(actor_outputs, dim=1)
                    critic_output = meta_model(combined_features)
                    method_errors[critic_method] += criterion(critic_output, target).item()

                elif critic_method == "Simple-averaging":
                    # Simple averaging: Average the outputs across all base learners
                    average_output = torch.mean(torch.stack(actor_outputs, dim=1), dim=1)
                    method_errors[critic_method] += criterion(average_output, target).item()

                elif critic_method == "Weighted-averaging":
                    # Weighted averaging: Weighted average of outputs based on actor performance
                    inverse_errors = 1.0 / (actor_errors + 1e-8)  # Add a small value to avoid division by zero
                    weights = inverse_errors / inverse_errors.sum()
                    weighted_output = sum(weight * output for weight, output in zip(weights, actor_outputs))
                    method_errors[critic_method] += criterion(weighted_output, target).item()

        # Normalize errors by the number of samples
        method_errors = {key: value / len(test_loader) for key, value in method_errors.items()}
        actor_errors = actor_errors / len(test_loader)

        method_errors["Actors"] = actor_errors
        if verbose:
            for key, value in method_errors.items():
                print(f'Method: ({key}), Test Error: {value}')
        return method_errors





#-------------------------------------#
#------old benchmark methods----------#
#-------------------------------------#

def train_bm(method, model, train_loader, val_loader, criterion, optimizer, epochs=10, progress=True, verbose=True):
    best_val_loss = float('inf')
    best_model_state = None

    training_losses = []
    validation_losses = []

    iterator = tqdm(range(epochs)) if progress else range(epochs)
    for epoch in iterator:
        # Training phase
        model.train()
        total_train_loss = 0.0
        for mod1, mod2, target in train_loader:
            optimizer.zero_grad()
            if method == "Mod1":
                output = model(mod1)
            elif method == "Mod2":
                output = model(mod2)
            elif method == "Early":
                output = model(torch.cat((mod1, mod2), dim=1))
            elif method == 'Joint':
                output = model(mod1, mod2)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mod1, mod2, target in val_loader:
                if method == "Mod1":
                    output = model(mod1)
                elif method == "Mod2":
                    output = model(mod2)
                elif method == "Early":
                    output = model(torch.cat((mod1, mod2), dim=1))
                elif method == 'Joint':
                    output = model(mod1, mod2)

                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save the best model's state

        if verbose:
            print(f'Method ({method}) Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    
    model.load_state_dict(best_model_state)
    return model, training_losses, validation_losses



# Evaluating the neural network
def test_bm(method, model, test_loader, criterion, verbose=True):
    if method == 'Coop':
        model[0].eval()
        model[1].eval()
    else:
        model.eval()
        
    total_loss = 0.0
    with torch.no_grad():
        for mod1, mod2, target in test_loader:
            if method == "Mod1":
                output = model(mod1)
            elif method == "Mod2":
                output = model(mod2)
            elif method == "Early":
                output = model(torch.cat((mod1, mod2), dim=1))
            elif method == 'Joint':
                output = model(mod1, mod2)
            elif method == "Coop":
                model1, model2 = model[0], model[1]
                output = model1(mod1) + model2(mod2)
                        
            loss = criterion(output, target)
            total_loss += loss.item()

    # Calculate the average loss over the test dataset
    average_loss = total_loss / len(test_loader)
    
    if verbose:
        print(f'Method ({method}) Test Loss: {average_loss:.4f}')
    
    return average_loss




class Trainer_all_at_once(Trainer):
    def __init__(self, config, models, data_loaders):
        super().__init__(config, models, data_loaders)

        # Initialize the meta-learner and optimizer
        self.initialize_meta_learner()



    def initialize_meta_learner(self):
        """
        Initialize the meta-learner model which by default is a shallow MLP with one single hidden layer.
        """
        if self.task_type == "classification":
            input_dim = int(self.model_num * self.num_classes)
            output_dim = int(self.num_classes)
        else:
            input_dim = int(self.model_num)
            output_dim = 1
        #hidden_dim = int(input_dim/2)
        hidden_dim = int(input_dim)

        meta_learner = MLP_Net(input_dim, [hidden_dim], output_dim)
        optimizer = torch.optim.Adam(meta_learner.parameters(), lr=0.1, weight_decay=0)

        self.meta_learner = meta_learner
        self.meta_learner_optimizer = optimizer

        self.initial_meta_learner_state = copy.deepcopy(meta_learner.state_dict())
        self.initial_meta_optimizer_state = copy.deepcopy(optimizer.state_dict())


    
    def train(self):
        """
        Main training function that performs the following:
        1. trains benchmark models and a list of rhos provided by the user;
        2. chooses and record the best rho via cross-validation.
        """
        
        print("Start training benchmark models...")
        super().train_all_rhos()
        print("Finished training benchmark models!")

        print("Selecting the optimal disgreement penalty via cross-validation...")
        super().choose_best_rho()
        print("Done!")



    def test(self, test_loader):
        """
        Main testing function that performs the following:
        1. Load the best cohort chosen by cross-validation;
        2. Evaluate the method performance on the test data.
        """

        # Make sure the best model is loaded
        _ = self.load_checkpoints(self.best_rho)
        _ = self.load_meta_learner(self.best_rho)

        if self.task_type == "classification":
            results = self.test_classification(test_loader)
        else:
            results = self.test_regression(test_loader)
        
        return results



    #-----------------------#
    #----- Test Helpers ----#
    #-----------------------#
    def test_regression(self, test_loader):
        """
        Evaluate the performance of the ensemble on the validation set for regression tasks.
        """

        method_mse = AverageMeter()

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode

        self.meta_learner.eval()

        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(test_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(mod1, mod2)
                    outputs.append(output)

                outputs_concat = torch.cat(outputs, dim=1)
                final_output = self.meta_learner(outputs_concat)

                method_mse.update(self.loss_mse(final_output, target), target.size()[0])
            
            mse_value = method_mse.avg
            if hasattr(mse_value, 'item'):
                mse_value = mse_value.item()
            else:
                mse_value =  float(mse_value)

            if self.verbose:
                print(f'Method: ({"all_at_once"}), Test_MSE: {mse_value}')

            #return {"all_at_once":method_mse.avg}
            return {"all_at_once":mse_value}



    def choose_best_rho(self):
        """
        Select rho such that the valiadation loss of meta-learner is the smallest
        """
        
        best_rho = None
        best_val_loss = float('inf')

        # Loop over each rho value and validate
        for rho in self.rho_list:
            
            # Load the saved models for this rho
            val_loss = self.load_meta_learner(rho)
        
            # Check for best performance (minimize loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_rho = rho
            
        # Print the best rho and performance for the current method
        print(f"Best rho: {best_rho} with val loss: {best_val_loss:.4f}")

        self.best_rho = best_rho



    def train_all_rhos(self):
        for rho in self.rho_list:
            print(f"Training with disagreement penalty = {rho}")
            
            # Reset models, optimizers, and schedulers to their initial states
            for i in range(self.model_num):
                self.models[i].load_state_dict(self.initial_model_states[i])
                self.optimizers[i].load_state_dict(self.initial_optimizer_states[i])
            
            self.meta_learner.load_state_dict(self.initial_meta_learner_state)
            self.meta_learner_optimizer.load_state_dict(self.initial_meta_optimizer_state)

            # Call the train function to train with the current rho
            self.train_fixed_rho(rho)



    def train_fixed_rho(self, rho):
        """
        Train the models for a fixed level of disagreement penalty. 
        """
        
        # Variable to keep track of the best validation loss
        self.best_val_loss = float('inf')

        for epoch in range(self.epochs):
    
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )

            # train for 1 epoch
            train_loss = self.train_one_epoch(epoch, rho)

            # evaluate on validation set
            val_loss = self.validate(epoch)

            # Check if the current model has the best validation task loss
            is_best = val_loss.avg < self.best_val_loss
            
            # Print train and validation losses
            msg = "- train loss: {:.3f}, val loss: {:.3f}"
            if is_best:
                msg += " [*] Best so far"
            
            if self.verbose:
                print(
                    msg.format(train_loss.avg, val_loss.avg)
                )

            # If this is the best model so far, update best loss and save checkpoint
            if is_best:
                self.best_val_loss = val_loss.avg
                for i in range(self.model_num):
                    self.save_checkpoint(rho, i,
                        {
                            "epoch": epoch + 1,
                            "disagreement_penalty": rho,
                            "model_state": self.models[i].state_dict(),
                            "optim_state": self.optimizers[i].state_dict(),
                            "best_val_task_loss": self.best_val_loss,
                        }
                    )
                self.save_meta_learner(rho,
                    {
                    'epoch': epoch + 1,
                    'model_state': self.meta_learner.state_dict(),
                    'optimizer_state': self.meta_learner_optimizer.state_dict(),
                    'best_val_task_loss': self.best_val_loss
                    }
                )




    def train_one_epoch(self, epoch, rho):
        """
        Train the model for 1 epoch of the training set.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        meta_loss = AverageMeter()

        # Add loss meter for meta-learner
        losses = []
        task_losses = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            task_losses.append(AverageMeter())


        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch, (mod1, mod2, target) in enumerate(self.train_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
                
                # Temporarily freeze the meta-learner parameters to avoid updating them
                for param in self.meta_learner.parameters():
                    param.requires_grad = False

                # Forward pass
                for i in range(self.model_num):
                    outputs = [[]] * self.model_num
                    outputs[i] = self.models[i](mod1, mod2)
                    divergence_loss = 0
                    for j in range(self.model_num):
                        if i != j:
                            outputs[j] = self.models[j](mod1, mod2).detach()
                            if self.task_type == "regression":
                                divergence_loss = divergence_loss + self.loss_mse(outputs[i], outputs[j])
                            else:
                                divergence_loss = divergence_loss +  self.loss_kl(F.log_softmax(outputs[i], dim = 1), 
                                                                                  F.softmax(outputs[j], dim=1))
                    
                    final_output = self.meta_learner(torch.cat(outputs, dim=1))
                    task_loss = self.loss_task(final_output, target)
                    loss = task_loss + rho * divergence_loss / (self.model_num - 1)
                    
                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

                    # compute gradients and update optimizer
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    #self.schedulers[i].step()
                
                # Unfreeze the meta-learner parameters for updating
                for param in self.meta_learner.parameters():
                    param.requires_grad = True

                self.meta_learner.train()

                # Detach all student model outputs and pass them to the meta-learner
                outputs = []
                for i in range(self.model_num):
                    outputs.append(self.models[i](mod1, mod2).detach()) 

                final_output = self.meta_learner(torch.cat(outputs, dim=1))
                meta_task_loss = self.loss_task(final_output, target)

                self.meta_learner_optimizer.zero_grad()
                meta_task_loss.backward()
                self.meta_learner_optimizer.step()

                # Record meta-learner loss
                meta_loss.update(meta_task_loss.item(), target.size(0))

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                # pbar.set_description(
                #     (
                #         "{:.1f}s - meta_task_loss: {:.3f}".format(
                #             (toc-tic), meta_loss.avg
                #         )
                #     )
                # )
                self.batch_size = target.shape[0]
                pbar.update(self.batch_size)

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + batch
                    for i in range(self.model_num):
                        log_value('train_loss_%d' % (i+1), losses[i].avg, iteration)
                        log_value('train_task_loss_%d' % (i+1), task_losses[i].avg, iteration)
                    log_value('train_meta_task_loss', meta_loss.avg, iteration)

            return meta_loss



    def validate(self, epoch):
        """
        Evaluate the model on the validation set on the current model.
        """
        loss_meter = AverageMeter()

        for i in range(self.model_num):
            self.models[i].eval()

        self.meta_learner.eval()
        
        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(self.val_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                #forward pass
                outputs=[]
                for model in self.models:
                    outputs.append(model(mod1, mod2))
                
                final_output = self.meta_learner(torch.cat(outputs, dim=1))
                loss = self.loss_task(final_output, target)

                # Update loss meters
                loss_meter.update(loss.item(), target.size(0))

            # log to tensorboard for every epoch
            if self.use_tensorboard:
                for i in range(self.model_num):
                    log_value('val_meta_task_loss', loss_meter.avg, epoch+1)

        return loss_meter


    
    #-----------------------#
    #----- S&L Helpers -----#
    #-----------------------#
    def save_meta_learner(self, rho, state):
        """
        Save a copy of the meta learner so that it can be loaded at a future date. 
        """
    
        filename = "meta_learner" + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, str(rho), filename)
        
        # Get the directory path (parent directories)
        directory = os.path.dirname(ckpt_path)
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Save the checkpoint
        torch.save(state, ckpt_path)
    


    def load_meta_learner(self, rho):
        """
        Load a saved meta learner model from a checkpoint.
        """
    
        filename = "meta_learner" + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir,  str(rho), filename)

        checkpoint = torch.load(ckpt_path)
        self.meta_learner.load_state_dict(checkpoint['model_state'])
        self.meta_learner_optimizer.load_state_dict(checkpoint['optimizer_state'])
        #self.meta_learner_scheduler.load_state_dict(checkpoint['scheduler_state'])

        return checkpoint['best_val_task_loss']








# Following code is adapted from the following repository:
# https://github.com/dingdaisy/cooperative-learning/tree/main/cooperative_learning
class CoopLinear():
    """
    Implements the cooperative regularized linear regression provided by Algorithm 1 in the following paper:
    https://www.pnas.org/doi/full/10.1073/pnas.2202113119
    """
    def __init__(self, data_loaders, 
                 rho_list = None, nfolds = None,
                 random_state = 0, verbose = True):
        
        # data parameters
        self.train_loader = data_loaders[0]
        self.val_loader = data_loaders[1]
        
        if rho_list is None:
            # disagreement penalties used in the reference paper
            self.rho_list = np.array([0,0.2,0.4,0.6,0.8,1,3,5,9])
        else:
            self.rho_list = rho_list

        self.nfolds = 10 if not nfolds else nfolds

        # logging parameters
        self.random_state = random_state
        self.verbose = verbose



    def train(self):
        self.best_val_mse = float('inf')
        self.best_rho = None
        self.best_model = None

        for rho in self.rho_list:
            print(f"Fitting cooperative regularized linear regression with disagreement penalty = {rho}")

            coop_cv = self.train_fixed_rho(rho)
            val_mse = self.validate(coop_cv)

            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.best_rho = rho
                self.best_model = coop_cv
        
        if self.verbose:
            # Print the best rho and best mse
            print(f"Best rho: {self.best_rho} with val mse: {self.best_val_mse:.4f}")
        
    

    def test(self, test_loader):
        """
        Evaluate the best model performance on the test data
        """
        X1, X2, Y = load_all_data(test_loader)
        X = np.hstack((X1, X2))
        Y_pred = self.best_model.predict(X)
        mse = mean_squared_error(Y, Y_pred)

        if self.verbose:
            print(f'Method: ({"coop_linear"}), Test_MSE: {mse}')
 
        return {"coop_linear":mse}



    def train_fixed_rho(self, rho):
        X1, X2, Y = load_all_data(self.train_loader)

        # Create the augmented features and labels 
        X = np.vstack((np.hstack((X1, X2)), np.hstack((-np.sqrt(rho) * X1, np.sqrt(rho) * X2))))
        Y = np.concatenate((Y, np.zeros_like(Y))).ravel()

        # Use ElasticNetCV to automatically handle the cross-validation for regularization
        coop_cv = ElasticNetCV(cv = self.nfolds,
                               fit_intercept = True, 
                               random_state = self.random_state
                               ).fit(X, Y)

        return coop_cv


    
    def validate(self, model):
        """
        Evaluate the model performance on the validation data
        """
        X1, X2, Y = load_all_data(self.val_loader)

        X = np.hstack((X1, X2))
        Y_pred = model.predict(X)
        mse = mean_squared_error(Y, Y_pred)

        return mse 



    def get_penalties(self):
        """
        Get the disagreement penalty and the rugularization penalty. 
        Must be called after model fitting.
        """

        if not self.best_model:
            print("Best model not found, fit the models first!")
        else:
            alpha = self.best_model.alpha_
        
        return self.best_rho, alpha