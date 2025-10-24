import pdb
import itertools
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import random
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import os
import time
import copy


sys.path.append('../')
from meta_fusion.utils import *
from meta_fusion.models import *
torch.autograd.set_detect_anomaly(True)


class Extractors:
    """
    This class defines the feature extractors for multiple tabular modalities.
    """
    def __init__(self, mod_outs, dims, train_loader, val_loader):
        """
        Args:
            mod_outs: List of lists, where each sublist contains the output dimensions for a modality.
            dims: List of integers, where each integer is the input dimension for a modality.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """
        self.mod_outs = [sorted(out, reverse=True) for out in mod_outs]
        self.dims = dims
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.extractors = []  # Initialize as an empty list


    def get_dummy_extractors(self):
        """
        Create dummy extractors for each modality.
        """
        
        err_msg = "Invalid output dimension, create dummy extractors only when output dimension is equal to 0 or the raw feature size."
        
        for mod_out, dim in zip(self.mod_outs, self.dims):
            mod_extractors = []
            for out_dim in mod_out:
                if out_dim == dim:
                    mod_extractors.append('full')
                elif out_dim == 0:
                    mod_extractors.append(None)
                else:
                    print(err_msg)
            self.extractors.append(mod_extractors)
        
        return self.extractors


    def get_PCA_extractors(self):
        """
        Dimension reduction by PCA for each modality.
        """
        # Load data for train_loader and process
        train_modalities, _ = load_all_data(self.train_loader)
        val_modalities, _ = load_all_data(self.val_loader)

        # Combine train and validation data for each modality
        combined_modalities = [
            np.concatenate((train_mod, val_mod), axis=0)
            for train_mod, val_mod in zip(train_modalities, val_modalities)
        ]

        # Initialize the extractors list
        self.extractors = []

        # Train PCA extractors for each combined modality
        for mod_data, mod_out, dim in zip(combined_modalities, self.mod_outs, self.dims):
            mod_extractors = self._train_pca_extractors(mod_data, mod_out, dim)
            self.extractors.append(mod_extractors)

        return self.extractors


    def _train_pca_extractors(self, X, output_dims, full_dim):
        """
        Train PCA models on the provided numpy data matrix with varying numbers of components.
        
        Args:
            X: 2D numpy array with shape (n_samples, n_features), the dataset to train PCA on.
            output_dims: List of integers specifying the number of principal components for each PCA extractor.
            
        Returns:
            List of trained PCA models.
        """
        # List to store the PCA models
        pca_models = []
        
        # Loop through the number of components in component_list and train PCA models
        for output_dim in output_dims:
            if output_dim == full_dim:
                pca_models.append("full")
            elif output_dim == 0:
                pca_models.append(None)
            else:
                pca = PCA(n_components=output_dim)
                pca.fit(X)
                pca_models.append(pca)          
        return pca_models


    def get_encoder_extractors(self, mod_hiddens, separate=True, config=None):
        """
        Construct encoder-style feature extractors for each modality.
        """
        if separate:
            self.extractors = [
                self._separate_encoders(dim, hiddens, mod_out)
                for dim, hiddens, mod_out in zip(self.dims, mod_hiddens, self.mod_outs)
            ]
        else:
            self._get_training_args(config)
            self.encoders = [
                self._single_encoder(dim, hiddens, mod_out)
                for dim, hiddens, mod_out in zip(self.dims, mod_hiddens, self.mod_outs)
            ]

            self.train_encoders()
            _ = self.load_checkpoints()

            self.extractors = [
                self._slice_encoders(encoder, hiddens, mod_out, dim)
                for encoder, hiddens, mod_out, dim in zip(self.encoders, mod_hiddens, self.mod_outs, self.dims)
            ]
        return self.extractors
    

    def _slice_encoders(self, encoder, hidden_dims, mod_outs, full_dim):
        """
        Slice the large encoders into sub models whose outputs are different latent representations
        from the parent encoder
        """
        extractors = []
        latent_dims = mod_outs.copy()

        # Assume the output_dims is sorted in decreasing order
        if mod_outs[0] == full_dim:
            latent_dims.pop(0)
        if mod_outs[-1] == 0:
            latent_dims.pop(-1)

        if mod_outs[0] == full_dim:
            extractors.append("full")
        
        # Remove the output layer
        encoder_layers = list(encoder.children())[0][:-1]
        base_num = len(hidden_dims)    # number of shared layers
        for i, latent_dim in enumerate(latent_dims):
            sub_layers = encoder_layers[: 2 * (base_num + i + 1)]
            extractors.append(nn.Sequential(*copy.deepcopy(sub_layers)))

        if mod_outs[-1] == 0:
            extractors.append(None)

        return extractors


    def train_encoders(self):
        # Initialize optimizers 
        # Note: This implementation uses the same hyperparameters to initialize optimizers and schedulers 
        # for all models, but one can adjust the code to allow for heteregeneous sets of hyperparameters
        self.optimizers = [
            optim.Adam(encoder.parameters(), lr=self.lr[i], weight_decay=self.weight_decay[i])
            for i, encoder in enumerate(self.encoders)
        ]
        self.best_val_losses = [float('inf')] * len(self.encoders)

        # Training loop for meta-learner
        for epoch in range(self.epochs):
            
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {}".format(
                    epoch+1, self.epochs, self.lr,)
                )
            
            train_losses = self.train_encoders_one_epoch(epoch)
            val_losses = self.validate_encoders()

            for i, encoder in enumerate(self.encoders):
                is_best = val_losses[i].avg < self.best_val_losses[i]
                msg = f"mod{i+1}_encoder: train loss: {train_losses[i].avg:.3f} - val loss: {val_losses[i].avg:.3f}"
                if is_best:
                    msg += " [*] Best so far"
                if self.verbose:
                    print(msg)

                # If this is the best model so far, update best loss and save checkpoint
                if is_best:
                    self.best_val_losses[i] = val_losses[i].avg
                    self.save_checkpoint(i, {
                        "epoch": epoch + 1,
                        "model_state": encoder.state_dict(),
                        "optim_state": self.optimizers[i].state_dict(),
                        "best_val_loss": self.best_val_losses[i],
                    })


    def validate_encoders(self):
        """
        Evaluate the encoders on the validation set.
        """
        losses = [AverageMeter() for _ in self.encoders]
        for encoder in self.encoders:
            encoder.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                mods, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    mods = [mod.cuda() for mod in mods]
                    target = target.cuda()

                for i, encoder in enumerate(self.encoders):
                    prediction = encoder(mods[i])
                    loss = self.loss_task(prediction, target)
                    losses[i].update(loss.item(), target.size(0))
        
        return losses


    def train_encoders_one_epoch(self, epoch):
        """
        Train the encoders for 1 epoch of the training set.
        This is used by train_encoders() and should not be called manually.
        """

        losses = [AverageMeter() for _ in self.encoders]
        for encoder in self.encoders:
            encoder.train()

        with tqdm(total=len(self.train_loader.dataset)) as pbar:
            for batch in self.train_loader:
                
                mods, target = batch[:-1], batch[-1]

                if self.use_gpu:
                    mods = [mod.cuda() for mod in mods]
                    target = target.cuda()

                for i, encoder in enumerate(self.encoders):
                    prediction = encoder(mods[i])
                    loss = self.loss_task(prediction, target)

                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                    losses[i].update(loss.item(), target.size(0))
                pbar.update(target.size(0))

        return losses

    
    def _get_training_args(self, config):
        self.config = config

        # data parameters
        self.num_train = len(self.train_loader.dataset)
        self.num_valid = len(self.val_loader.dataset)
        self.task_type = config["task_type"]
        if self.task_type == "classification":
            self.num_classes = config["output_dim"]
        
        # basic training parameters
        self.use_gpu = config["use_gpu"]
        self.epochs = config["epochs"]
        self.lr = config["init_lr"]
        self.weight_decay = config["weight_decay"]
        self.gamma = config["gamma"]

        # logging parameters
        self.progress = config["progress"]
        self.random_state = config["random_state"]
        self.verbose = config["verbose"]
        self.ckpt_dir = config["ckpt_dir"]

        # Initialize the task loss
        if self.task_type == "classification":
            self.loss_task = nn.CrossEntropyLoss()
        elif self.task_type == "regression":
            self.loss_task = nn.MSELoss()
        else:
            print(
                "Unkown task type {}. Task type should be either classification or regression.".format(self.task_type)
                )
  

    def _separate_encoders(self, full_dim, hidden_dims, output_dims):
        nn_models = [[]]*len(output_dims)

        for i, output_dim in enumerate(output_dims):
            if output_dim == full_dim:
                nn_models[i] = "full"
            elif output_dim == 0:
                nn_models[i] = None
            else:
                extractor = MLP_Net(full_dim, hidden_dims, output_dim)
                nn_models[i] = extractor
        return nn_models


    def _single_encoder(self, full_dim, hidden_dims, output_dims):
        # Assume the output_dims is sorted in decreasing order
        latent_dims = output_dims.copy()
        if output_dims[0] == full_dim:
            latent_dims.pop(0)
        if output_dims[-1] == 0:
            latent_dims.pop(-1)

        # define a single encoder
        if self.task_type == "classification":
            encoder = MLP_Net(full_dim, hidden_dims+latent_dims, self.num_classes)
        else:
            encoder = MLP_Net(full_dim, hidden_dims+latent_dims, 1)
            
        return encoder


    #-----------------------#
    #----- S&L Helpers -----#
    #-----------------------#
    def save_checkpoint(self, i, state):
        """
        Save a copy of the model so that it can be loaded at a future date. 
        """
    
        filename = "encoder" + str(i+1) + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        
        # Get the directory path (parent directories)
        directory = os.path.dirname(ckpt_path)
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Save the checkpoint
        torch.save(state, ckpt_path)


    def load_checkpoint(self, i):
        """
        Load a saved model from a checkpoint.
        """
        filename = "encoder" + str(i+1) + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        
        checkpoint = torch.load(ckpt_path)
        self.encoders[i].load_state_dict(checkpoint['model_state'])
        self.optimizers[i].load_state_dict(checkpoint['optim_state'])
        #self.schedulers[i].load_state_dict(checkpoint['scheduler_state'])
        
        return checkpoint['best_val_loss']


    def load_checkpoints(self):
        best_val_losses = []

        for i in range(len(self.encoders)):
            best_val_loss = self.load_checkpoint(i)
            best_val_losses.append(best_val_loss)
        
        return best_val_losses



class Trainer():
    """
    All hyperparameters are contained in the config file. Modify the config file to change
    the default parameters as needed.
    """
    def __init__(self, config, models, data_loaders):
        self.config = config

        # data parameters
        self.train_loader = data_loaders[0]
        self.val_loader = data_loaders[1]
        self.num_train = len(self.train_loader.dataset)
        self.num_valid = len(self.val_loader.dataset)
        self.task_type = config["task_type"]
        if self.task_type == "classification":
            self.num_classes = config["output_dim"]

        # basic training parameters
        self.use_gpu = config["use_gpu"]
        self.rho_list = config["rho_list"]        
        self.epochs = config["epochs"]
        self.lr = config["init_lr"]
        self.weight_decay = config["weight_decay"]
        self.gamma = config["gamma"]

        # cohort training parameters
        self.divergence_weight_type = config["divergence_weight_type"]
        self.burn_in_epochs = config["burn_in_epochs"]
        self.optimal_k = config["optimal_k"]

        # ensemble parameters
        self.ensemble_methods = config["ensemble_methods"]
        self.epochs_meta_learner = config["epochs_meta_learner"]
        
        # logging parameters
        self.progress = config["progress"]
        self.random_state = config["random_state"]
        self.verbose = config["verbose"]
        self.ckpt_dir = config["ckpt_dir"]
            
        # Initialize the divergence loss
        if self.task_type == "classification":
            self.loss_kl = nn.KLDivLoss(reduction='batchmean')
            self.loss_task = nn.CrossEntropyLoss()
        elif self.task_type == "regression":
            self.loss_mse = nn.MSELoss()
            self.loss_task = nn.MSELoss()
        else:
            print(
                "Unkown task type {}. Task type should be either classification or regression.".format(self.task_type)
                )


        # Initialize optimizers and schedulers
        # Note: This implementation uses the same hyperparameters to initialize optimizers and schedulers 
        # for all models, but one can adjust the code to allow for heteregeneous sets of hyperparameters
        self.model_num = len(models)
        assert self.model_num >= 2, "Expecting at least two student models!"

        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.models = models

        self.optimizers = []
        # self.schedulers = []
        for i in range(self.model_num):
            # initialize optimizer and scheduler
            self.models[i].to(self.device)
            optimizer = optim.Adam(self.models[i].parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
            self.optimizers.append(optimizer)
            
            # # set learning rate decay
            # scheduler = optim.lr_scheduler.StepLR(self.optimizers[i], step_size=60, gamma=self.gamma, last_epoch=-1)
            # self.schedulers.append(scheduler)
            
        # Save the initial states to retrain the models for each disagreement penalty
        self.initial_model_states = [copy.deepcopy(model.state_dict()) for model in self.models]
        self.initial_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in self.optimizers]
        #self.initial_scheduler_states = [copy.deepcopy(scheduler.state_dict()) for scheduler in self.schedulers]



    #-----------------------#
    #---- Main Functions ---#
    #-----------------------#
    def train(self):
        """
        Main training function that performs the following:
        1. Trains model cohort for user specified list of disagreement penalties;
        2. Chooses and record the best cohort via cross-validation.
        """
        
        print("Start training student cohort...")
        self.train_all_rhos()
        print("Finished training student cohort!")

        print("Selecting the optimal disgreement penalty via cross-validation...")
        self.choose_best_rho()
        print("Done!")

        if "meta_learner" in self.ensemble_methods:
            print("Training meta learner on the best cohort...")
            self.train_meta_learner(self.best_rho)
            print("Done!")
        
        if "greedy_ensemble" in self.ensemble_methods:
            print("Selecting greedy ensemble on the best cohort...")
            es = EnsembleSelection(self.loss_task, self.val_loader, task_type = self.task_type, 
                                   verbose=self.verbose, random_state=self.random_state)
            # Make sure the best model is loaded
            _ = self.load_checkpoints(self.best_rho)
            self.ens_idxs = es.build_ensemble_greedy(self.models, weighted=True)
            print("Done!")


    def test(self, test_loader):
        """
        Main testing function that performs the following:
        1. Load the best cohort chosen by cross-validation;
        2. Evaluate the method performance (including various user-specified ensembles) on the test data.
        """

        # Make sure the best model is loaded
        best_val_task_losses = self.load_checkpoints(self.best_rho)

        if "meta_learner" in self.ensemble_methods:
            self.load_meta_learner(self.best_rho)
        
        if self.task_type == "classification":
            results = self.test_classification(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        else:
            results = self.test_regression(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        
        return results



    #-----------------------#
    #--- Benchmark Funcs ---#
    #-----------------------#
    def train_ablation(self):
        """
        Trains the meta learner on cohort with no disagreement penalty
        """
        if "meta_learner" in self.ensemble_methods and self.best_rho != 0:
            print("Training meta learner with no disagreement penalty...")
            self.train_meta_learner(0)
            print("Done!")



    def test_ablation(self, test_loader):
        """
        Ablation study that tests the contribution of ensemble learning and mutual learning by the
        following benchmarks:
        1. ensembles of independent models trained with rho=0
        2. best single model trained with rho=0
        """

        # Make sure the models with rho=0 exist
        rho0_val_losses = self.load_checkpoints(0)

        if "meta_learner" in self.ensemble_methods:
            self.load_meta_learner(0)
        
        if self.task_type == "classification":
            results = self.test_classification(self.ensemble_methods+["best_single"], test_loader, rho0_val_losses)
        else:
            results = self.test_regression(self.ensemble_methods+["best_single"], test_loader, rho0_val_losses)

        # results = {f"indep_{key}": value for key, value in results.items()}
        return results



    #-----------------------#
    #----- Test Helpers ----#
    #-----------------------#
    def test_regression(self, ensemble_methods, test_loader, best_val_task_losses,
                        **kwargs):
        """
        Evaluate the performance of the ensemble on the validation set for regression tasks.
        """
        regression_ensembles = ["weighted_average", "simple_average", "meta_learner", 
                                "bagging_ensemble", "greedy_ensemble"]

        # Dictionary to store the test MSE for cohorts and ensemble methods 
        full_mse = {}

        method_mse = []
        cohort_mse = []
        for ensemble_method in ensemble_methods:
            method_mse.append(AverageMeter())

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode
            cohort_mse.append(AverageMeter())

        if "meta_learner" in ensemble_methods:
            self.meta_learner.eval()

        with torch.no_grad():
            for batch in test_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(modalities)
                    outputs.append(output)
                    cohort_mse[i].update(self.loss_mse(output, target), target.size()[0])

                outputs_stack = torch.stack(outputs).to("cpu")  # Shape: (self.model_num, batch_size, regression_outputs)

                for i, ensemble_method in enumerate(ensemble_methods):
                    # Apply ensemble method
                    if ensemble_method == "simple_average":
                        final_output = torch.mean(outputs_stack, dim=0)

                    elif ensemble_method == "weighted_average":
                        weights = get_weights_by_task_loss(best_val_task_losses)
                        weights = weights.unsqueeze(1).unsqueeze(2)
                        final_output = torch.sum(weights * outputs_stack, dim=0)

                    elif ensemble_method == "meta_learner":
                        outputs_concat = torch.cat(outputs, dim=1)
                        final_output = self.meta_learner(outputs_concat)
                    
                    elif ensemble_method == "best_single":
                        best_model = best_val_task_losses.index(min(best_val_task_losses))
                        final_output = outputs_stack[best_model]
                    
                    elif ensemble_method == "greedy_ensemble":
                        weights = get_weights_by_task_loss(best_val_task_losses)[self.ens_idxs]
                        weights /= torch.sum(weights)
                        weights = weights.unsqueeze(1).unsqueeze(2)
                        final_output = torch.sum(weights * outputs_stack[self.ens_idxs], dim=0)

                    elif ensemble_method == "bagging_ensemble":
                        bagging_num = kwargs.get('bagging_num', None)
                        # Determine the number of models to bag
                        if bagging_num is None:
                            bagging_num = int(0.6 * self.model_num)  # Default to 60% of models

                        # Perform bagging on the models
                        selected_models = [random.choice(range(self.model_num)) for _ in range(bagging_num)]
                        bagged_outputs = outputs_stack[selected_models]
                        final_output = torch.mean(bagged_outputs, dim=0)

                    else:
                        raise ValueError(f"Unknown ensemble method '{ensemble_method}', \
                                        valid options are {regression_ensembles}.")

                    if self.use_gpu:
                        final_output = final_output.cuda()
                    method_mse[i].update(self.loss_mse(final_output, target), target.size()[0])
            
            for i, ensemble_method in enumerate(ensemble_methods):
                # Convert to float if the value is a tensor
                mse_value = method_mse[i].avg
                if hasattr(mse_value, 'item'):  # Check if it's a tensor with the 'item' method
                    full_mse[ensemble_method] = mse_value.item()
                else:
                    full_mse[ensemble_method] = float(mse_value)

            # Convert to float for the 'cohort' list of MSEs
            full_mse["cohort"] = [
                cohort_mse[i].avg.item() if hasattr(cohort_mse[i].avg, 'item') else float(cohort_mse[i].avg)
                for i in range(self.model_num)
            ]

            if self.verbose:
                for key, value in full_mse.items():
                    print(f'Method: ({key}), Test_MSE: {value}')

            return full_mse



    def test_classification(self, ensemble_methods, test_loader, best_val_task_losses):
        """
        Validate the performance of the ensemble on the validation set for classification tasks.
        """

        classification_ensembles = ["majority_voting", "weighted_voting", 
                                    "weighted_average", "simple_average", 
                                    "meta_learner", "greedy_ensemble"]

        # Dictionary to store the test MSE for cohorts and ensemble methods 
        full_accuracy = {}

        method_accuracy = []
        cohort_accuracy = []
        for ensemble_method in ensemble_methods:
            method_accuracy.append(AverageMeter())

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode
            cohort_accuracy.append(AverageMeter())

        if "meta_learner" in ensemble_methods:
            self.meta_learner.eval()

        with torch.no_grad():
            for batch in test_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(modalities)
                    outputs.append(output)
                    cohort_accuracy[i].update(calculate_accuracy(output, target), target.size()[0])
                outputs_stack = torch.stack(outputs)  # Shape: (self.model_num, batch_size, num_classes)

                #pdb.set_trace()

                for i, ensemble_method in enumerate(ensemble_methods):
                    # Apply ensemble method
                    if ensemble_method == "simple_average":
                        final_output = torch.mean(outputs_stack, dim=0)
                        method_accuracy[i].update(calculate_accuracy(final_output, target), target.size()[0])

                    elif ensemble_method == "weighted_average":
                        weights = get_weights_by_task_loss(best_val_task_losses)                        
                        final_output = torch.sum(weights.unsqueeze(1).unsqueeze(2) * outputs_stack, dim=0)
                        method_accuracy[i].update(calculate_accuracy(final_output, target), target.size()[0])

                    elif ensemble_method == "majority_voting":
                        final_pred = torch.mode(outputs_stack.argmax(dim=2), dim=0).values  # Majority voting
                        acc = final_pred.eq(target.view_as(final_pred)).sum().item() / target.size()[0]
                        method_accuracy[i].update(acc, target.size()[0])

                    elif ensemble_method == "weighted_voting":
                        weights = get_weights_by_task_loss(best_val_task_losses)
                        top_preds = torch.argmax(outputs_stack, dim=2)
                        num_classes = outputs_stack.shape[2]
                        batch_size = outputs_stack.shape[1]
                        weighted_votes = torch.zeros((batch_size, num_classes), device=outputs_stack.device)
                        for k, model_preds in enumerate(top_preds):
                            weighted_votes.scatter_add_(1, model_preds.unsqueeze(1), 
                                                        torch.full((batch_size, 1), weights[k], device=outputs_stack.device))
                        final_pred = torch.argmax(weighted_votes, dim=1)
                        acc = final_pred.eq(target.view_as(final_pred)).sum().item() / target.size()[0]
                        method_accuracy[i].update(acc, target.size()[0])

                    elif ensemble_method == "meta_learner":
                        outputs_concat = torch.cat(outputs, dim=1)
                        final_output = self.meta_learner(outputs_concat)
                        method_accuracy[i].update(calculate_accuracy(final_output, target), target.size()[0])
                    
                    elif ensemble_method == "best_single":
                        best_model = best_val_task_losses.index(min(best_val_task_losses))
                        final_output = outputs_stack[best_model]
                        method_accuracy[i].update(calculate_accuracy(final_output, target), target.size()[0])
                    
                    elif ensemble_method == "greedy_ensemble":
                        weights = get_weights_by_task_loss(best_val_task_losses)[self.ens_idxs]
                        weights /= torch.sum(weights)
                        weights = weights.unsqueeze(1).unsqueeze(2)
                        final_output = torch.sum(weights * outputs_stack[self.ens_idxs], dim=0)
                        method_accuracy[i].update(calculate_accuracy(final_output, target), target.size()[0])
                        
                    else:
                        raise ValueError(f"Unknown ensemble method '{ensemble_method}', \
                                        valid options are {classification_ensembles}.")

                for i, ensemble_method in enumerate(ensemble_methods):
                    full_accuracy[ensemble_method] = method_accuracy[i].avg
                full_accuracy["cohort"] = [cohort_accuracy[i].avg for i in range(self.model_num)]
                
            if self.verbose:
                for key, value in full_accuracy.items():
                    print(f'Method: ({key}), Test_Accuracy: {value}')

            return full_accuracy



    #-----------------------#
    #---- Train Helpers ----#
    #-----------------------#
    def train_meta_learner(self, rho):
        """
        Train the meta learner ensemble on the best cohort, meta learner takes the concatenated outputs of the cohort
        and produce a final prediction. 
        """
        
        # Load the cohort with the specific disagreement penalty
        _ = self.load_checkpoints(rho)
        
        for i in range(self.model_num):
            self.models[i].eval()

        # Initialize meta-learner model
        self.meta_learner = self.initialize_meta_learner()
        self.meta_learner = self.meta_learner.to(self.device)

        # Define optimizer and scheduler for meta-learner
        self.meta_learner_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.1, weight_decay=0)
        #self.meta_learner_scheduler = optim.lr_scheduler.StepLR(self.optimizers[i], step_size=10, gamma=0.1, last_epoch=-1)

        best_val_loss = float('inf')

        # Training loop for meta-learner
        for epoch in range(self.epochs_meta_learner):
            
            # Train for one epoch
            train_loss = self.train_meta_learner_one_epoch(epoch)

            # Validate on the validation set
            val_loss = self.validate_meta_learner(epoch)

            is_best = val_loss.avg < best_val_loss

            # Print train and validation losses
            msg1 = "meta_learner: train task loss: {:.3f} "
            msg2 = "- val task loss: {:.3f}"
            if is_best:
                msg2 += " [*] Best so far"
            msg = msg1 + msg2

            if self.verbose:
                    print(
                        msg.format(train_loss.avg, val_loss.avg)
                    )

            # Save the best model based on validation loss
            if is_best:
                best_val_loss = val_loss.avg
                self.save_meta_learner(rho, {
                    'epoch': epoch + 1,
                    "disagreement_penalty": rho,
                    'model_state': self.meta_learner.state_dict(),
                    'optimizer_state': self.meta_learner_optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                })



    def train_meta_learner_one_epoch(self, epoch):
        """
        Train the meta-learner for 1 epoch of the training set.
        This is used by train_meta_learner() and should not be called manually.
        """
        # Set the meta-learner to training mode
        self.meta_learner.train()

        # Initialize meters to track losses
        loss_meter = AverageMeter()

        with tqdm(total=len(self.train_loader)) as pbar:
            for batch in self.train_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                outputs = [model(modalities) for model in self.models]
                outputs_concat = torch.cat(outputs, dim=1)

                predictions = self.meta_learner(outputs_concat)
                loss = self.loss_task(predictions, target)

                self.meta_learner_optimizer.zero_grad() 
                loss.backward()
                self.meta_learner_optimizer.step()

                loss_meter.update(loss.item(), target.size(0))
                pbar.update(target.size(0))

        return loss_meter


    def validate_meta_learner(self, epoch):
        """
        Evaluate the meta-learner on the validation set.
        
        Args:
            meta_learner (torch.nn.Module): The meta-learner model.
            val_loader (DataLoader): Validation data loader for meta-learner.
        
        Returns:
            float: Validation loss.
        """
        self.meta_learner.eval()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for batch in self.val_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                outputs = [model(modalities) for model in self.models]
                outputs_concat = torch.cat(outputs, dim=1)

                predictions = self.meta_learner(outputs_concat)
                loss = self.loss_task(predictions, target)

                loss_meter.update(loss.item(), target.size(0))

        return loss_meter



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

        return meta_learner




    def choose_best_rho(self):
        """
        Select the best rho that has smallest average task loss on the validation set.
        """
        
        best_rho = None
        best_avg_loss = float('inf')

        # Loop over each rho value and validate
        for rho in self.rho_list:
            avg_loss = 0
            
            # Load the saved models for this rho
            val_losses = self.load_checkpoints(rho)
            if hasattr(self, "cluster_idxs"):
                avg_loss = np.mean(np.array(val_losses)[self.cluster_idxs])
            else:
                avg_loss = np.mean(val_losses)
        
            # Check for best performance (minimize loss)
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_rho = rho
            
        # Print the best rho and performance for the current method
        print(f"Best rho: {best_rho} with average task loss: {best_avg_loss:.4f}")

        self.best_rho = best_rho



    def train_all_rhos(self):
        for rho in self.rho_list:
            print(f"Training with disagreement penalty = {rho}")
            
            # Reset models, optimizers, and schedulers to their initial states
            for i in range(self.model_num):
                self.models[i].load_state_dict(self.initial_model_states[i])
                self.optimizers[i].load_state_dict(self.initial_optimizer_states[i])
                #self.schedulers[i].load_state_dict(self.initial_scheduler_states[i])

            # Call the train function to train with the current rho
            self.train_fixed_rho(rho)

        

    def train_fixed_rho(self, rho):
        """
        Train the models for a fixed level of disagreement penalty. 
        """
        
        # Variable to keep track of the best validation loss
        self.best_val_task_losses = [float('inf')] * self.model_num

        # determine the divergence weights, for method fixed and clustering, this is an one-time effort
        if self.divergence_weight_type == "fixed":
            if rho > 0 and not hasattr(self, 'fixed_weights'):  
                # use the validation performance of the models under rho = 0
                try:
                    rho0_val_losses = self.load_checkpoints(0)

                    if self.verbose:
                        print("Computing divergence weights by scaling method...")

                    self.fixed_weights = get_weights_by_task_loss(rho0_val_losses, scale=self.divergence_weight_scale)
                    weights=self.fixed_weights

                    if self.verbose:
                        print(f"Computed divergence weights by scaling method, weights are {weights}")

                except Exception as e:
                    # Handle any other exceptions that may occur
                    print(f"An error occurred while loading the checkpoint with rho = 0: {e}, \
                            using unweighted disagreement loss instead.")
                    weights = torch.ones(self.model_num)

            elif rho > 0 and hasattr(self, 'fixed_weights'): 
                weights=self.fixed_weights
            else:
                weights = torch.ones(self.model_num)

        elif self.divergence_weight_type == "clustering":
            if rho > 0 and not hasattr(self, 'clustering_weights'):   
                # use the validation performance of the models under rho = 0
                try:
                    rho0_val_losses = self.load_checkpoints(0)

                    if self.verbose:
                        print("Computing divergence weights by clustering method...")

                    self.clustering_weights, optim_k, self.cluster_idxs = get_weights_by_clustering(rho0_val_losses, optimal_k=self.optimal_k, 
                                                                                 verbose=self.verbose, random_state=self.random_state)
                    self.optimal_k = optim_k   # update the optimal cluster number
                    weights=self.clustering_weights

                    if self.verbose:
                        print(f"Computed divergence weights by clustering method, weights are {weights}")

                except Exception as e:
                    # Handle any other exceptions that may occur
                    print(f"An error occurred while loading the checkpoint with rho = 0: {e}, \
                            using unweighted disagreement loss instead.")
                    weights = torch.ones(self.model_num)

            elif rho > 0 and hasattr(self, 'clustering_weights'):   
                weights = self.clustering_weights
            else:
                weights = torch.ones(self.model_num)

        else:
            weights = torch.ones(self.model_num)

        for epoch in range(self.epochs):

            # for scheduler in self.schedulers:
            #     scheduler.step(epoch)
    
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )
            
            
            if epoch < self.burn_in_epochs:
                # during burn in epochs, train models independently to avoid negative mutual learning
                train_losses, train_task_losses = self.train_one_epoch(epoch, 0, weights)
                val_losses, val_task_losses = self.validate(epoch, 0)
            else: 
                train_losses, train_task_losses = self.train_one_epoch(epoch, rho, weights)
                val_losses, val_task_losses = self.validate(epoch, rho, weights)
                
                # adaptively update the divergence weights
                if self.divergence_weight_type == "adaptive":
                    weights = get_weights_by_task_loss([val_task_losses[i].avg for i in range(self.model_num)], 
                                                        scale=self.divergence_weight_scale)
                
            for i in range(self.model_num):
                # Check if the current model has the best validation task loss
                is_best = val_task_losses[i].avg < self.best_val_task_losses[i]
                
                # Print train and validation losses
                msg1 = "model_{:d}: train loss: {:.3f}, train task loss: {:.3f} "
                msg2 = "- val loss: {:.3f}, val task loss: {:.3f}"
                if is_best:
                    msg2 += " [*] Best so far"
                msg = msg1 + msg2
                
                if self.verbose:
                    print(
                        msg.format(i+1, train_losses[i].avg, train_task_losses[i].avg, 
                                        val_losses[i].avg, val_task_losses[i].avg)
                    )

                # If this is the best model so far, update best loss and save checkpoint
                if is_best:
                    self.best_val_task_losses[i] = val_task_losses[i].avg
                    self.save_checkpoint(rho, i,
                        {
                            "epoch": epoch + 1,
                            "disagreement_penalty": rho,
                            "model_state": self.models[i].state_dict(),
                            "optim_state": self.optimizers[i].state_dict(),
                            "best_val_task_loss": self.best_val_task_losses[i],
                        }
                    )

    
    def train_one_epoch(self, epoch, rho, weights):
        """
        Train the model for 1 epoch of the training set.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        task_losses = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            task_losses.append(AverageMeter())


        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch in self.train_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()
                
                outputs = [model(modalities) for model in self.models]
                for i in range(self.model_num):
                    task_loss = self.loss_task(outputs[i], target)
                    divergence_loss = 0
                    if rho != 0:
                        for j in range(self.model_num):
                            if i != j:
                                weight = weights[j]
                                if self.task_type == "regression":
                                    divergence_loss = divergence_loss + weight * self.loss_mse(outputs[i], outputs[j].detach())
                                else:
                                    divergence_loss = divergence_loss + weight * self.loss_kl(
                                                                        F.log_softmax(outputs[i], dim=1), 
                                                                        F.softmax(outputs[j].detach(), dim=1)
                        )
                    #loss = task_loss + rho * divergence_loss / (sum(weights) - weights[i])
                    loss = task_loss + rho * divergence_loss / sum(weights)
                    
                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

                    # compute gradients and update optimizer
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    #self.schedulers[i].step()
                
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                # pbar.set_description(
                #     (
                #         "{:.1f}s - average loss: {:.3f} - average task loss: {:.3f}".format(
                #             (toc-tic), np.mean([losses[i].avg for i in range(self.model_num)]), 
                #                        np.mean([task_losses[i].avg for i in range(self.model_num)])
                #         )
                #     )
                # )
                pbar.update(target.shape[0])

            return losses, task_losses


    def validate(self, epoch, rho, weights):
        """
        Evaluate the model on the validation set on the current model.
        """
        losses = []
        task_losses = []
        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            task_losses.append(AverageMeter())
        
        with torch.no_grad():
            for batch in self.val_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()

                outputs = [model(modalities) for model in self.models]
                for i in range(self.model_num):
                    task_loss = self.loss_task(outputs[i], target)
                    divergence_loss = 0
                    if rho != 0:
                        for j in range(self.model_num):
                            if i != j:
                                weight = weights[j]
                                if self.task_type == "regression":
                                    divergence_loss = divergence_loss + weight * self.loss_mse(outputs[i], outputs[j])
                                else:
                                    divergence_loss = divergence_loss + weight * self.loss_kl(
                                                                        F.log_softmax(outputs[i], dim=1), 
                                                                        F.softmax(outputs[j], dim=1)
                        )
                    #loss = task_loss + rho * divergence_loss / (sum(weights) - weights[i])
                    loss = task_loss + rho * divergence_loss / sum(weights)

                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

        return losses, task_losses



    #-----------------------#
    #----- S&L Helpers -----#
    #-----------------------#
    def save_checkpoint(self, rho, i, state):
        """
        Save a copy of the model so that it can be loaded at a future date. 
        """
    
        filename = "model" + str(i+1) + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, str(rho), filename)
        
        # Get the directory path (parent directories)
        directory = os.path.dirname(ckpt_path)
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Save the checkpoint
        torch.save(state, ckpt_path)



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

        return checkpoint['best_val_loss']



    def load_checkpoint(self, rho, i):
        """
        Load a saved model from a checkpoint.
        """
        filename = "model" + str(i+1) + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, str(rho), filename)
        
        checkpoint = torch.load(ckpt_path)
        self.models[i].load_state_dict(checkpoint['model_state'])
        self.optimizers[i].load_state_dict(checkpoint['optim_state'])
        #self.schedulers[i].load_state_dict(checkpoint['scheduler_state'])
        
        return checkpoint['best_val_task_loss']


    def load_checkpoints(self, rho):
        """
        Load all checkpoints for the specific rho
        """
        best_val_task_losses = []

        for i in range(self.model_num):
            best_val_task_loss = self.load_checkpoint(rho, i)
            best_val_task_losses.append(best_val_task_loss)
        
        return best_val_task_losses



# The following class is adapted from the ensemble selection method from the following papers:
# ------------------------------------------------------------------------------------------
# [1] Caruana, et al, "Ensemble Selection from Libraries of Rich Models",
#        Proceedings of the 21st International Conference on Machine Learning
#        (ICML `04).
# [2] Caruana, et al, "Getting the Most Out of Ensemble Selection",
#        Proceedings of the 6th International Conference on Data Mining
#        (ICDM `06).
#-------------------------------------------------------------------------------------------
class EnsembleSelection():
    def __init__(self, loss_task, val_loader, task_type='regression',
                       n_best=3, prune_fraction=0.2, max_models=30,
                       verbose=False, random_state=0):            
        ''' 
        Args:
        loss_task: loss function for the task
        val_loader: dataloader for the validation(hillclimbing) dataset
        n_best: number of top performing models to initialize the ensemble
        prune_fraction: fraction of worst performing models to be pruned
        max_models: maximum number of models in the final ensemble
        '''
        self.task_type = task_type
        self.loss_task = loss_task
        self.val_loader = val_loader
        self.n_best = n_best
        self.prune_fraction = prune_fraction
        self.max_models = max_models
        self.verbose = verbose
        self.random_state = random_state


    def build_ensemble_greedy(self, models, weighted=True):
        """
        Build the ensemble by selecting without replacement the model that minimizes the task loss. 

        Args:
        outputs_stack: stacked output of the validation (hillclimbing) set
        target: labels of the validation set
        weighted: If True, the final prediction is a weighted averaging of the base predictions, 
                  otherwise, use simple averaging.
        """
        num_models = len(models)
        ensemble = Counter()

        outputs = []
       # Load the hillclimb set, assume the val_loader has only one full batch
        full_batch = next(iter(self.val_loader))
        modalities, target = full_batch[:-1], full_batch[-1]

        for model in models:
            model.eval()
            # Forward pass for each model using the list of modalities
            outputs.append(model(modalities))

        outputs_stack = torch.stack(outputs)
        del outputs

        # Calculate losses for all models
        if self.task_type == "regression":
            losses = [self.loss_task(outputs_stack[i], target) for i in range(num_models)]
        elif self.task_type == "classification":
            losses = [self.loss_task(outputs_stack[i], target.squeeze().long()) for i in range(num_models)]

        # Prune the worst performing models
        num_to_keep = int(num_models * (1.0 - self.prune_fraction))
        ranked_models = sorted(range(num_models), key=lambda i: losses[i])
        pruned_models = ranked_models[:num_to_keep]

        # Update the max model num so that it does not exceed the number of pruned models
        self.max_models = min([num_to_keep, self.max_models])

        if self.verbose:
            print(f"Pruned {num_models - num_to_keep} worst models, keeping {num_to_keep} models")

        # Initialize ensemble with the n_best models from the pruned set
        ensemble.update(pruned_models[:self.n_best])  # Start with n_best models

        if self.verbose:
            print(f"Initial best models: {pruned_models[:self.n_best]} with losses: {[losses[i] for i in pruned_models[:self.n_best]]}")

        if weighted:
            weights = get_weights_by_task_loss(losses)
        else:
            weights = tensor.ones(num_models)
        
        # Compute the initial ensemble loss
        ens_loss = self._get_weighted_ens_loss(list(ensemble), outputs_stack, target, weights)

        while len(ensemble) < self.max_models:
            best_candidate_loss = float('inf')
            best_candidate_index = None

            for i in pruned_models:
                if i in ensemble:
                    continue

                # Calculate loss with this candidate model added
                candidate_ensemble = list(ensemble) + [i]
                candidate_loss = self._get_weighted_ens_loss(candidate_ensemble, outputs_stack, target, weights)

                if candidate_loss < best_candidate_loss:
                    best_candidate_loss = candidate_loss
                    best_candidate_index = i

            if best_candidate_loss < ens_loss:
                ensemble.update([best_candidate_index])
                ens_loss = best_candidate_loss

                if self.verbose:
                    print(f"Added model {best_candidate_index} with loss improvement to {ens_loss}")

            else:
                break

        return list(ensemble)


    def _get_weighted_ens_loss(self, ens_idxs, outputs_stack, target, weights):
        ens_weights = weights[ens_idxs]
        ens_weights /= torch.sum(ens_weights)
        ens_weights = ens_weights.unsqueeze(1).unsqueeze(2)

        final_output = torch.sum(ens_weights * outputs_stack[ens_idxs], dim=0)

        if self.task_type == "classification":
            target = target.squeeze().long()
        return self.loss_task(final_output, target)


