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
from meta_fusion.archive_utils import *
from meta_fusion.archive_models import MLP_Net
torch.autograd.set_detect_anomaly(True)


class Extractors:
    """
    This class defines the feature extractors.
    """
    # [Note] This class now only implements tubular information, i.e. d1, d2 are dimensions
    # of vector-like modalities. Generalize the class in the future to accomondate images or
    # modalities of other types! 
    def __init__(self, mod1_outs, mod2_outs, d1, d2,
                       train_loader, val_loader):
        self.mod1_outs = sorted(mod1_outs, reverse=True)  # output dimensions of the feature extractors 
        self.mod2_outs = sorted(mod2_outs, reverse=True)
        self.d1 = d1
        self.d2 = d2
        self.train_loader = train_loader
        self.val_loader = val_loader


    
    def get_dummy_extractors(self):
        """
        Create two types of dummy extractors:
        1. When output dim is equal to the data dim, dummy extractor is 'full' which returns the raw data
        2. When output dim is 0, dummy extractor is None which ignores the modality
        This function is called when mod1_outs/mod2_outs contains only 0 or full dimension
        """
        self.mod1_extractors = []
        self.mod2_extractors = []
        
        err_msg = "Invalid output dimension, create dummy extractors only when \
                   output dimension is equal to 0 or the raw feature size."
        for dim in self.mod1_outs:
            if dim == self.d1:
                self.mod1_extractors.append('full')
            elif dim == 0:
                self.mod1_extractors.append(None)
            else:
                print(err_msg)
        
        for dim in self.mod2_outs:
            if dim == self.d2:
                self.mod2_extractors.append('full')
            elif dim == 0:
                self.mod2_extractors.append(None)
            else:
                print(err_msg)

        return self.mod1_extractors, self.mod2_extractors



    def get_PCA_extractors(self):
        """
        Dimension reduction by PCA 
        """
        X1_train, X2_train, _ = load_all_data(self.train_loader)
        X1_val, X2_val, _ = load_all_data(self.val_loader)
        X1 = np.concatenate((X1_train, X1_val), axis=0)
        X2 = np.concatenate((X2_train, X2_val), axis=0)
    
        self.mod1_extractors = self._train_pca_extractors(X1, self.mod1_outs, self.d1)
        self.mod2_extractors = self._train_pca_extractors(X2, self.mod2_outs, self.d2)
        
        return self.mod1_extractors, self.mod2_extractors


    
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


    
    def get_encoder_extractors(self, mod1_hiddens, mod2_hiddens, 
                                     separate=True, config=None):
        """
        Construct the encoder-style feature extractor
    
        Args:
        mod1_hiddens: List of integers specifying the dimensions of hidden layers of the encoder trained on modality 1, 
                      assuming all encoders share the same hidden strucutures.
        mod2_hiddens: List of integers specifying the dimensions of hidden layers of the encoder trained on modality 2, 
                      assuming all encoders share the same hidden strucutures.
        separate: If True, encoders are separate models with same hidden layers but different output dimensions.
                  If False, define a large encoder of (input, hiddens, output_dims) and pretrain the large encoder
                  in a supervised manner.
        config: Additional arguments for training the large encoder if separate=False.
        """

        if separate:
            self.mod1_extractors = self._separate_encoders(self.d1, mod1_hiddens, self.mod1_outs)
            self.mod2_extractors = self._separate_encoders(self.d2, mod2_hiddens, self.mod2_outs)
        else:
            # Assume the config specifies the training arguments
            self._get_training_args(config)
                
            # define the single encoders
            mod1_encoder = self._single_encoder(self.d1, mod1_hiddens, self.mod1_outs)
            mod2_encoder = self._single_encoder(self.d2, mod2_hiddens, self.mod2_outs)
            self.encoders = [mod1_encoder, mod2_encoder]
                           
            self.train_encoders()

            # Load the best checkpoints
            _ = self.load_checkpoints()

            # Slice the encoder to get different latent representations
            self.mod1_extractors = self._slice_encoders(self.encoders[0], mod1_hiddens, self.mod1_outs, self.d1)
            self.mod2_extractors = self._slice_encoders(self.encoders[1], mod2_hiddens, self.mod2_outs, self.d2)

        return self.mod1_extractors, self.mod2_extractors


    
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
        mod1_optimizer = optim.Adam(self.encoders[0].parameters(), lr=self.lr, weight_decay=self.weight_decay)
        mod2_optimizer = optim.Adam(self.encoders[1].parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizers = [mod1_optimizer, mod2_optimizer]
    
        self.best_val_losses = [float('inf')]*2

        # Training loop for meta-learner
        for epoch in range(self.epochs):
            
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )
            
            # Train for one epoch
            train_losses = self.train_encoders_one_epoch(epoch)

            # Validate on the validation set
            val_losses = self.validate_encoders()

            for i in range(2):
                # Check if the current model has the best validation task loss
                is_best = val_losses[i].avg < self.best_val_losses[i]
                
                # Print train and validation losses
                msg = "mod{:d}_encoder: train loss: {:.3f}- val loss: {:.3f}"
                if is_best:
                    msg += " [*] Best so far"
                
                if self.verbose:
                    print(
                        msg.format(i+1, train_losses[i].avg, val_losses[i].avg)
                    )

                # If this is the best model so far, update best loss and save checkpoint
                if is_best:
                    self.best_val_losses[i] = val_losses[i].avg
                    self.save_checkpoint(i,
                        {
                            "epoch": epoch + 1,
                            "model_state": self.encoders[i].state_dict(),
                            "optim_state": self.optimizers[i].state_dict(),
                            "best_val_loss": self.best_val_losses[i],
                        }
                    )



    def validate_encoders(self):
        """
        Evaluate the encoders on the validation set.
        """
        # Set the encoders to training mode
        losses = []
        for i in range(2):
            self.encoders[i].eval()
            losses.append(AverageMeter())
        
        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(self.val_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
            
                mods = [mod1, mod2]
                for i in range(2):          
                    # Forward pass through encoders
                    prediction = self.encoders[i](mods[i])
                    loss = self.loss_task(prediction, target)

                    # Update loss meters
                    losses[i].update(loss.item(), target.size(0))
        
        return losses



    def train_encoders_one_epoch(self, epoch):
        """
        Train the encoders for 1 epoch of the training set.
        This is used by train_encoders() and should not be called manually.
        """
        # Set the encoders to training mode
        losses = []
        for i in range(2):
            self.encoders[i].train()
            losses.append(AverageMeter())

        # Start progress bar for the training epoch
        with tqdm(total=len(self.train_loader.dataset)) as pbar:
            for batch, (mod1, mod2, target) in enumerate(self.train_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
                
                mods = [mod1, mod2]
                for i in range(2):          
                    # Forward pass through encoders
                    prediction = self.encoders[i](mods[i])
                    loss = self.loss_task(prediction, target)

                    # Backward pass and optimization step
                    self.optimizers[i].zero_grad() 
                    loss.backward()
                    self.optimizers[i].step()
    
                    # Update loss meters
                    losses[i].update(loss.item(), target.size(0))

                # Update the progress bar
                self.batch_size = target.size(0)
                pbar.update(self.batch_size)

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
        # Loop through the number of components in component_list and train PCA models
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
        """
        Load all checkpoints for the specific rho
        """

    def load_checkpoints(self, rho):
        """
        #CHANGED_BY_PARNIAN
        Load all checkpoints for the specific rho
        """
        
        mod1, mod2, target = self.val_loader
        if self.use_gpu:
            mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
        
        #forward pass
        outputs=[]
        for model in self.models:
            outputs.append(model(mod1, mod2))

        Joint_prediction = 0
        for x in range(self.num_models):
            Joint_prediction += 1/self.num_models * outputs[x]
        val_loss =  self.loss_mse(Joint_prediction, target)      
        
        return val_loss



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
        self.use_tensorboard = False
        self.ckpt_dir = config["ckpt_dir"]

        if self.use_tensorboard:
            tensorboard_dir = config["log_dir"]
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
                print(f"Created TensorBoard log directory: {tensorboard_dir}")
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

            
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
        # CHANGED_BY_PARNIAN
        Main testing function that performs the following:
        1. Load the best cohort chosen by cross-validation;
        2. Evaluate the method performance (including various user-specified ensembles) on the test data.
        """

        mod1, mod2, target = self.test_loader
        if self.use_gpu:
            mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
        
        #forward pass
        outputs=[]
        for model in self.models:
            outputs.append(model(mod1, mod2))

        Joint_prediction = 0
        for x in range(self.num_models):
            Joint_prediction += 1/self.num_models * outputs[x]
        test_loss =  self.loss_mse(Joint_prediction, target)      
        
        return test_loss



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

        # if self.task_type == "classification":
        #     single_model_result = self.test_classification(, test_loader, rho0_val_losses)
        # else:
        #     single_model_result = self.test_regression(["best_single"], test_loader, rho0_val_losses)

        # results.update(single_model_result)
        results = {f"indep_{key}": value for key, value in results.items()}
        return results



    #-----------------------#
    #----- Test Helpers ----#
    #-----------------------#
    def test_regression(self, ensemble_methods, test_loader, best_val_task_losses,
                        **kwargs):
        """
        # CHANGED_BY_PARNIAN
        Evaluate the performance of the ensemble on the validation set for regression tasks.
        """
        mod1, mod2, target = self.test_loader
        if self.use_gpu:
            mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
        
        #forward pass
        outputs=[]
        for model in self.models:
            outputs.append(model(mod1, mod2))

        Joint_prediction = 0
        for x in range(self.num_models):
            Joint_prediction += 1/self.num_models * outputs[x]
        test_loss =  self.loss_mse(Joint_prediction, target)      
        
        return test_loss
 
 
 


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
            for batch, (mod1, mod2, target) in enumerate(test_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(mod1, mod2)
                    outputs.append(output)
                    cohort_accuracy[i].update(calculate_accuracy(output, target), target.size()[0])
                outputs_stack = torch.stack(outputs)  # Shape: (self.model_num, batch_size, num_classes)

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
                        final_pred = torch.sum(weights.unsqueeze(1).unsqueeze(2) * outputs_stack.argmax(dim=2), dim=0)
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

        # Start progress bar for the training epoch
        with tqdm(total=len(self.train_loader)) as pbar:
            for batch, (mod1, mod2, target) in enumerate(self.train_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Collect model outputs from all models in self.models
                outputs = []
                for model in self.models:
                    outputs.append(model(mod1, mod2))

                outputs_concat = torch.cat(outputs, dim=1)  # Concatenate outputs for meta-learner input

                # Forward pass through meta-learner
                predictions = self.meta_learner(outputs_concat)
                loss = self.loss_task(predictions, target)

                # Backward pass and optimization step
                self.meta_learner_optimizer.zero_grad() 
                loss.backward()
                self.meta_learner_optimizer.step()

                # Update loss meters
                loss_meter.update(loss.item(), target.size(0))

                # pbar.set_description(
                #     (
                #         "task loss: {:.3f}".format(loss_meter.avg)
                #     )
                # )

                # Update the progress bar
                self.batch_size = target.size(0)
                pbar.update(self.batch_size)

                # Log to TensorBoard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + batch
                    log_value('meta_learner_train_loss', loss_meter.avg, iteration)

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
        # Initialize meters to track losses
        loss_meter = AverageMeter()

        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(self.val_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Collect model outputs from all models in self.models
                outputs = []
                for model in self.models:
                    outputs.append(model(mod1, mod2))
                outputs_concat = torch.cat(outputs, dim=1)  # Concatenate outputs for meta-learner input

                # Forward pass through meta-learner
                predictions = self.meta_learner(outputs_concat)
                loss = self.loss_task(predictions, target)

                # Update loss meters
                loss_meter.update(loss.item(), target.size(0))

            # log to tensorboard for every epoch
            if self.use_tensorboard:
                log_value('meta_learner_val_loss', loss_meter.avg, epoch+1)

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
        print(weights)
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
                val_losses, val_task_losses = self.validate(epoch, rho)
                
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
        #CHANGED_BY_PARNIAN
        Train the model for 1 epoch of the training set.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        task_losses = []

        for i in range(self.model_num):
            self.models[i].train() # Enables layers like Dropout and BatchNorm to behave as they would during training.
            losses.append(AverageMeter())
            task_losses.append(AverageMeter())


        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch, (mod1, mod2, target) in enumerate(self.train_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
                
                #forward pass
                outputs=[]
                for model in self.models:
                    outputs.append(model(mod1, mod2))

                Joint_prediction = 0
                for x in range(self.num_models):
                    Joint_prediction += 1/self.num_models * outputs[x]
                loss =  self.loss_mse(Joint_prediction, target)

                for i in range(self.model_num):
                    # record loss
                    losses[i].update(loss.item(), target.size()[0])

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
                self.batch_size = target.shape[0]
                pbar.update(self.batch_size)

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + batch
                    for i in range(self.model_num):
                        log_value('train_loss_%d' % (i+1), losses[i].avg, iteration)
                        log_value('train_task_loss_%d' % (i+1), task_losses[i].avg, iteration)
            
            return losses, task_losses



    def validate(self, epoch, rho):
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
            for batch, (mod1, mod2, target) in enumerate(self.val_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                #forward pass
                outputs=[]
                for model in self.models:
                    outputs.append(model(mod1, mod2))
                for i in range(self.model_num):
                    task_loss = self.loss_task(outputs[i], target)
                    divergence_loss = 0
                    for j in range(self.model_num):
                        if i != j:
                            if self.task_type == "regression":
                                divergence_loss = divergence_loss + self.loss_mse(outputs[i], outputs[j])
                            else:
                                divergence_loss = divergence_loss + self.loss_kl(F.log_softmax(outputs[i], dim = 1), 
                                                                                 F.softmax(outputs[j], dim=1))
                    loss = task_loss + rho * divergence_loss / (self.model_num - 1)

                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

            # log to tensorboard for every epoch
            if self.use_tensorboard:
                for i in range(self.model_num):
                    log_value('val_loss_%d' % (i+1), losses[i].avg, epoch+1)
                    log_value('val_acc_%d' % (i+1), accs[i].avg, epoch+1)

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
        #CHANGED_BY_PARNIAN
        Load all checkpoints for the specific rho
        """
        
        mod1, mod2, target = self.val_loader
        if self.use_gpu:
            mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()
        
        #forward pass
        outputs=[]
        for model in self.models:
            outputs.append(model(mod1, mod2))

        Joint_prediction = 0
        for x in range(self.num_models):
            Joint_prediction += 1/self.num_models * outputs[x]
        val_loss =  self.loss_mse(Joint_prediction, target)      
        
        return val_loss



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
        # Load the hillclimb set
        X1, X2, target = load_all_data(self.val_loader)
        X1 = torch.from_numpy(X1)
        X2 = torch.from_numpy(X2)
        target = torch.from_numpy(target)
        for model in models:
            model.eval()
            outputs.append(model(X1, X2))
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


