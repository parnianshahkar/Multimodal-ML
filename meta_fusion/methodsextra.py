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
from itertools import chain, combinations


import random
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import os
import time
import copy


sys.path.append('../')
from meta_fusion.utils import *
from meta_fusion.models import *
from meta_fusion.methods import *
torch.autograd.set_detect_anomaly(True)



class Trainer_Joint(Trainer):
    """
    All hyperparameters are contained in the config file. Modify the config file to change
    the default parameters as needed.
    """
    def __init__(self, config, models, data_loaders):
        super().__init__(config, models, data_loaders)
        # Fix rho=0 for simple benchmarks
        self.rho_list = [0]
        self.best_rho = 0

    #-----------------------#
    #---- Main Functions ---#
    #-----------------------#
    def train(self, mode):
        """
        Main training function that performs the following:
        1. Trains model cohort for user specified list of disagreement penalties;
        2. Chooses and record the best cohort via cross-validation.
        """
        
        print("Start training student cohort...")

        for i in range(self.model_num):
                self.models[i].load_state_dict(self.initial_model_states[i])
                self.optimizers[i].load_state_dict(self.initial_optimizer_states[i])

        self.train_epochs(mode)        

        ############################################
        print("Finished training student cohort!")
        
        # The rest stays the same as original code cause how each model is trained does not matter. 

        if "meta_learner" in self.ensemble_methods:
            print("Training meta learner on the best cohort...")
            self.train_meta_learner() # Given current trained models and their outputs, the best combination? 
            print("Done!")
        
        if "greedy_ensemble" in self.ensemble_methods:
            print("Selecting greedy ensemble on the best cohort...")
            es = EnsembleSelection(self.loss_task, self.val_loader, task_type = self.task_type, 
                                   verbose=self.verbose, random_state=self.random_state)
            # Make sure the best model is loaded
            # _ = self.load_checkpoints(self.best_rho)
            self.ens_idxs = es.build_ensemble_greedy(self.models, weighted=True)
            print("Done!")


    def test(self, test_loader):
        """
        Main testing function that performs the following:
        1. Load the best cohort chosen by cross-validation;
        2. Evaluate the method performance (including various user-specified ensembles) on the test data.
        """

        # Make sure the best model is loaded
        # best_val_task_losses = self.load_checkpoints(self.best_rho)

        if "meta_learner" in self.ensemble_methods:
            self.load_meta_learner()
        
        if self.task_type == "classification":
            pass
            # PAR: LATER
            # results = self.test_classification(self.ensemble_methods+["best_single"], test_loader, best_val_task_losses)
        else:
            results = self.test_regression(self.ensemble_methods+["best_single"], test_loader)
        
        return results


    #-----------------------#
    #----- Test Helpers ----#
    #-----------------------#
    def test_regression(self, ensemble_methods, test_loader,
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

        ###############################

        best_val_task_losses = self.validate()
        best_val_task_losses = [best_val_task_losses[i].avg for i in range(self.model_num)]

        #############################

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
    def train_meta_learner(self):
        """
        Train the meta learner ensemble on the best cohort, meta learner takes the concatenated outputs of the cohort
        and produce a final prediction. 
        """
                
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
                self.save_meta_learner( {
                    'epoch': epoch + 1,
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
        

    def train_epochs(self, mode):
        """
        Training the models. 
        """

        for epoch in range(self.epochs):

            # for scheduler in self.schedulers:
            #     scheduler.step(epoch)
    
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )         
            if mode == 'marginal':
                self.train_one_epoch(epoch)
            else:
                self.train_one_epoch_shapley(epoch)    

                
    # def train_one_epoch_shapley(self, epoch):
    #     """
    #     Train the model for 1 epoch of the training set.

    #     This is used by train() and should not be called manually.
    #     """
    #     batch_time = AverageMeter()
    #     losses = []
    #     task_losses = []
    #     relevant_subsets = []
    #     all_students = set(range(self.model_num))
    #     for i in range(self.model_num):
    #         all_but_one = list(all_students - {i})
    #         relevant_subsets.append([[i] + list(subset) for subset in chain.from_iterable(combinations(all_but_one, r) for r in range(len(all_but_one)+1))])


    #     for i in range(self.model_num):
    #         self.models[i].train() # Set train status
    #         losses.append(AverageMeter()) #takes the average so far 
    #         task_losses.append(AverageMeter())


    #     tic = time.time()
    #     with tqdm(total=self.num_train) as pbar:
    #         for batch in self.train_loader:
    #             modalities, target = batch[:-1], batch[-1]
    #             if self.use_gpu:
    #                 modalities = [mod.cuda() for mod in modalities]
    #                 target = target.cuda()
                
                
    #             for i in range(self.model_num):
    #                 outputs = [model(modalities) for model in self.models]
    #                 loss = 0
    #                 for subset in relevant_subsets[i]:
    #                     Joint_prediction = 0
    #                     for e in subset:
    #                         Joint_prediction += (1/len(subset)) * outputs[e]
    #                     loss +=  self.loss_mse(Joint_prediction, target)/len(relevant_subsets[i])

    #                 # record loss
    #                 losses[i].update(loss.item(), target.size()[0])

    #                 # compute gradients and update optimizer
    #                 self.optimizers[i].zero_grad()
    #                 loss.backward()
    #                 self.optimizers[i].step()
    #                 #self.schedulers[i].step()
                
    #             # measure elapsed time
    #             toc = time.time()
    #             batch_time.update(toc-tic)

    #             # pbar.set_description(
    #             #     (
    #             #         "{:.1f}s - average loss: {:.3f} - average task loss: {:.3f}".format(
    #             #             (toc-tic), np.mean([losses[i].avg for i in range(self.model_num)]), 
    #             #                        np.mean([task_losses[i].avg for i in range(self.model_num)])
    #             #         )
    #             #     )
    #             # )
    #             pbar.update(target.shape[0])

    #         return 

    def train_one_epoch_shapley(self, epoch):
        """
        Train the model for 1 epoch using Shapley value-based losses.
        Uses exact computation for n <= 4, approximate (permutation sampling) for n > 4.
        """
        batch_time = AverageMeter()
        losses = [AverageMeter() for _ in range(self.model_num)]
        
        # Decide: exact vs approximate Shapley
        use_approximate = self.model_num > 4
        
        if use_approximate:
            # Approximate Shapley: sample n*log(n) permutations
            num_permutations = int(self.model_num * np.log(self.model_num))
            print(f"Using approximate Shapley with {num_permutations} permutations")
        else:
            # Exact Shapley: precompute all relevant subsets
            print(f"Using exact Shapley computation")
            relevant_subsets = []
            all_students = set(range(self.model_num))
            for i in range(self.model_num):
                all_but_one = list(all_students - {i})
                subsets_with_i = [[i] + list(subset) for subset in 
                                chain.from_iterable(combinations(all_but_one, r) 
                                for r in range(len(all_but_one) + 1))]
                relevant_subsets.append(subsets_with_i)
        
        # Set all models to train mode
        for model in self.models:
            model.train()
        
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch in self.train_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()
                
                # Zero all gradients
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # Forward pass ONCE
                outputs = [model(modalities) for model in self.models]
                
                # Compute losses based on method
                if use_approximate:
                    # Approximate Shapley via permutation sampling
                    self._compute_approximate_shapley_losses(
                        outputs, target, num_permutations, losses
                    )
                else:
                    # Exact Shapley computation
                    self._compute_exact_shapley_losses(
                        outputs, target, relevant_subsets, losses
                    )
                
                # Update ALL models together
                for optimizer in self.optimizers:
                    optimizer.step()
                
                # Measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)
                tic = toc
                
                # Update progress bar
                avg_loss = np.mean([losses[i].avg for i in range(self.model_num)])
                pbar.set_postfix({
                    'avg_loss': f'{avg_loss:.4f}',
                    'batch_time': f'{batch_time.avg:.3f}s'
                })
                pbar.update(target.shape[0])
        
        return [losses[i].avg for i in range(self.model_num)]


    def _compute_exact_shapley_losses(self, outputs, target, relevant_subsets, losses):
        """Compute exact Shapley values using all coalitions."""
        for i in range(self.model_num):
            loss = 0
            for subset in relevant_subsets[i]:
                Joint_prediction = sum(outputs[e] for e in subset) / len(subset)
                loss += self.loss_mse(Joint_prediction, target) / len(relevant_subsets[i])
            
            # Accumulate gradients
            retain = (i < self.model_num - 1)
            loss.backward(retain_graph=retain)
            losses[i].update(loss.item(), target.size(0))


    def _compute_approximate_shapley_losses(self, outputs, target, num_permutations, losses):
        """Optimized: only compute V(S ∪ {i}), skip V(S)."""
        permutations = [np.random.permutation(self.model_num).tolist() 
                        for _ in range(num_permutations)]
        
        for i in range(self.model_num):
            loss = 0
            
            for perm in permutations:
                pos = perm.index(i)
                coalition_with_i = perm[:pos] + [i]  # S ∪ {i}
                
                pred = sum(outputs[e] for e in coalition_with_i) / len(coalition_with_i)
                loss += self.loss_mse(pred, target) / num_permutations
            
            retain = (i < self.model_num - 1)
            loss.backward(retain_graph=retain)
            losses[i].update(loss.item(), target.size(0))


    # def train_one_epoch(self, epoch):
    #     """
    #     Train the model for 1 epoch of the training set.

    #     This is used by train() and should not be called manually.
    #     """
    #     batch_time = AverageMeter()
    #     losses = []
    #     task_losses = []

    #     for i in range(self.model_num):
    #         self.models[i].train() # Set train status
    #         losses.append(AverageMeter()) #takes the average so far 
    #         task_losses.append(AverageMeter())


    #     tic = time.time()
    #     with tqdm(total=self.num_train) as pbar:
    #         for batch in self.train_loader:
    #             modalities, target = batch[:-1], batch[-1]
    #             if self.use_gpu:
    #                 modalities = [mod.cuda() for mod in modalities]
    #                 target = target.cuda()
                
                
    #             for i in range(self.model_num):
    #                 outputs = [model(modalities) for model in self.models]
    #                 Joint_prediction = 0
    #                 for x in range(self.model_num):
    #                     Joint_prediction += (1/self.model_num) * outputs[x]
    #                 loss =  self.loss_mse(Joint_prediction, target)
                    
    #                 # record loss
    #                 losses[i].update(loss.item(), target.size()[0])

    #                 # compute gradients and update optimizer
    #                 self.optimizers[i].zero_grad()
    #                 loss.backward()
    #                 self.optimizers[i].step()
    #                 #self.schedulers[i].step()
                
    #             # measure elapsed time
    #             toc = time.time()
    #             batch_time.update(toc-tic)

    #             pbar.update(target.shape[0])

    #         return 

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.
        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()  # Single loss meter for joint training
        
        # Set all models to train mode
        for i in range(self.model_num):
            self.models[i].train()
        
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch in self.train_loader:
                modalities, target = batch[:-1], batch[-1]
                if self.use_gpu:
                    modalities = [mod.cuda() for mod in modalities]
                    target = target.cuda()
                
                # Zero gradients for ALL optimizers
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # Forward pass: compute outputs from all models (once!)
                outputs = [model(modalities) for model in self.models]
                
                # Average predictions (joint prediction)
                Joint_prediction = sum(outputs) / self.model_num
                
                # Compute loss
                loss = self.loss_mse(Joint_prediction, target)
                
                # Backward pass
                loss.backward()
                
                # Update all optimizers
                for optimizer in self.optimizers:
                    optimizer.step()
                
                # Optionally step schedulers
                # for scheduler in self.schedulers:
                #     scheduler.step()
                
                # Record loss
                losses.update(loss.item(), target.size(0))
                
                # Measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)
                tic = toc  # Reset for next batch
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'batch_time': f'{batch_time.avg:.3f}s'
                })
                pbar.update(target.shape[0])
        
        # return losses.avg
        return


    #-----------------------#
    #----- S&L Helpers -----#
    #-----------------------#
    def save_checkpoint(self, i, state):
        """
        Save a copy of the model so that it can be loaded at a future date. 
        """
    
        filename = "model" + str(i+1) + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        
        # Get the directory path (parent directories)
        directory = os.path.dirname(ckpt_path)
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Save the checkpoint
        torch.save(state, ckpt_path)



    def save_meta_learner(self, state):
        """
        Save a copy of the meta learner so that it can be loaded at a future date. 
        """
    
        filename = "meta_learner" + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        
        # Get the directory path (parent directories)
        directory = os.path.dirname(ckpt_path)
    
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Save the checkpoint
        torch.save(state, ckpt_path)
        


    def load_meta_learner(self):
        """
        Load a saved meta learner model from a checkpoint.
        """
        filename = "meta_learner" + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)

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

    def validate(self):
        """
        Evaluate the model on the validation set on the current models.
        """
        task_losses = []
        for i in range(self.model_num):
            self.models[i].eval()
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
                
                    # record loss
                    task_losses[i].update(task_loss.item(), target.size()[0])

        return task_losses
    

class Trainer_NCL():
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
        self.rho_list = config["rho_list_ncl"]        
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

        results = {f"indep_{key}": value for key, value in results.items()}
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

        for epoch in range(self.epochs):

            # for scheduler in self.schedulers:
            #     scheduler.step(epoch)
    
            if self.verbose:
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
                )
        
            train_losses, train_task_losses = self.train_one_epoch(epoch, rho)
            val_losses, val_task_losses = self.validate(epoch, rho)
                
                
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

    
    def train_one_epoch(self, epoch, rho):
        """
        Train the model for 1 epoch of the training set.
        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        task_losses = []

        # put all models in train mode and set up meters
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

                # ensure floating targets for regression losses
                target = target.float()

                # forward all models once
                outputs = [model(modalities) for model in self.models]  # each [B, ...]
                O = torch.stack(outputs, dim=0)                          # [M, B, ...]
                M = self.model_num
                B = target.size(0)

                # build total loss as sum of per-model losses
                total_loss = 0.0
                for i in range(M):
                    # base regression loss (e.g., MSE)
                    task_loss = self.loss_task(outputs[i], target)

                    # diversity penalty (push away from peers' mean)
                    if rho > 0 and M > 1:
                        # leave-one-out mean of peers (exclude model i)
                        F_i = (O.sum(dim=0) - outputs[i]) / (M - 1)      # [B, ...]
                        corr_loss = self.loss_mse(outputs[i], F_i)       # scalar
                        loss = task_loss - rho * corr_loss
                    else:
                        loss = task_loss

                    total_loss = total_loss + loss

                    # record per-model metrics
                    # (use .item() to detach; target.size(0) is batch size)
                    losses[i].update(loss.item(), B)
                    task_losses[i].update(task_loss.item(), B)

                # single backward pass for the joint objective
                for opt in self.optimizers:
                    opt.zero_grad()
                total_loss.backward()
                for opt in self.optimizers:
                    opt.step()
                # if you have schedulers, step them here per-optimizer as desired
                # for sch in self.schedulers: sch.step()

                # measure elapsed time and update progress
                toc = time.time()
                batch_time.update(toc - tic)
                pbar.update(B)

        return losses, task_losses


    def validate(self, epoch, rho):
        """
        Evaluate the model cohort on the validation set.
        """
        losses = []
        task_losses = []

        # eval mode + meters
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

                # regression targets as float; align shapes if needed
                target = target.float()
                # e.g., if outputs are [B,1] and target is [B], uncomment:
                # target = target.unsqueeze(-1)

                # forward all models once
                outputs = [model(modalities) for model in self.models]  # each [B, ...]
                O = torch.stack(outputs, dim=0)                          # [M, B, ...]
                M = self.model_num
                B = target.size(0)

                for i in range(M):
                    task_loss = self.loss_task(outputs[i], target)

                    if rho > 0 and M > 1:
                        # Leave-one-out peers' mean (diversity proxy for NCL)
                        F_i = (O.sum(dim=0) - outputs[i]) / (M - 1)      # [B, ...]
                        corr_loss = self.loss_mse(outputs[i], F_i)       # scalar
                        loss = task_loss - rho * corr_loss
                    else:
                        loss = task_loss

                    # record per-model metrics
                    losses[i].update(loss.item(), B)
                    task_losses[i].update(task_loss.item(), B)

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