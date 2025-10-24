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
from sklearn.metrics import mean_squared_error
import random
import time
from tqdm import tqdm

sys.path.append('../')
from meta_fusion.archive_utils import AverageMeter, load_all_data, calculate_accuracy
from meta_fusion.archive_methods import Trainer


class Benchmarks(Trainer):
    """
    Implements three fusion benchmarks using the general framework defined in the Trainer class.
    
    Early fusion: raw modalities are concatenated at the input stage. 
    Late fusion: models are trained separately for each modality, and outputs are ensembled to get final decision.
    Special version of cooperative learning: aside from the disagreement penalty, each model is trained to
    match the grouth truth, instead of the partial residual as in the original cooperative learning. 
    """
    def __init__(self, config, models, model_dims, data_loaders):
        super().__init__(config, models, data_loaders)

        # Only three models are expected, two using only one modality and one early fusion model
        assert len(models) == len(model_dims) == 3, "Exactly three models are expected!"

        # Identify early fusion model vs single modality models by checking dimensions
        for i, (dim1, dim2) in enumerate(model_dims):
            if dim1 != 0 and dim2 != 0:
                self.idx_early = i
            elif dim1 == 0:   # only modality 2 is used 
                self.idx_mod2 = i
            elif dim2 == 0:   # only modality 1 is used
                self.idx_mod1 = i

        # Fix rho=0 for simple benchmarks: No dependencies across models    
        self.rho_list = [0]



    def train(self):
        """
        Main training function that performs the following:
        1. trains benchmark models and a list of rhos provided by the user;
        2. chooses and record the best rho via cross-validation.
        """
        
        print("Start training benchmark models...")
        super().train_all_rhos()
        print("Finished training benchmark models!")
    
    

    def test(self, test_loader, benchmarks = None):
        """
        Evaluate the benchmark performance on the test data 
        """
        valid_benchmarks = ["mod1", "mod2", "early_fusion", "late_fusion"]
        
        if benchmarks == None: 
            benchmarks = valid_benchmarks
        elif not set(benchmarks).issubset(valid_benchmarks):
            raise ValueError("Some benchmarks are invalid.")
        
        result_list = []
        if self.task_type == "classification":
            # Load models with no penalty for non-coop benchmarks
            _ = self.load_checkpoints(0)
            tmp_result = self.test_classification(benchmarks, test_loader)
            result_list.append(tmp_result)
        else:
            # Load models with no penalty for non-coop benchmarks
            _ = self.load_checkpoints(0)
            tmp_result = self.test_regression(benchmarks, test_loader)
            result_list.append(tmp_result)

        results = {}
        for d in result_list:
            results.update(d)    
        
        return results
    


    def test_regression(self, benchmarks, test_loader):
        """
        Evaluate the performance of the benchmarks on the validation set for regression tasks.
        """

        # Dictionary to store the test MSE for benchmarks
        full_mse = {}

        method_mse = []
        for benchmark in benchmarks:
            method_mse.append(AverageMeter())

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(test_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(mod1, mod2)
                    outputs.append(output)

                outputs_stack = torch.stack(outputs)  # Shape: (self.model_num, batch_size, regression_outputs)

                for i, benchmark in enumerate(benchmarks):
                    if benchmark == "mod1":   # Extract the modality 1 output
                        final_output = outputs_stack[self.idx_mod1]
                    elif benchmark == "mod2":   # Extract the modality 2 output
                        final_output = outputs_stack[self.idx_mod2]
                    elif benchmark == "early_fusion":   # Extract the early fusion output
                        final_output = outputs_stack[self.idx_early]
                    elif benchmark == "late_fusion":   
                        # [Note] Assumed simple average of late fusion, add more ensembles if needed
                        final_output = (outputs_stack[self.idx_mod1] + 
                                        outputs_stack[self.idx_mod2]) / 2  # Shape: (batch_size, regression_outputs)

                    method_mse[i].update(self.loss_mse(final_output, target), target.size()[0])
            
            # for i, benchmark in enumerate(benchmarks):
            #     full_mse[benchmark] = method_mse[i].avg

            for i, benchmark in enumerate(benchmarks):
                # Convert to float if the value is a tensor
                mse_value = method_mse[i].avg
                if hasattr(mse_value, 'item'):  # Check if it's a tensor with the 'item' method
                    full_mse[benchmark] = mse_value.item()
                else:
                    full_mse[benchmark] = float(mse_value)  # Convert to float if not already

            if self.verbose:
                for key, value in full_mse.items():
                    print(f'Method: ({key}), Test_MSE: {value}')
 
            return full_mse



    def test_classification(self, benchmarks, test_loader):
        """
        Validate the performance of the benchmarks on the validation set for classification tasks.
        """

        # Dictionary to store the test MSE for cohorts and ensemble methods 
        full_accuracy = {}

        method_accuracy = []
        for benchmark in benchmarks:
            method_accuracy.append(AverageMeter())

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(test_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(mod1, mod2)
                    outputs.append(output)
                outputs_stack = torch.stack(outputs)  # Shape: (self.model_num, batch_size, num_classes)

                # [*] For now assumes simple averaging of the logits, add more ensemble methods for
                # late fusion and coop learning as needed.

                for i, benchmark in enumerate(benchmarks):
                    if benchmark == "mod1":   # Extract the modality 1 output
                        final_output = outputs_stack[self.idx_mod1]
                    elif benchmark == "mod2":   # Extract the modality 2 output
                        final_output = outputs_stack[self.idx_mod2]
                    elif benchmark == "early_fusion":   # Extract the early fusion output
                        final_output = outputs_stack[self.idx_early]
                    elif benchmark == "late_fusion":   
                        # [Note] Assumed simple average of late fusion, add more ensembles if needed
                        final_output = (outputs_stack[self.idx_mod1] + 
                                        outputs_stack[self.idx_mod2]) / 2  # Shape: (batch_size, classification_outputs)

                    method_accuracy[i].update(calculate_accuracy(final_output, target), target.size()[0])

            for i, benchmark in enumerate(benchmarks):
                full_accuracy[benchmark] = method_accuracy[i].avg
                
            if self.verbose:
                for key, value in full_accuracy.items():
                    print(f'Method: ({key}), Test_Accuracy: {value}')

            return full_accuracy





class Coop(Trainer):
    """
    Implements original oversion of the cooperative learning via alternating gradient descent. 
    Aside from the disagreement penalty, each model is trained to match the partial residual of 
    the ground truth.
    """
    def __init__(self, config, models, model_dims, data_loaders):
        super().__init__(config, models, data_loaders)

        # Identify model types by checking dimensions
        for i, (dim1, dim2) in enumerate(model_dims):
            if dim1 != 0 and dim2 != 0:
                self.idx_early = i
            elif dim1 == 0:   # only modality 2 is used 
                self.idx_mod2 = i
            elif dim2 == 0:   # only modality 1 is used
                self.idx_mod1 = i
        
        if len(models) == 2:
            self.idx_early = None
        
        # Original cooperative learning only supports regression
        self.task_type = "regression"

        # range of rho should be [0,1)
        self.rho_list = [rho for rho in self.rho_list if 0 <= rho < 1]


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
        Evaluate the cooperative learning performance on the test data 
        """

        # Make sure the best model is loaded
        _ = self.load_checkpoints(self.best_rho)

        results = self.test_regression(test_loader)
        
        return results



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

        for i in range(self.model_num):
            self.models[i].train()
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
                for i in range(self.model_num):

                    # model fitted on partial residuals
                    if i == self.idx_early:
                        final_output = outputs[self.idx_mod1].detach() + outputs[self.idx_mod2].detach() + outputs[i]
                        divergence_loss = rho / (1 - rho) * self.loss_mse(outputs[i],torch.zeros_like(outputs[i]))
                    
                    elif i == self.idx_mod1:
                        final_output = outputs[self.idx_mod2].detach() + outputs[i] if not self.idx_early else\
                                      outputs[self.idx_mod2].detach() + outputs[self.idx_early].detach() + outputs[i]
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod2].detach())
                    
                    elif i == self.idx_mod2:
                        final_output = outputs[self.idx_mod1].detach() + outputs[i] if not self.idx_early else\
                                      outputs[self.idx_mod1].detach() + outputs[self.idx_early].detach() + outputs[i]
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod1].detach())

                    task_loss = self.loss_mse(final_output, target)
                    loss = task_loss + divergence_loss
                    
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
                
                final_output = torch.sum(torch.stack(outputs), dim=0)

                for i in range(self.model_num):
                    # model fitted on partial residuals
                    if i == self.idx_early:
                        divergence_loss = rho / (1 - rho) * self.loss_mse(outputs[i],torch.zeros_like(outputs[i]))
                    
                    elif i == self.idx_mod1:
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod2])
                    
                    elif i == self.idx_mod2:
                        divergence_loss  = rho * self.loss_mse(outputs[i], outputs[self.idx_mod1])

                    task_loss = self.loss_mse(final_output, target)
                    loss = task_loss + divergence_loss

                    # record loss
                    losses[i].update(loss.item(), target.size()[0])
                    task_losses[i].update(task_loss.item(), target.size()[0])

            # log to tensorboard for every epoch
            if self.use_tensorboard:
                for i in range(self.model_num):
                    log_value('val_loss_%d' % (i+1), losses[i].avg, epoch+1)
                    log_value('val_acc_%d' % (i+1), accs[i].avg, epoch+1)

        return losses, task_losses


    
    def test_regression(self, test_loader):
        """
        Evaluate the performance of the ensemble on the validation set for regression tasks.
        """

        method_mse = AverageMeter()

        for i in range(self.model_num):
            self.models[i].eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch, (mod1, mod2, target) in enumerate(test_loader):
                if self.use_gpu:
                    mod1, mod2, target = mod1.cuda(), mod2.cuda(), target.cuda()

                # Forward pass for each model
                outputs = []
                for i, model in enumerate(self.models):
                    output = model(mod1, mod2)
                    outputs.append(output)

                final_output = torch.sum(torch.stack(outputs), dim=0)
                method_mse.update(self.loss_mse(final_output, target), target.size()[0])
            
            mse_value = method_mse.avg
            if hasattr(mse_value, 'item'):
                mse_value = mse_value.item()
            else:
                mse_value =  float(mse_value)

            if self.verbose:
                print(f'Method: ({"coop"}), Test_MSE: {mse_value}')

            
            #return {"coop":method_mse.avg}
            return {"coop":mse_value}




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