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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('../')
from meta_fusion.archive_methods import *
from meta_fusion.archive_models import *
from meta_fusion.archive_utils import CustomDataset




class DataSharingReg():
    """
    DataSharingReg is a class that models data-sharing across different modalities 
    using latent variables. 

    Parameters
    ----------
    dim_modalities : list of int
        A list where each entry represents the dimensionality of a particular modality.
        
    dim_latent : int
        A list where each entry represents the latent dimensionality of a particular modality.
        
    niose_ratios : list of float
        A list where each entry is a float in [0, 1] representing the noise level for the corresponding modality. 
        
    beta_bounds : list of float
        A list of floats that represent the upper/lower bound for model coefficients 
        for modalities alone and interactive terms.
        
    iteractive_prop : float
        A proportion of significant (non zero) interactions
        
    """
    def __init__(self, dim_modalities, dim_latent,
                 niose_ratios,
                 interactive_prop = None, 
                 beta_bounds = None,
                 standardize = True, 
                 random_state = None):

        # Assume for now this class only supports two modalities

        self.dim_modalities = dim_modalities
        self.dim_latent = dim_latent
        self.niose_ratios = niose_ratios
        self.standardize = standardize

        if random_state is not None:
            np.random.seed(random_state)

        # Generate the ground truth coefficients
        if beta_bounds is not None:
            beta_bounds = np.abs(beta_bounds)
        else:
            beta_bounds = np.repeat(3, 4)

        if interactive_prop is not None: 
            interactive_prop = np.clip(interactive_prop, 0.0, 1.0)
        else:
            interactive_prop = 0.1
        
        self.beta1 = np.random.uniform(-beta_bounds[0], beta_bounds[0], self.dim_latent[0])    # modality 1 specific
        self.beta2 = np.random.uniform(-beta_bounds[1], beta_bounds[1], self.dim_latent[1])    # modality 2 specific
        self.betas = np.random.uniform(-beta_bounds[2], beta_bounds[2], self.dim_latent[2])    # shared information
        self.betai = np.repeat(0, self.dim_latent[0] * self.dim_latent[1])
        
        non_zero_indices = np.random.choice(len(self.betai), size=int(len(self.betai) * interactive_prop), replace=False)
        self.betai[non_zero_indices] = np.random.uniform(-beta_bounds[3], beta_bounds[3], len(non_zero_indices))
        
        # Define the transformation from latent modality space to observed modality space
        self.T1 = np.random.normal(0,1,(self.dim_latent[0]+self.dim_latent[2], self.dim_modalities[0]))
        self.T2 = np.random.normal(0,1,(self.dim_latent[1]+self.dim_latent[2], self.dim_modalities[1]))



    def compute_interactions(self, X1_latent, X2_latent):
        n = X1_latent.shape[0]
        interaction_size = self.dim_latent[0] * self.dim_latent[1]
        Xi_latent = np.zeros((n, interaction_size))
        
        for i in range(n):
            # Compute the outer product between the ith row of X1_latent and X2_latent
            outer_product = np.outer(X1_latent[i], X2_latent[i])
            # Flatten the outer product matrix to form the interaction row
            Xi_latent[i, :] = outer_product.flatten()
        
        return Xi_latent



    def transform_X(self, X, trans_type):
        if trans_type == "linear":
            return X
        elif trans_type == "quadratic":
            return X **2 - X
        else:
            print("Unknown data transformation type!")



    def sample_X(self, n):        
        X1_latent = np.random.normal(0,1,(n,self.dim_latent[0]))   # modality 1 latent
        X2_latent = np.random.normal(0,1,(n,self.dim_latent[1]))   # modality 2 latent
        XS_latent = np.random.normal(0,1,(n,self.dim_latent[2]))   # shared latent information

        scaler = StandardScaler()

        # Sample the covariates for different modalities
        noise1 = np.random.normal(0, 1, (n, self.dim_modalities[0]))
        X1_base = np.matmul(np.hstack((X1_latent, XS_latent)), self.T1)
        X1_base = scaler.fit_transform(X1_base) if self.standardize else X1_base  
        X1 = self.niose_ratios[0] * noise1 + (1 - self.niose_ratios[0]) * X1_base
        
        noise2 = np.random.normal(0, 1, (n, self.dim_modalities[1]))
        X2_base = np.matmul(np.hstack((X2_latent, XS_latent)), self.T2)
        X2_base = scaler.fit_transform(X2_base) if self.standardize else X2_base  
        X2 = self.niose_ratios[1] * noise2 + (1 - self.niose_ratios[1]) * X2_base
        return X1, X2, X1_latent, X2_latent, XS_latent



    def sample_Y(self, X1_latent, X2_latent, XS_latent, 
                 trans_type = ["linear", "linear", "linear"], mod_prop = [1,1,0,0]):
        n = len(X1_latent)
        
        X1_latent_trans = self.transform_X(X1_latent, trans_type = trans_type[0])
        X2_latent_trans = self.transform_X(X2_latent, trans_type = trans_type[1])
        XS_latent_trans = self.transform_X(XS_latent, trans_type = trans_type[2])
        
        Y_X1 = np.matmul(X1_latent_trans,self.beta1)
        Y_X2 = np.matmul(X2_latent_trans,self.beta2)
        Y_XS = np.matmul(XS_latent_trans,self.betas)
        Xi_latent = self.compute_interactions(X1_latent, X2_latent)
        Y_Xi = np.matmul(Xi_latent,self.betai)

        Y = mod_prop[0] * Y_X1 + mod_prop[1] * Y_X2 + mod_prop[2] * Y_XS + mod_prop[3] * Y_Xi
        return Y



    def sample(self, n, trans_type = ["linear", "linear"], mod_prop = [1,1,0], random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        
        X1, X2, X1_latent, X2_latent, XS_latent = self.sample_X(n)
        Y = self.sample_Y(X1_latent, X2_latent, XS_latent, trans_type, mod_prop)
        
        return X1, X2, X1_latent, X2_latent, XS_latent, Y





class Prepare_synthetic_data:
    def __init__(self, data_name, test_size = 0.2, val_size = 0.2):
        self.test_size = test_size
        self.val_size = val_size
        self.data_name = data_name

    def get_data_info(self):
        return [self.d1, self.d2, self.n, self.train_num, self.val_num, self.test_num] 

    def get_data_loaders(self, n, 
                    trans_type = ["linear", "linear", "linear"], mod_prop = [1,1,0,0], random_state = 0,
                    train_batch_size = 64, val_batch_size = None, test_batch_size = None,
                    **kwargs):

        if val_batch_size is None:
            val_batch_size = n
        if test_batch_size is None:
            test_batch_size = n
        
        if self.data_name == "regression":
            dim_modalities = kwargs.get('dim_modalities')
            dim_latent = kwargs.get('dim_latent')
            noise_ratios = kwargs.get('noise_ratios')
            interactive_prop = kwargs.get('interactive_prop')
            beta_bounds = kwargs.get('beta_bounds')

            data_model = DataSharingReg(dim_modalities, dim_latent,
                                        noise_ratios,
                                        interactive_prop = interactive_prop, 
                                        beta_bounds = beta_bounds, 
                                        random_state = random_state)

        
        X1, X2, X1_latent, X2_latent, XS_latent, Y = data_model.sample(n, trans_type=trans_type, mod_prop=mod_prop, random_state = random_state)
        
        X1_tmp, X1_test, X2_tmp, X2_test, \
        X1_latent_tmp, X1_latent_test, X2_latent_tmp, X2_latent_test,  XS_latent_tmp, XS_latent_test, \
        Y_tmp, Y_test = train_test_split(X1, X2, X1_latent, X2_latent, XS_latent, Y, test_size=self.test_size, random_state=random_state)
        
        X1_train, X1_val, X2_train, X2_val, \
        X1_latent_train, X1_latent_val, X2_latent_train, X2_latent_val, XS_latent_train, XS_latent_val, \
        Y_train, Y_val = train_test_split(X1_tmp, X2_tmp, X1_latent_tmp, X2_latent_tmp, XS_latent_tmp, Y_tmp, test_size=self.val_size, random_state=random_state)

        train_data = np.column_stack((X1_train, X2_train, Y_train))
        val_data = np.column_stack((X1_val, X2_val, Y_val))
        test_data = np.column_stack((X1_test, X2_test, Y_test))

        oracle_train_data = np.column_stack((X1_latent_train, X2_latent_train, XS_latent_train, Y_train))
        oracle_val_data = np.column_stack((X1_latent_val, X2_latent_val, XS_latent_val, Y_val))
        oracle_test_data = np.column_stack((X1_latent_test, X2_latent_test, XS_latent_test, Y_test))

        train_dataset = CustomDataset(train_data, dim_modalities)
        val_dataset = CustomDataset(val_data, dim_modalities)
        test_dataset = CustomDataset(test_data, dim_modalities)

        dim_oracle = np.zeros_like(dim_modalities)
        dim_oracle[0] = dim_latent[0]
        dim_oracle[1] = dim_latent[1]+dim_latent[2]   # merge the shared latent features with modality 2 latent features to avoid redundancy
        
        oracle_train_dataset = CustomDataset(oracle_train_data, dim_oracle)
        oracle_val_dataset = CustomDataset(oracle_val_data, dim_oracle)
        oracle_test_dataset = CustomDataset(oracle_test_data, dim_oracle)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        oracle_train_loader = DataLoader(oracle_train_dataset, batch_size=train_batch_size, shuffle=True)
        oracle_val_loader = DataLoader(oracle_val_dataset, batch_size=val_batch_size, shuffle=False)
        oracle_test_loader = DataLoader(oracle_test_dataset, batch_size=test_batch_size, shuffle=False)

        # Store data information
        self.d1 = X1_train.shape[1]
        self.d2 = X2_train.shape[1]
        self.n = n
        self.train_num = X1_train.shape[0]
        self.test_num = X1_test.shape[0]
        self.val_num = X1_val.shape[0]
        self.dim_modalities = [self.d1, self.d2]

        return train_loader, val_loader, test_loader, oracle_train_loader, oracle_val_loader, oracle_test_loader

