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
import pdb

sys.path.append('../')
from meta_fusion.methods import *
from meta_fusion.models import *
from meta_fusion.utils import CustomDataset
from meta_fusion.third_party import *




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
        
    noise_ratios : list of float
        A list where each entry is a float in [0, 1] representing the noise level for the corresponding modality. 
        
    beta_bounds : list of float
        A list of floats that represent the upper/lower bound for model coefficients 
        for modalities specific terms, shared terms and interactive terms.
    
    interactive_modalities : list of binary
        A list of binary where each entry indicates if the modality is involved in the interactive terms.

    interactive_prop : float
        A proportion of significant (non zero) interactions
        
    """
    def __init__(self, dim_modalities, dim_latent,
                 noise_ratios,
                 interactive_modalities = None, 
                 interactive_prop = None, 
                 beta_bounds = None,
                 standardize = True, 
                 random_state = None):


        self.dim_modalities = dim_modalities
        self.dim_latent = dim_latent
        self.noise_ratios = noise_ratios
        self.standardize = standardize
        self.num_modalities = len(dim_modalities)
        self.interactive_modalities = interactive_modalities

        if random_state is not None:
            np.random.seed(random_state)

        # Generate the ground truth coefficients
        if beta_bounds is not None:
            beta_bounds = np.abs(beta_bounds)
        else:
            beta_bounds = np.repeat(3, self.num_modalities + 2)

        if interactive_prop is not None: 
            interactive_prop = np.clip(interactive_prop, 0.0, 1.0)
        else:
            interactive_prop = 0.1

        if interactive_modalities is None:
            self.interactive_modalities = np.ones_like(self.dim_modalities)
        
        self.beta_list = []
        self.interaction_size = 1
        # Generate the ground truth coefficients for modality-specific and shared information
        for i in range(self.num_modalities):     
            tmp_beta = np.random.uniform(-beta_bounds[i], beta_bounds[i], self.dim_latent[i])
            self.beta_list.append(tmp_beta)

            if self.interactive_modalities[i]:
                self.interaction_size *= self.dim_latent[i]

        self.betas = np.random.uniform(-beta_bounds[self.num_modalities], beta_bounds[self.num_modalities], self.dim_latent[self.num_modalities])
        self.betai = np.repeat(0, self.interaction_size)
        
        non_zero_indices = np.random.choice(len(self.betai), size=int(len(self.betai) * interactive_prop), replace=False)
        self.betai[non_zero_indices] = np.random.uniform(-beta_bounds[self.num_modalities+1], beta_bounds[self.num_modalities+1], len(non_zero_indices))
        
        # Define the transformation from latent modality space to observed modality space
        self.T_list = []
        for i in range(self.num_modalities):    
            tmp_T = np.random.normal(0,1,(self.dim_latent[i]+self.dim_latent[-1], self.dim_modalities[i]))
            self.T_list.append(tmp_T)


    def _compute_interactions(self, X_latent_list):
        if len(X_latent_list) < 2:
            raise ValueError("At least two latent representations are required.")

        n = X_latent_list[0].shape[0]
        Xi_latent = np.zeros((n, self.interaction_size))
        
        for i in range(n):
            # Start with the first latent vector
            interaction_term = X_latent_list[0][i]
            
            # Compute the outer product iteratively with each subsequent latent vector
            for X_latent in X_latent_list[1:]:
                interaction_term = np.outer(interaction_term, X_latent[i])
            
            # Flatten the final interaction matrix to form the interaction row
            Xi_latent[i, :] = interaction_term.flatten()
        
        return Xi_latent



    def _transform_X(self, X, trans_type):
        if trans_type == "linear":
            return X
        elif trans_type == "quadratic":
            return X **2 - X
        elif trans_type == "sin":
            return 5*np.sin(0.2*X**2) + 0.2*X**2
        else:
            print("Unknown data transformation type!")



    def sample_X(self, n):
        X_latent_list = []
        for i in range(self.num_modalities):    
            X_latent = np.random.normal(0,1,(n,self.dim_latent[i]))   # modality 1 latent
            X_latent_list.append(X_latent)

        XS_latent = np.random.normal(0,1,(n,self.dim_latent[self.num_modalities]))   # shared latent information

        scaler = StandardScaler()

        # Sample the covariates for different modalities
        X_list = []
        for i in range(self.num_modalities):
            noise = np.random.normal(0, 1, (n, self.dim_modalities[i]))
            X_base = np.matmul(np.hstack((X_latent_list[i], XS_latent)), self.T_list[i])
            X_base = scaler.fit_transform(X_base) if self.standardize else X_base
            X = self.noise_ratios[i] * noise + (1 - self.noise_ratios[i]) * X_base
            X_list.append(X)

        return X_list, X_latent_list, XS_latent

    
    def sample_Y(self, X_latent_list, XS_latent, trans_type, mod_prop):
        n = len(X_latent_list[0])
        Y_components = []

        # Transform each modality's latent representation
        for i, X_latent in enumerate(X_latent_list):
            X_latent_trans = self._transform_X(X_latent, trans_type=trans_type[i])
            Y_component = np.matmul(X_latent_trans, self.beta_list[i])
            Y_components.append(mod_prop[i] * Y_component)

        # Transform the shared latent representation
        XS_latent_trans = self._transform_X(XS_latent, trans_type=trans_type[self.num_modalities])
        Y_XS = np.matmul(XS_latent_trans, self.betas)
        Y_components.append(mod_prop[self.num_modalities] * Y_XS)

        # Compute interactions between modalities if needed
        if mod_prop[-1] != 0:
            Xi_latent = self._compute_interactions([X_latent_list[i] for i in range(self.num_modalities) if int(self.interactive_modalities[i])])
            Y_Xi = np.matmul(Xi_latent, self.betai)
            Y_components.append(mod_prop[-1] * Y_Xi)

        # Sum up all components to get the final output and normalize it by letting modality proportions sum up to 1.
        #Y = sum(Y_components)/sum(mod_prop)
        Y = sum(Y_components)
        return Y


    def sample(self, n, trans_type=None, mod_prop=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        # Set default values if not provided
        if trans_type is None:
            trans_type = ["linear"] * (self.num_modalities + 1)  # +1 for the shared latent
        if mod_prop is None:
            mod_prop = [1] * self.num_modalities + [0, 0]  # Last two for shared and interaction

        # Sample latent and observed data
        X_list, latent_list, XS_latent = self.sample_X(n)

        # Sample the output Y
        Y = self.sample_Y(latent_list, XS_latent, trans_type, mod_prop)

        return X_list, latent_list, XS_latent, Y




class PrepareSyntheticData:
    def __init__(self, data_name, test_size=0.2, val_size=0.2):
        self.test_size = test_size
        self.val_size = val_size
        self.data_name = data_name

    def get_data_info(self):
        return [self.dim_modalities, self.n, self.train_num, self.val_num, self.test_num]

    def get_data_loaders(self, n, 
                         trans_type=None, mod_prop=None, random_state=0,
                         train_batch_size=64, val_batch_size=None, test_batch_size=None,
                         **kwargs):

        if val_batch_size is None:
            val_batch_size = n
        if test_batch_size is None:
            test_batch_size = n

        if self.data_name == "regression":
            dim_modalities = kwargs.get('dim_modalities')
            dim_latent = kwargs.get('dim_latent')
            noise_ratios = kwargs.get('noise_ratios')
            interactive_modalities = kwargs.get('interactive_modalities')
            interactive_prop = kwargs.get('interactive_prop')
            beta_bounds = kwargs.get('beta_bounds')

            data_model = DataSharingReg(dim_modalities, dim_latent,
                                        noise_ratios,
                                        interactive_modalities=interactive_modalities,
                                        interactive_prop=interactive_prop, 
                                        beta_bounds=beta_bounds, 
                                        random_state=random_state)

        # Sample data
        X_list, latent_list, XS_latent, Y = data_model.sample(n, trans_type, mod_prop, random_state)
        
        # Split data into train, validation, and test sets
        data_splits = train_test_split(*X_list, *latent_list, XS_latent, Y, test_size=self.test_size, random_state=random_state)
        train_data = [data_splits[i] for i in range(0, len(data_splits),2)]
        test_data = [data_splits[i] for i in range(1, len(data_splits),2)]

        train_splits = train_test_split(*train_data, test_size=self.val_size, random_state=random_state)
        train_data = [train_splits[i] for i in range(0, len(train_splits),2)]
        val_data = [train_splits[i] for i in range(1, len(train_splits),2)]

        # Store data information
        self.dim_modalities = [x.shape[1] for x in X_list]
        self.num_modalities = len(self.dim_modalities)
        self.n = n
        self.train_num = train_data[0].shape[0]
        self.test_num = test_data[0].shape[0]
        self.val_num = val_data[0].shape[0]

        # Create datasets and data loaders
        train_dataset = CustomDataset(np.column_stack(train_data[:self.num_modalities] + [train_data[-1]]), self.dim_modalities)
        val_dataset = CustomDataset(np.column_stack(val_data[:self.num_modalities]+ [val_data[-1]]), self.dim_modalities)
        test_dataset = CustomDataset(np.column_stack(test_data[:self.num_modalities]+ [test_data[-1]]), self.dim_modalities)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        # Prepare oracle datasets
        dim_oracle = np.zeros_like(dim_modalities)
        for i, dim in enumerate(dim_latent[:-1]):
            dim_oracle[i] = dim
        dim_oracle[-1] += dim_latent[-1]  # merge the shared latent features with last modality to avoid redundancy

        oracle_train_data = np.column_stack(train_data[self.num_modalities:])
        oracle_val_data = np.column_stack(val_data[self.num_modalities:])
        oracle_test_data = np.column_stack(test_data[self.num_modalities:])

        oracle_train_dataset = CustomDataset(oracle_train_data, dim_oracle)
        oracle_val_dataset = CustomDataset(oracle_val_data, dim_oracle)
        oracle_test_dataset = CustomDataset(oracle_test_data, dim_oracle)

        oracle_train_loader = DataLoader(oracle_train_dataset, batch_size=train_batch_size, shuffle=True)
        oracle_val_loader = DataLoader(oracle_val_dataset, batch_size=val_batch_size, shuffle=False)
        oracle_test_loader = DataLoader(oracle_test_dataset, batch_size=test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, oracle_train_loader, oracle_val_loader, oracle_test_loader