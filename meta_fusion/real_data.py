import numpy as np
import pandas as pd
import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from sklearn.decomposition import PCA

import pdb

sys.path.append('../')
from meta_fusion.methods import *
from meta_fusion.models import *
from meta_fusion.utils import CustomDataset

sys.path.append('../data/NACC/')
from data_cleaning_util import *


class PrepareNACC:
    def __init__(self, fine_grained = False, test_size=0.2, val_size=0.2):
        self.test_size = test_size
        self.val_size = val_size

        if fine_grained:
            data_dir = '../data/NACC/preprocessed_v3/'
            self.modalities = [
                pd.read_csv(data_dir + 'uds_history.csv'),  # Modality 1
                pd.read_csv(data_dir + 'uds_survey.csv'),   # Modality 2
                pd.read_csv(data_dir + 'uds_testing.csv'),  # Modality 3 
                pd.read_csv(data_dir + 'mrisbm.csv'),       # Modality 4
            ]
            self.Y = pd.read_csv(data_dir + 'label.csv').astype(int) - 1  # Convert labels from 1-4 to 0-3
            self.classification = True
            self.num_modalities = 4
        else:
            data_dir = '../data/NACC/preprocessed_v2/'
            self.modalities = [
                pd.read_csv(data_dir + 'uds_history.csv'),  # Modality 1
                pd.read_csv(data_dir + 'uds_survey.csv'),   # Modality 2
                pd.read_csv(data_dir + 'mrisbm.csv'),       # Modality 3
            ]
            self.Y = pd.read_csv(data_dir + 'label.csv').astype(int) - 1  # Convert labels from 1-4 to 0-3
            self.classification = True
            self.num_modalities = 3

    def get_data_info(self):
        return [self.dim_modalities, self.n, self.train_num, self.val_num, self.test_num]
        

    def get_data_loaders(self, train_batch_size=64, val_batch_size=None, test_batch_size=None, random_state=0):
        '''
        Perform data splitting and prepare data loaders for model training
        '''
        # Split the data into training, validation, and test sets
        data_splits = train_test_split(
            *self.modalities, self.Y,
            test_size=self.test_size,
            random_state=random_state
        )

        # Extract training and test sets for each modality and the labels
        X_train_list = [data_splits[i] for i in range(0, len(data_splits)-2, 2)]
        X_test_list = [data_splits[i] for i in range(1, len(data_splits)-2, 2)]
        Y_train = data_splits[-2]
        Y_test = data_splits[-1]

        # Impute missing data for each modality
        for i in range(self.num_modalities):
            if i < self.num_modalities-1:  # First few modalities are UDS-derived
                X_train_list[i], X_test_list[i] = impute_invariant_features(X_train_list[i], X_test_list[i])
                preprocessor = imputer_by_feature_type(X_train_list[i]).fit(X_train_list[i])
                X_train_list[i] = pd.DataFrame(preprocessor.transform(X_train_list[i]), columns=preprocessor.get_feature_names_out(), index=X_train_list[i].index.values)
                X_test_list[i] = pd.DataFrame(preprocessor.transform(X_test_list[i]), columns=preprocessor.get_feature_names_out(), index=X_test_list[i].index.values)
            else:  # SBM modality
                X_train_list[i], X_test_list[i] = impute_mrisbm_features(X_train_list[i], X_test_list[i])

        # Further split the training data into training and validation sets
        train_splits = train_test_split(
            *X_train_list, Y_train,
            test_size=self.val_size,
            random_state=random_state
        )
        # Extract training and val sets for each modality and the labels
        X_train_list = [train_splits[i] for i in range(0, len(train_splits)-2, 2)]
        X_val_list = [train_splits[i] for i in range(1, len(train_splits)-2, 2)]
        Y_train = train_splits[-2]
        Y_val = train_splits[-1]

        # Store data information
        self.dim_modalities = [x.shape[1] for x in X_train_list]
        self.num_modalities = len(self.dim_modalities)
        self.n = len(self.Y)
        self.train_num = X_train_list[0].shape[0]
        self.test_num = X_test_list[0].shape[0]
        self.val_num = X_val_list[0].shape[0]

        # Create datasets and data loaders
        train_data = np.column_stack(X_train_list + [Y_train])
        val_data = np.column_stack(X_val_list + [Y_val])
        test_data = np.column_stack(X_test_list + [Y_test])

        train_dataset = CustomDataset(train_data, self.dim_modalities, self.classification)
        val_dataset = CustomDataset(val_data, self.dim_modalities, self.classification)
        test_dataset = CustomDataset(test_data, self.dim_modalities, self.classification)
    
        if val_batch_size is None:
            val_batch_size = self.n
        if test_batch_size is None:
            test_batch_size = self.n
        
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
        return train_loader, val_loader, test_loader





#########################
# Neuron data functions #
#########################

class CustomGraphDataset(Dataset):
    def __init__(self, data, dim_modalities, classification=False, lfp_adj=None, spk_adj=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.dim_modalities = dim_modalities
        self.classification = classification
        self.lfp_adj = lfp_adj
        self.spk_adj = spk_adj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_idx = 0
        graph_data = []
        
        for i, dim in enumerate(self.dim_modalities):
            end_idx = start_idx + np.prod(dim)
            modality_data = self.data[idx, start_idx:end_idx].reshape(dim)
            
            # Use the corresponding adjacency matrix
            adj = self.lfp_adj if i == 0 else self.spk_adj
            edge_index = adj.nonzero().t().contiguous()
            
            # Create PyTorch Geometric Data object
            graph = Data(x=modality_data, edge_index=edge_index)
            graph_data.append(graph)
            
            start_idx = end_idx
        
        Y = self.data[idx, start_idx:]
        
        if self.classification:
            Y = Y.long().squeeze()

        return tuple(graph_data) + (Y,)



def custom_collate(batch):
    lfp_graphs = [item[0] for item in batch]
    spk_graphs = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    lfp_batch = Batch.from_data_list(lfp_graphs)
    spk_batch = Batch.from_data_list(spk_graphs)

    return lfp_batch, spk_batch, torch.stack(labels)




class PrepareNeuronData:
    def __init__(self, rat_name, data_dir="../data/Neuron2021/Shahbaba_2021_dataset/stattensor", 
                 test_size=0.2, val_size=0.2, oversample=True, time_range=(200, 250), 
                 adj_thres=[0.4, 0.2], task_type='binary', classes_to_keep=None, classes_to_remove=None,
                 is_graph=True, reduce_dim=False, n_lfp_features=15, epsilon=1e-10):
        """
        Initialize data preparation with options for binary or multi-class classification
        
        Parameters:
        rat_name (str): Name of the rat
        data_dir (str): Directory containing the data
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        oversample (bool): Whether to oversample minority classes (for binary task)
        time_range (tuple): Range of time points to use
        adj_thres (list): Thresholds for adjacency matrices [lfp_threshold, spike_threshold]
        task_type (str): 'binary' or 'multiclass' - determines which column of bvr to use for labels
        classes_to_keep (list): List of classes to include (for multiclass task)
        classes_to_remove (list): List of classes to exclude (for multiclass task)
        reduce_dim (bool): Whether to perform dimension reduction
        n_lfp_features (int): Number of PCA components for lfp data
        epsilon (float): Threshold for determining active neurons
        """
        self.rat_name = rat_name
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.oversample = oversample
        self.time_range = time_range
        self.adj_thres = adj_thres
        self.task_type = task_type
        self.classes_to_keep = classes_to_keep
        self.classes_to_remove = classes_to_remove
        self.is_graph = is_graph
        self.reduce_dim = reduce_dim
        self.n_lfp_features = n_lfp_features
        self.epsilon = epsilon

        self.load_data()
        self.preprocess_data()
        if self.reduce_dim:
            self.reduce_dimensions()
        self.estimate_adjacency_matrices()
        
        # Filter classes for multiclass task if needed
        if self.task_type == 'multiclass' and (self.classes_to_keep is not None or self.classes_to_remove is not None):
            self.filter_classes()

        self.n_trials, self.n_nodes_lfp, self.dim_lfp = self.lfp.shape
        self.n_nodes_spk, self.dim_spk = self.spk.shape[1], self.spk.shape[2]
        
        self.classification = True
        self.num_modalities = 2

        if self.is_graph:
            # Preserve the graph structure
            self.dim_modalities = [
                (self.n_nodes_lfp, self.dim_lfp),
                (self.n_nodes_spk, self.dim_spk)
            ]
        else:
            # Flatten the data
            self.dim_modalities = [
                self.n_nodes_lfp * self.dim_lfp, 
                self.n_nodes_spk * self.dim_spk
            ]

    def get_data_info(self):
        return {
            "rat_name": self.rat_name,
            "dim_modalities": self.dim_modalities,
            "is_graph": self.is_graph,
            "n_trials": self.n_trials,
            "train_num": self.train_num if hasattr(self, 'train_num') else None,
            "val_num": self.val_num if hasattr(self, 'val_num') else None,
            "test_num": self.test_num if hasattr(self, 'test_num') else None,
            "n_nodes_lfp": self.n_nodes_lfp,
            "dim_lfp": self.dim_lfp,
            "n_nodes_spk": self.n_nodes_spk,
            "dim_spk": self.dim_spk,
            "n_classes": len(np.unique(self.labels)),
            "class_distribution": dict(Counter(self.labels)),
            "task_type":self.task_type,
            "oversample": self.oversample,
            "time_range": self.time_range,
            "adjacency_threshold": self.adj_thres
        }


    def get_adjacency_matrices(self):
        return self.lfp_adj, self.spk_adj


    def load_data(self):
        """Load raw data from files"""
        lfp = np.load(f"{self.data_dir}/{self.rat_name}/{self.rat_name.lower()}_lfp_data_sampled.npy", allow_pickle=True).swapaxes(1, 2)
        spk = np.load(f"{self.data_dir}/{self.rat_name}/{self.rat_name.lower()}_spike_data_binned.npy", allow_pickle=True)
        bvr = np.load(f"{self.data_dir}/{self.rat_name}/{self.rat_name.lower()}_trial_info.npy", allow_pickle=True)
        
        self.lfp = lfp[:, :, self.time_range[0]:self.time_range[1]]
        self.spk = spk[:, :, self.time_range[0]:self.time_range[1]]
        self.bvr = bvr


    def filter_classes(self):
        """Filter data to keep only specified classes or remove specified classes (for multiclass task)"""
        # Determine which classes to keep
        if self.classes_to_keep is not None:
            # Keep only the specified classes
            keep_mask = np.zeros_like(self.labels, dtype=bool)
            for cls in self.classes_to_keep:
                keep_mask = keep_mask | (self.labels == cls)
            print(f"Keeping only classes: {self.classes_to_keep}")
        elif self.classes_to_remove is not None:
            # Remove the specified classes
            keep_mask = np.ones_like(self.labels, dtype=bool)
            for cls in self.classes_to_remove:
                keep_mask = keep_mask & (self.labels != cls)
            print(f"Removing classes: {self.classes_to_remove}")
        else:
            # Keep all classes if neither parameter is specified
            keep_mask = np.ones_like(self.labels, dtype=bool)
            print("Keeping all classes")
        
        # Apply the filter
        original_count = len(self.labels)
        self.lfp = self.lfp[keep_mask]
        self.spk = self.spk[keep_mask]
        self.bvr = self.bvr[keep_mask]
        self.labels = self.labels[keep_mask]
        filtered_count = len(self.labels)
        
        print(f"Filtered {original_count - filtered_count} trials")
        print(f"Remaining trials: {filtered_count}")
        
        # Print new class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution after filtering:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} trials")


    def preprocess_data(self):
        """Process data to keep only correct trials"""
        correct_index = np.where(self.bvr[:, 0] == 1)[0]
        self.lfp = self.lfp[correct_index, :, :]
        self.spk = self.spk[correct_index, :, :]
        self.bvr = self.bvr[correct_index, :]
        
        # Update labels after filtering correct trials
        if self.task_type == 'binary':
            self.labels = self.bvr[:, 1]
        else:  # multiclass
            # Only take the Inseq trials
            inseq_index = np.where(self.bvr[:, 1] == 1)[0]
            self.lfp = self.lfp[inseq_index, :, :]
            self.spk = self.spk[inseq_index, :, :]
            self.bvr = self.bvr[inseq_index, :]
            
            self.labels = self.bvr[:, 3]
            # Shift labels to be zero-indexed
            self.labels = self.labels - 1

        # Print original class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Original {self.task_type} class distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} trials")

    
    def reduce_dimensions(self):
        """
        Perform dimension reduction on LFP and spike data:
        - LFP: Apply PCA to each node separately
        - Spike: compute average firing rate for active nodes
        """
        print("Performing dimension reduction...")
        
        # 1. LFP dimension reduction using PCA for each node
        n_trials, n_nodes, n_timepoints = self.lfp.shape
        reduced_lfp = np.zeros((n_trials, n_nodes, 0))  # Will expand as we add components
        
        for node_idx in range(n_nodes):
            node_data = self.lfp[:, node_idx, :]
            
            # Apply PCA to this node's data
            pca = PCA(n_components=self.n_lfp_features)
            node_reduced = pca.fit_transform(node_data)
            
            # Expand reduced_lfp
            if node_idx == 0:
                reduced_lfp = np.zeros((n_trials, n_nodes, node_reduced.shape[1]))
            
            # Store the reduced features
            reduced_lfp[:, node_idx, :] = node_reduced
        
        self.lfp = reduced_lfp
        print(f"  LFP data reduced to shape: {self.lfp.shape}")


        # 2. Spike data reduction using average firing rates
        n_trials, n_neurons, n_timepoints = self.spk.shape
        
        # Compute average firing rates for each neuron
        firing_rates = np.mean(self.spk, axis=2)  # Shape: (n_trials, n_neurons)
        
        # Identify active neurons (neurons with variance in firing rate)
        neuron_variances = np.var(firing_rates, axis=0)
        active_neurons = np.where(neuron_variances > self.epsilon)[0]
        
        if len(active_neurons) == 0:
            raise ValueError("No active neurons found! All neurons have constant firing rates.")
        
        # Keep only active neurons and reshape to have a single time point (the average)
        firing_rates_active = firing_rates[:, active_neurons]
        reduced_spk = firing_rates_active.reshape(n_trials, len(active_neurons), 1)
        
        self.spk = reduced_spk
        print(f"  Spike data reduced to average firing rates")
        print(f"  Kept {len(active_neurons)} active neurons out of {n_neurons} total neurons")
        print(f"  Spike data reduced to shape: {self.spk.shape}")


    def estimate_adjacency_matrices(self):
        """Estimate adjacency matrices for LFP and spike data"""
        def compute_adj_matrix(data, thres):
            n_nodes = data.shape[1]
            adj_mat = np.zeros((n_nodes, n_nodes))
            
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    corr_total = 0
                    for k in range(data.shape[0]):
                        i_channel = data[k, i, :]
                        j_channel = data[k, j, :]
                        if (not np.all(i_channel==i_channel[0]) and not np.all(j_channel==j_channel[0])):
                            corr_total += np.corrcoef(i_channel, j_channel)[0, 1]
                    corr_i_j = corr_total / data.shape[0]
                    adj_mat[i, j] = adj_mat[j, i] = corr_i_j

            # Construct hard adjacency matrix
            indices = np.where(np.abs(adj_mat) > thres)
            adj_mat = np.zeros((n_nodes, n_nodes))
            adj_mat[indices] = 1

            return torch.tensor(adj_mat)

        def compute_adj_matrix_univariate(data, thres):
            n_nodes = data.shape[1]
            adj_mat = np.zeros((n_nodes, n_nodes))
            
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    i_channel = data[:, i, 0]
                    j_channel = data[:, j, 0]

                    if (not np.all(i_channel==i_channel[0]) and not np.all(j_channel==j_channel[0])):
                        corr_i_j = np.corrcoef(i_channel, j_channel)[0, 1]
                    else:
                        corr_i_j = 0
                    adj_mat[i, j] = adj_mat[j, i] = corr_i_j
        
            # Construct hard adjacency matrix
            indices = np.where(np.abs(adj_mat) > thres)
            adj_mat = np.zeros((n_nodes, n_nodes))
            adj_mat[indices] = 1

            return torch.tensor(adj_mat)


        self.lfp_adj = compute_adj_matrix(self.lfp, self.adj_thres[0])
        if self.reduce_dim:
            self.spk_adj = compute_adj_matrix_univariate(self.spk, self.adj_thres[1])
        else:
            self.spk_adj = compute_adj_matrix(self.spk, self.adj_thres[1])


    def get_dataloaders(self, train_batch_size=16, val_batch_size=None, test_batch_size=None, 
                        random_state=0, oversample_method='random', recycle=True, scale=2):
        """
        Create train, validation, and test dataloaders
        
        Parameters:
        train_batch_size (int): Batch size for training dataloader
        val_batch_size (int): Batch size for validation dataloader (defaults to n_trials if None)
        test_batch_size (int): Batch size for test dataloader (defaults to n_trials if None)
        random_state (int): Random seed for reproducibility
        oversample_method (str): Method for oversampling ('random' for random oversampling)
        recycle (bool): Whether to recycle data in dataloaders
        
        Returns:
        tuple: (train_loader, val_loader, test_loader)
        """
        np.random.seed(random_state)
        
        if self.task_type == 'binary':
            return self._get_binary_dataloaders(
                train_batch_size, val_batch_size, test_batch_size, 
                random_state, oversample_method, recycle
            )
        else:  # multiclass
            return self._get_multiclass_dataloaders(
                train_batch_size, val_batch_size, test_batch_size, 
                random_state
            )
    

    def _oversample_train_data_pairwise(self, X_lfp_train, X_spk_train, y_train, random_state, method='random'):
        X_combined = np.concatenate((X_lfp_train.reshape(X_lfp_train.shape[0], -1), 
                                    X_spk_train.reshape(X_spk_train.shape[0], -1)), axis=1)
        
        if method == 'random':
            oversampler = RandomOverSampler(random_state=random_state)
        elif method == 'smote':
            oversampler = SMOTE(random_state=random_state)
        else:
            raise ValueError("Unsupported oversampling method. Choose 'random' or 'smote'.")
        
        X_resampled, y_resampled = oversampler.fit_resample(X_combined, y_train)
        
        lfp_size = X_lfp_train.shape[1] * X_lfp_train.shape[2]
        X_lfp_resampled = X_resampled[:, :lfp_size].reshape(-1, X_lfp_train.shape[1], X_lfp_train.shape[2])
        X_spk_resampled = X_resampled[:, lfp_size:].reshape(-1, X_spk_train.shape[1], X_spk_train.shape[2])
        
        return X_lfp_resampled, X_spk_resampled, y_resampled
    
    
    def _get_multiclass_dataloaders(self, train_batch_size=64, val_batch_size=None, test_batch_size=None, 
                                    random_state=0):
        np.random.seed(random_state)
    
        # First split into train and temp (val+test)
        X = np.arange(len(self.labels)).reshape(-1, 1)  # dummy X for stratification
        y = self.labels
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size + self.val_size, random_state=random_state)
        train_idx, temp_idx = next(sss.split(X, y))
        
        # Further split temp_idx into validation and test
        test_val_ratio = self.test_size / (self.test_size + self.val_size)
        sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=random_state)
        val_idx, test_idx = next(sss_val_test.split(X[temp_idx], y[temp_idx]))
        
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]
        
        # Prepare the datasets
        X_lfp_test, X_spk_test = self.lfp[test_idx], self.spk[test_idx]
        y_test = self.labels[test_idx]

        X_lfp_val, X_spk_val = self.lfp[val_idx], self.spk[val_idx]
        y_val = self.labels[val_idx]

        X_lfp_train, X_spk_train = self.lfp[train_idx], self.spk[train_idx]
        y_train = self.labels[train_idx]
        
        self.train_num = len(y_train)
        self.val_num = len(y_val)
        self.test_num = len(y_test)
        
        # Print class distribution
        print("\nClass distribution:")
        print("  Train:", dict(OrderedDict(sorted(Counter(y_train).items()))))
        print("  Validation:", dict(OrderedDict(sorted(Counter(y_val).items()))))
        print("  Test:", dict(OrderedDict(sorted(Counter(y_test).items()))))
                
        # Prepare the data for the CustomGraphDataset
        train_data = np.column_stack([X_lfp_train.reshape(X_lfp_train.shape[0], -1), 
                                    X_spk_train.reshape(X_spk_train.shape[0], -1), 
                                    y_train])
        val_data = np.column_stack([X_lfp_val.reshape(X_lfp_val.shape[0], -1), 
                                    X_spk_val.reshape(X_spk_val.shape[0], -1), 
                                    y_val])
        test_data = np.column_stack([X_lfp_test.reshape(X_lfp_test.shape[0], -1), 
                                    X_spk_test.reshape(X_spk_test.shape[0], -1), 
                                    y_test])

        if val_batch_size is None:
            val_batch_size = self.n_trials
        if test_batch_size is None:
            test_batch_size = self.n_trials

        if self.is_graph:
            train_dataset = CustomGraphDataset(train_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
            val_dataset = CustomGraphDataset(val_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
            test_dataset = CustomGraphDataset(test_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)

            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate)
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=custom_collate)
        
        else:
            train_dataset = CustomDataset(train_data, self.dim_modalities, self.classification)
            val_dataset = CustomDataset(val_data, self.dim_modalities, self.classification)
            test_dataset = CustomDataset(test_data, self.dim_modalities, self.classification)

            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    def _get_binary_dataloaders(self, train_batch_size=64, val_batch_size=None, test_batch_size=None, 
                                random_state=0, oversample_method='random', recycle=True, scale=2):
        np.random.seed(random_state)

        # 1. Take out a fixed number of majority label to make the dataset roughly balanced
        minority_indices = np.where(self.labels == 0)[0]
        majority_indices = np.where(self.labels == 1)[0]
        n_minority = len(minority_indices)
        n_majority_to_keep = int(scale * n_minority)
        
        majority_indices_to_keep = np.random.choice(majority_indices, size=n_majority_to_keep, replace=False)
        majority_indices_holdout = np.setdiff1d(majority_indices, majority_indices_to_keep)
        
        balanced_indices = np.concatenate([minority_indices, majority_indices_to_keep])
        
        # 2. Stratified split of the roughly balanced dataset
        X = np.arange(len(balanced_indices)).reshape(-1, 1)  # dummy X for stratification
        y = self.labels[balanced_indices]
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size + self.val_size, random_state=random_state)
        train_idx, temp_idx = next(sss.split(X, y))
        
        # Further split temp_idx into validation and test
        test_val_ratio = self.test_size / (self.test_size + self.val_size)
        sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=random_state)
        val_idx, test_idx = next(sss_val_test.split(X[temp_idx], y[temp_idx]))
        
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]

        # Map back to original indices
        train_indices = balanced_indices[train_idx]
        val_indices = balanced_indices[val_idx]
        test_indices = balanced_indices[test_idx]

        # Prepare the datasets
        X_lfp_test, X_spk_test = self.lfp[test_indices], self.spk[test_indices]
        y_test = self.labels[test_indices]

        X_lfp_val, X_spk_val = self.lfp[val_indices], self.spk[val_indices]
        y_val = self.labels[val_indices]

        X_lfp_train, X_spk_train = self.lfp[train_indices], self.spk[train_indices]
        y_train = self.labels[train_indices]

        # 3. Add the holdout samples from the majority class to the training data
        if recycle:
            X_lfp_train = np.concatenate([X_lfp_train, self.lfp[majority_indices_holdout]])
            X_spk_train = np.concatenate([X_spk_train, self.spk[majority_indices_holdout]])
            y_train = np.concatenate([y_train, self.labels[majority_indices_holdout]])

        # Oversample the training data to ensure balance
        if self.oversample:
            X_lfp_train, X_spk_train, y_train = self._oversample_train_data_pairwise(X_lfp_train, X_spk_train, y_train, random_state, method=oversample_method)

        self.train_num = len(y_train)
        self.val_num = len(y_val)
        self.test_num = len(y_test)

        # Prepare the data for the CustomGraphDataset
        train_data = np.column_stack([X_lfp_train.reshape(X_lfp_train.shape[0], -1), 
                                    X_spk_train.reshape(X_spk_train.shape[0], -1), 
                                    y_train])
        val_data = np.column_stack([X_lfp_val.reshape(X_lfp_val.shape[0], -1), 
                                    X_spk_val.reshape(X_spk_val.shape[0], -1), 
                                    y_val])
        test_data = np.column_stack([X_lfp_test.reshape(X_lfp_test.shape[0], -1), 
                                    X_spk_test.reshape(X_spk_test.shape[0], -1), 
                                    y_test])
        
        if val_batch_size is None:
            val_batch_size = self.n_trials
        if test_batch_size is None:
            test_batch_size = self.n_trials

        if self.is_graph:
            train_dataset = CustomGraphDataset(train_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
            val_dataset = CustomGraphDataset(val_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
            test_dataset = CustomGraphDataset(test_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)

            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate)
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=custom_collate)
        else:
            train_dataset = CustomDataset(train_data, self.dim_modalities, self.classification)
            val_dataset = CustomDataset(val_data, self.dim_modalities, self.classification)
            test_dataset = CustomDataset(test_data, self.dim_modalities, self.classification)

            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
