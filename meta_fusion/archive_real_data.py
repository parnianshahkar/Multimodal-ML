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
from meta_fusion.methods import *
from meta_fusion.models import *
from meta_fusion.utils import CustomDataset
from meta_fusion.third_party import *

sys.path.append('../data/NACC/')
from data_cleaning_util import *

class Prepare_realdata:
    def __init__(self, data_name, test_size = 0.2, val_size = 0.2, version=None):
        self.test_size = test_size
        self.val_size = val_size
        self.data_name = data_name

        # Load the dataset
        if data_name == "NACC":
            if version is None:
                data_dir = '../data/NACC/preprocessed_v2/'
            elif version == 1:
                data_dir = '../data/NACC/preprocessed_v1/'
            uds_data = pd.read_csv(data_dir+"uds_matched.csv")
            sbm_data = pd.read_csv(data_dir+"mrisbm_matched.csv")
            self.classification = True
    
            # Get the covariates and label
            cat_feats, ord_feats, num_feats, label_feat, cdr_feats = get_feature_types(uds_data)
            #all_feats = cat_feats + ord_feats + num_feats + cdr_feats
            all_feats = cat_feats + ord_feats + num_feats # remove cdf features since these are another set of 'labels'
            
            self.X1 = uds_data[all_feats]
            self.X2 = sbm_data
            self.Y = uds_data[label_feat].astype(int) - 1   # Convert labels from 1-4 to 0-3


    def get_data_info(self):
        return [self.d1, self.d2, self.n, self.train_num, self.val_num, self.test_num]
        

    def get_data_loaders(self, train_batch_size = 64, val_batch_size = None, test_batch_size = None, random_state=0):
        '''
        Perform data splitting and prepare data loaders for model training
        '''
        X1_tmp, X1_test, X2_tmp, X2_test, Y_tmp, Y_test = train_test_split(self.X1, self.X2, self.Y, 
                                                                            test_size=self.test_size, 
                                                                            random_state=random_state)

        if self.data_name == "NACC":
            # impute the missing uds data
            X1_tmp, X1_test = impute_invariant_features(X1_tmp, X1_test)
            preprocessor = imputer_by_feature_type(X1_tmp)
            preprocessor = preprocessor.fit(X1_tmp)
            X1_tmp = pd.DataFrame(preprocessor.transform(X1_tmp), columns=preprocessor.get_feature_names_out(), index=X1_tmp.index.values)
            X1_test = pd.DataFrame(preprocessor.transform(X1_test), columns=preprocessor.get_feature_names_out(), index=X1_test.index.values)
            
            # impute the missing sbm data
            X2_tmp, X2_test = impute_mrisbm_features(X2_tmp, X2_test)
            
        X1_train, X1_val, X2_train, X2_val, Y_train, Y_val = train_test_split(X1_tmp, X2_tmp, Y_tmp, 
                                                                                test_size=self.val_size, 
                                                                                random_state=random_state)

        # Store data information
        self.d1 = X1_train.shape[1]
        self.d2 = X2_train.shape[1]
        self.n = len(self.Y)
        self.train_num = X1_train.shape[0]
        self.test_num = X1_test.shape[0]
        self.val_num = X1_val.shape[0]
        self.dim_modalities = [self.d1, self.d2]

        # Prepare data loaders
        train_data = np.column_stack((X1_train, X2_train, Y_train))
        val_data = np.column_stack((X1_val, X2_val, Y_val))
        test_data = np.column_stack((X1_test, X2_test, Y_test))
    
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



class PrepareNeuronData:
    def __init__(self, rat_name, data_dir="../data/Neuron2021/Shahbaba_2021_dataset/stattensor", 
                 test_size=0.2, val_size=0.2, oversample=True, time_range=(200, 250), 
                 adj_thres=[0.4, 0.001]):
        self.rat_name = rat_name
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.oversample = oversample
        self.time_range = time_range
        self.adj_thres = adj_thres
        
        self.load_data()
        self.preprocess_data()
        self.estimate_adjacency_matrices()

        
        self.n_trials, self.n_nodes_lfp, self.n_timepoints_lfp = self.lfp.shape
        self.n_nodes_spk, self.n_timepoints_spk = self.spk.shape[1], self.spk.shape[2]
        
        self.classification = True
        self.num_modalities = 2
        self.dim_modalities = [
            (self.n_nodes_lfp, self.n_timepoints_lfp),
            (self.n_nodes_spk, self.n_timepoints_spk)
        ]


    def load_data(self):
        lfp = np.load(f"{self.data_dir}/{self.rat_name}/{self.rat_name.lower()}_lfp_data_sampled.npy", allow_pickle=True).swapaxes(1, 2)
        spk = np.load(f"{self.data_dir}/{self.rat_name}/{self.rat_name.lower()}_spike_data_binned.npy", allow_pickle=True)
        bvr = np.load(f"{self.data_dir}/{self.rat_name}/{self.rat_name.lower()}_trial_info.npy", allow_pickle=True)
        
        self.lfp = lfp[:, :, self.time_range[0]:self.time_range[1]]
        self.spk = spk[:, :, self.time_range[0]:self.time_range[1]]
        self.bvr = bvr


    def preprocess_data(self):
        correct_index = np.where(self.bvr[:, 0] == 1)[0]
        self.lfp = self.lfp[correct_index, :, :]
        self.spk = self.spk[correct_index, :, :]
        self.bvr = self.bvr[correct_index, :]
        self.labels = self.bvr[:, 1]

    
    def estimate_adjacency_matrices(self):
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

        self.lfp_adj = compute_adj_matrix(self.lfp, self.adj_thres[0])
        self.spk_adj = compute_adj_matrix(self.spk, self.adj_thres[1])


    def get_data_info(self):
        return {
            "rat_name": self.rat_name,
            "dim_modalities": self.dim_modalities,
            "n_trials": self.n_trials,
            "train_num": self.train_num if hasattr(self, 'train_num') else None,
            "val_num": self.val_num if hasattr(self, 'val_num') else None,
            "test_num": self.test_num if hasattr(self, 'test_num') else None,
            "n_nodes_lfp": self.n_nodes_lfp,
            "n_timepoints_lfp": self.n_timepoints_lfp,
            "n_nodes_spk": self.n_nodes_spk,
            "n_timepoints_spk": self.n_timepoints_spk,
            "n_classes": len(np.unique(self.labels)),
            "class_distribution": dict(Counter(self.labels)),
            "oversample": self.oversample,
            "time_range": self.time_range,
            "adjacency_threshold": self.adj_thres
        }


    def get_adjacency_matrices(self):
        return self.lfp_adj, self.spk_adj
        

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


    def get_data_loaders_old(self, train_batch_size=64, val_batch_size=None, test_batch_size=None, random_state=0):
        np.random.seed(random_state)

        n_minority = sum(self.labels == 0)
        test_size = int(n_minority * self.test_size * 2)

        minority_indices = np.where(self.labels == 0)[0]
        majority_indices = np.where(self.labels == 1)[0]

        test_minority_indices = np.random.choice(minority_indices, size=int(test_size/2), replace=False)
        test_majority_indices = np.random.choice(majority_indices, size=int(test_size/2), replace=False)
        test_indices = np.concatenate([test_minority_indices, test_majority_indices])

        X_lfp_test, X_spk_test = self.lfp[test_indices], self.spk[test_indices]
        y_test = self.labels[test_indices]

        train_indices = np.array([i for i in range(len(self.labels)) if i not in test_indices])
        X_lfp_train, X_spk_train = self.lfp[train_indices], self.spk[train_indices]
        y_train = self.labels[train_indices]

        if self.oversample:
            X_lfp_train, X_spk_train, y_train = self._oversample_train_data_pairwise(X_lfp_train, X_spk_train, y_train, random_state)

        X_lfp_train, X_lfp_val, X_spk_train, X_spk_val, y_train, y_val = train_test_split(
            X_lfp_train, X_spk_train, y_train, test_size=self.val_size, stratify=y_train, random_state=random_state
        )

        self.train_num = len(y_train)
        self.val_num = len(y_val)
        self.test_num = len(y_test)

        train_data = np.column_stack([X_lfp_train.reshape(X_lfp_train.shape[0], -1), 
                                      X_spk_train.reshape(X_spk_train.shape[0], -1), 
                                      y_train])
        val_data = np.column_stack([X_lfp_val.reshape(X_lfp_val.shape[0], -1), 
                                    X_spk_val.reshape(X_spk_val.shape[0], -1), 
                                    y_val])
        test_data = np.column_stack([X_lfp_test.reshape(X_lfp_test.shape[0], -1), 
                                     X_spk_test.reshape(X_spk_test.shape[0], -1), 
                                     y_test])

        train_dataset = CustomGraphDataset(train_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
        val_dataset = CustomGraphDataset(val_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
        test_dataset = CustomGraphDataset(test_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)

        if val_batch_size is None:
            val_batch_size = self.n_trials
        if test_batch_size is None:
            test_batch_size = self.n_trials

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=custom_collate)

        return train_loader, val_loader, test_loader
    

    def get_data_loaders(self, train_batch_size=64, val_batch_size=None, test_batch_size=None, 
                               random_state=0, oversample_method='random', recycle=True):
        np.random.seed(random_state)

        # 1. Take out a fixed number of majority label to make the dataset roughly balanced
        minority_indices = np.where(self.labels == 0)[0]
        majority_indices = np.where(self.labels == 1)[0]
        n_minority = len(minority_indices)
        n_majority_to_keep = n_minority
        
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

        # Create datasets
        train_dataset = CustomGraphDataset(train_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
        val_dataset = CustomGraphDataset(val_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
        test_dataset = CustomGraphDataset(test_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)

        # Set batch sizes
        if val_batch_size is None:
            val_batch_size = self.n_trials
        if test_batch_size is None:
            test_batch_size = self.n_trials

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=custom_collate)

        return train_loader, val_loader, test_loader


    # def get_data_loaders(self, train_batch_size=64, val_batch_size=None, test_batch_size=None, random_state=0):
    #     np.random.seed(random_state)

    #     # Combine LFP and SPK data
    #     X = np.column_stack([self.lfp.reshape(self.lfp.shape[0], -1), 
    #                         self.spk.reshape(self.spk.shape[0], -1)])
    #     y = self.labels

    #     # 1. Stratified split for test, val, and train
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)
    #     train_val_idx, test_idx = next(sss.split(X, y))

    #     X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    #     X_test, y_test = X[test_idx], y[test_idx]

    #     # Split train_val into train and validation
    #     val_size_adjusted = self.val_size / (1 - self.test_size)
    #     sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    #     train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))

    #     X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    #     X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    #     # 2. Oversample the training data
    #     ros = RandomOverSampler(random_state=random_state)
    #     X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    #     # Split back into LFP and SPK
    #     lfp_dim = self.lfp.shape[1] * self.lfp.shape[2]
    #     X_lfp_train, X_spk_train = X_train_resampled[:, :lfp_dim], X_train_resampled[:, lfp_dim:]
    #     X_lfp_val, X_spk_val = X_val[:, :lfp_dim], X_val[:, lfp_dim:]
    #     X_lfp_test, X_spk_test = X_test[:, :lfp_dim], X_test[:, lfp_dim:]

    #     # Reshape back to original dimensions
    #     X_lfp_train = X_lfp_train.reshape(-1, self.lfp.shape[1], self.lfp.shape[2])
    #     X_spk_train = X_spk_train.reshape(-1, self.spk.shape[1], self.spk.shape[2])
    #     X_lfp_val = X_lfp_val.reshape(-1, self.lfp.shape[1], self.lfp.shape[2])
    #     X_spk_val = X_spk_val.reshape(-1, self.spk.shape[1], self.spk.shape[2])
    #     X_lfp_test = X_lfp_test.reshape(-1, self.lfp.shape[1], self.lfp.shape[2])
    #     X_spk_test = X_spk_test.reshape(-1, self.spk.shape[1], self.spk.shape[2])

    #     self.train_num = len(y_train_resampled)
    #     self.val_num = len(y_val)
    #     self.test_num = len(y_test)

    #     # Prepare the data for the CustomGraphDataset
    #     train_data = np.column_stack([X_lfp_train.reshape(X_lfp_train.shape[0], -1), 
    #                                 X_spk_train.reshape(X_spk_train.shape[0], -1), 
    #                                 y_train_resampled])
    #     val_data = np.column_stack([X_lfp_val.reshape(X_lfp_val.shape[0], -1), 
    #                                 X_spk_val.reshape(X_spk_val.shape[0], -1), 
    #                                 y_val])
    #     test_data = np.column_stack([X_lfp_test.reshape(X_lfp_test.shape[0], -1), 
    #                                 X_spk_test.reshape(X_spk_test.shape[0], -1), 
    #                                 y_test])

    #     # Create datasets
    #     train_dataset = CustomGraphDataset(train_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
    #     val_dataset = CustomGraphDataset(val_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)
    #     test_dataset = CustomGraphDataset(test_data, self.dim_modalities, self.classification, self.lfp_adj, self.spk_adj)

    #     # Set batch sizes
    #     if val_batch_size is None:
    #         val_batch_size = self.n_trials
    #     if test_batch_size is None:
    #         test_batch_size = self.n_trials

    #     # Create data loaders
    #     train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate)
    #     val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate)
    #     test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=custom_collate)

    #     return train_loader, val_loader, test_loader