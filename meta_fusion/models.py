import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool, global_max_pool, TopKPooling

import random
from tqdm import tqdm
import pdb
import copy
import itertools

from meta_fusion.utils import load_all_data, AverageMeter


# Simple multi-layer neural network (Multi-Layer Perceptron)
class MLP_Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        # Create a list of layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # Apply ReLU to all but the last layer
                layers.append(nn.ReLU())
        
        # Unpack layers into the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
            x = x.view(x.size(0), -1)  # Flatten the input while keeping the batch size dimension
        return self.model(x)



# Simple single-layer neural network (Single-Layer Perceptron)
class SLP_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Define a single fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
            x = x.view(x.size(0), -1)  # Flatten the input while keeping the batch size dimension
        return self.fc(x)



class Fuse_Net(nn.Module):
    def __init__(self, feature_extractors, combined_model, 
                 is_static_list=None, freeze_extractors_list=None):
        """
        Args:
            feature_extractors: List of feature extractors for each modality
            combined_model: The final model that takes combined features as input
            is_static_list: List of booleans indicating if each feature extractor is static (e.g., PCA)
            freeze_extractors_list: List of booleans indicating if each feature extractor should be frozen
        """
        super(Fuse_Net, self).__init__()
        
        # The following is NOT properly registered
        # self.feature_extractors = feature_extractors

        self.feature_extractors = feature_extractors
        self.registered_extractors = nn.ModuleList([
            copy.deepcopy(fe) if isinstance(fe, nn.Module) else None
            for fe in feature_extractors
        ])
        self.combined_model = combined_model
        self.is_static_list = is_static_list or [False] * len(feature_extractors)
        self.freeze_extractors_list = freeze_extractors_list or [False] * len(feature_extractors)
        
        # Freeze learnable feature extractors if needed
        for extractor, is_static, freeze in zip(feature_extractors, self.is_static_list, self.freeze_extractors_list):
            if not is_static and freeze:
                self._freeze_feature_extractor(extractor)


    def _freeze_feature_extractor(self, feature_extractor):
        """Freeze the parameters of the feature extractor to prevent gradients from updating."""
        if feature_extractor is None or feature_extractor == "full":
            return 
        for param in feature_extractor.parameters():
            param.requires_grad = False


    def forward(self, modalities):
        features = []
        for mod, extractor, registered_extractor, is_static in zip(modalities, self.feature_extractors, self.registered_extractors, self.is_static_list):
            if extractor is None:
                features.append(None)
            elif extractor == "full":
                features.append(mod)
            else:
                if is_static:
                    features.append(torch.from_numpy(extractor.transform(mod.cpu().numpy())).float().to(mod.device))
                else:
                    features.append(registered_extractor(mod))
        
        # Concatenate all non-None features
        combined_input = torch.cat([f for f in features if f is not None], dim=1)
        
        # Pass the combined input to the final model
        return self.combined_model(combined_input)



class Cohorts:
    def __init__(self, extractors, combined_hidden_layers, output_dim,
                 is_mod_static = None, freeze_mod_extractors = None):
        self.extractor_models = extractors.extractors  # The actual extraction models
        self.extractors = extractors  # The extractor class object
        self.combined_hidden_layers = combined_hidden_layers
        self.output_dim = output_dim
        self.is_mod_static = is_mod_static or [False] * len(self.extractor_models)
        self.freeze_mod_extractors = freeze_mod_extractors or [False] * len(self.extractor_models)

    def get_cohort_models(self):
        self.cohort_models = []
        self.cohort_dims = []
        self.model_structures = []

        # Create combinations of extractors for each modality
        extractor_combinations = itertools.product(*self.extractor_models)
        dim_combinations = itertools.product(*self.extractors.mod_outs)

        counter = 0
        for extractor_combination, dim_combination in zip(extractor_combinations, dim_combinations):
            if all(dim == 0 for dim in dim_combination):
                continue

            combined_input_dim = sum(dim_combination)
            combined_model = MLP_Net(
                input_dim=combined_input_dim,
                hidden_dims=self.combined_hidden_layers,
                output_dim=self.output_dim
            )

            model = Fuse_Net(
                feature_extractors=extractor_combination,
                combined_model=combined_model,
                is_static_list=self.is_mod_static,
                freeze_extractors_list=self.freeze_mod_extractors
            )

            self.cohort_models.append(model)
            self.cohort_dims.append(dim_combination)

            # Store the structure for debugging or verification purposes
            model_structure = {
                'model_num': counter,
                'extractors': [ext.__class__.__name__ for ext in extractor_combination],
                'combined_hidden_layers': self.combined_hidden_layers
            }
            self.model_structures.append(model_structure)
            counter += 1

        return self.cohort_models

    def get_cohort_info(self):
        if not hasattr(self, 'cohort_models'):
            print("No model information yet, get models first.")

        return len(self.cohort_models), self.cohort_dims



