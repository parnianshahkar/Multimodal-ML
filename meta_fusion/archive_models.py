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

import random
from tqdm import tqdm
import pdb
import copy
import itertools

from meta_fusion.archive_utils import load_all_data, AverageMeter


# Simple multi-layer neural network (Multi-Layer Perceptron)
class MLP_Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP_Net, self).__init__()
        
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
        super(SLP_Net, self).__init__()
        
        # Define a single fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:  # This means the input is likely an image (e.g., [batch_size, channels, height, width])
            x = x.view(x.size(0), -1)  # Flatten the input while keeping the batch size dimension
        return self.fc(x)



class Fuse_Net(nn.Module):
    def __init__(self, feature_extractor_mod1, feature_extractor_mod2, combined_model, 
                       is_static_mod1=False, is_static_mod2=False,
                       freeze_extractor_mod1=False, freeze_extractor_mod2=False):
        """
        Args:
            feature_extractor_mod1: Feature extractor for modality 1 (e.g., NN, PCA)
            feature_extractor_mod2: Feature extractor for modality 2 (e.g., NN, PCA)
            combined_model: The final model that takes combined features as input
            is_static_mod1: If True, treat feature_extractor_mod1 as static (e.g., PCA)
            is_static_mod2: If True, treat feature_extractor_mod2 as static (e.g., PCA)
        """
        super(Fuse_Net, self).__init__()
        self.feature_extractor_mod1 = feature_extractor_mod1
        self.feature_extractor_mod2 = feature_extractor_mod2
        self.combined_model = combined_model
        self.is_static_mod1 = is_static_mod1
        self.is_static_mod2 = is_static_mod2
        self.freeze_extractor_mod1 = freeze_extractor_mod1
        self.freeze_extractor_mod2 = freeze_extractor_mod2
        
        # Freeze learnable feature extractors if needed
        if not self.is_static_mod1 and self.freeze_extractor_mod1:
            if self.feature_extractor_mod1 is not None and self.feature_extractor_mod1 != "full":
                self._freeze_feature_extractor(self.feature_extractor_mod1)
        
        if not self.is_static_mod2 and self.freeze_extractor_mod2:
            if self.feature_extractor_mod2 is not None and self.feature_extractor_mod2 != "full":
                self._freeze_feature_extractor(self.feature_extractor_mod2)


    def _freeze_feature_extractor(self, feature_extractor):
        """Freeze the parameters of the feature extractor to prevent gradients from updating."""
        for param in feature_extractor.parameters():
            param.requires_grad = False


    def forward(self, mod1, mod2):
        # Handle modality 1
        if self.feature_extractor_mod1 is None:
            mod1_features = None
        elif self.feature_extractor_mod1 == "full":
            mod1_features = mod1
        else:
            if self.is_static_mod1:
                # If static (e.g., PCA), convert to NumPy, transform, and convert back to PyTorch tensor
                mod1_features = torch.from_numpy(self.feature_extractor_mod1.transform(mod1.cpu().numpy())).float().to(mod1.device)
            else:
                # If gradient-based (e.g., NN), pass through the feature extractor
                mod1_features = self.feature_extractor_mod1(mod1)
        
        # Handle modality 2
        if self.feature_extractor_mod2 is None:
            mod2_features = None
        elif self.feature_extractor_mod2 == "full":
            mod2_features = mod2
        else:
            if self.is_static_mod2:
                # If static (e.g., PCA), convert to NumPy, transform, and convert back to PyTorch tensor
                mod2_features = torch.from_numpy(self.feature_extractor_mod2.transform(mod2.cpu().numpy())).float().to(mod2.device)
            else:
                # If gradient-based (e.g., NN), pass through the feature extractor
                mod2_features = self.feature_extractor_mod2(mod2)

        # Concatenate the features if both are present
        if mod1_features is not None and mod2_features is not None:
            combined_input = torch.cat((mod1_features, mod2_features), dim=1)
        elif mod1_features is not None:
            combined_input = mod1_features
        elif mod2_features is not None:
            combined_input = mod2_features
        else:
            raise ValueError("Both feature extractors cannot be None.")
        
        # Pass the combined input to the final model
        return self.combined_model(combined_input)



class Cohorts:
    """
    This class defines the models the in the cohort.
    """
    # [Note] This class now only implements tubular information, i.e. d1, d2 are dimensions
    # of vector-like modalities. Generalize the class in the future to accomondate images or
    # modalities of other types! 
    def __init__(self, extractors, combined_hidden_layers, output_dim,
                 is_static_mod1=False, is_static_mod2=False,
                 freeze_extractor_mod1=False, freeze_extractor_mod2=False):
        self.mod1_extractors = extractors.mod1_extractors
        self.mod2_extractors = extractors.mod2_extractors
        self.mod1_outs = extractors.mod1_outs  # output dimensions of the feature extractors 
        self.mod2_outs = extractors.mod2_outs  
        self.combined_hidden_layers = combined_hidden_layers   # hidden layers of the combined NN
        self.output_dim = output_dim
        self.d1 = extractors.d1
        self.d2 = extractors.d2

        self.is_static_mod1 = is_static_mod1
        self.is_static_mod2 = is_static_mod2  
        self.freeze_extractor_mod1 = freeze_extractor_mod1
        self.freeze_extractor_mod2 = freeze_extractor_mod2

    
    def get_cohort_models(self):
        self.cohort_pairs = []
        self.model_num = 0
        self.cohort_models = []
        self.model_structures = []  # Store model structure info for verification
        
        # Loop over all combinations of feature extractors from mod1 and mod2
        for i, mod1_extractor in enumerate(self.mod1_extractors):
            for j, mod2_extractor in enumerate(self.mod2_extractors):
                
                if self.mod1_outs[i] == self.mod2_outs[j] == 0:
                    continue
                
                model = Fuse_Net(
                    feature_extractor_mod1=copy.deepcopy(mod1_extractor),
                    feature_extractor_mod2=copy.deepcopy(mod2_extractor),
                    combined_model=MLP_Net(
                        input_dim=(self.mod1_outs[i] + self.mod2_outs[j]), 
                        hidden_dims=self.combined_hidden_layers, 
                        output_dim=self.output_dim
                    ),
                    is_static_mod1=self.is_static_mod1,
                    is_static_mod2=self.is_static_mod2,
                    freeze_extractor_mod1=self.freeze_extractor_mod1,
                    freeze_extractor_mod2=self.freeze_extractor_mod2
                )
                
                # Store model for later use
                self.cohort_models.append(model)
                self.model_num += 1
                self.cohort_pairs.append((self.mod1_outs[i], self.mod2_outs[j]))
                
                # Store the structure for debugging or verification purposes
                model_structure = {
                    'model_num': len(self.cohort_models),
                    'mod1_extractor': mod1_extractor.__class__.__name__,
                    'mod2_extractor': mod2_extractor.__class__.__name__,
                    'combined_hidden_layers': self.combined_hidden_layers
                }
                self.model_structures.append(model_structure)
        
        return self.cohort_models
    


    def get_cohort_info(self):
        if not hasattr(self, 'model_num') or not hasattr(self, 'cohort_pairs'):
            print("No model information yet, get models first.")

        return self.model_num, self.cohort_pairs





# Joint model to handle both sets of features
class Joint_Net(nn.Module):
    def __init__(self, feature_extractor_mod1, feature_extractor_mod2, combined_model):
        super(Joint_Net, self).__init__()
        self.feature_extractor_mod1 = feature_extractor_mod1
        self.feature_extractor_mod2 = feature_extractor_mod2
        self.combined_model = combined_model

    def forward(self, mod1, mod2):
        # If feature_extractor_mod1 is None, skip processing mod1
        if self.feature_extractor_mod1 is None:
            mod1_features = None
        elif self.feature_extractor_mod1 == "full":
            mod1_features = mod1
        else:
            mod1_features = self.feature_extractor_mod1(mod1)

        # If feature_extractor_mod2 is None, skip processing mod2
        if self.feature_extractor_mod2 is None:
            mod2_features = None
        elif self.feature_extractor_mod2 == "full":
            mod2_features = mod2
        else:
            mod2_features = self.feature_extractor_mod2(mod2)

        # If both feature sets are present, concatenate them
        if mod1_features is not None and mod2_features is not None:
            combined_input = torch.cat((mod1_features, mod2_features), dim=1)
        elif mod1_features is not None:
            combined_input = mod1_features  # Only mod1 is available
        elif mod2_features is not None:
            combined_input = mod2_features  # Only mod2 is available
        else:
            raise ValueError("Both feature extractors cannot be None.")

        # Pass the combined input to the final model
        return self.combined_model(combined_input)





class Cohorts_old:
    """
    This class defines the models the in the cohort.
    """
    # [Note] This class now only implements tubular information, i.e. d1, d2 are dimensions
    # of vector-like modalities. Generalize the class in the future to accomondate images or
    # modalities of other types! 
    def __init__(self, mod1_outs, mod2_outs, output_dim, d1, d2):
        self.mod1_outs = mod1_outs
        self.mod2_outs = mod2_outs
        self.output_dim = output_dim
        self.d1 = d1
        self.d2 = d2

        _, _ = self.get_cohort_info()


    def set_hidden_layers(self, mod1_hiddens, mod2_hiddens, combined_hiddens):
        """
        Sets the hidden layers of the feature extraction and combined layer manually
        """
        self.mod1_hiddens = mod1_hiddens
        self.mod2_hiddens = mod2_hiddens
        self.combined_hiddens = combined_hiddens

    
    def get_cohort_models(self):
        self.cohort_models = []
        self.model_structures = []  # Store model structure info for verification
        
        for i, (dim1, dim2) in enumerate(self.cohort_pairs):
            
            if dim1 == dim2 == 0:
                continue
            
            hidden1 = (self.d1 + dim1) // 2 if self.mod1_hiddens is None else self.mod1_hiddens[i]
            hidden2 = (self.d2 + dim2) // 2 if self.mod2_hiddens is None else self.mod2_hiddens[i]
            
            arg1 = 'full' if dim1 == self.d1 else MLP_Net(self.d1, [hidden1], dim1) if dim1 != 0 else None
            arg2 = 'full' if dim2 == self.d2 else MLP_Net(self.d2, [hidden2], dim2) if dim2 != 0 else None
            
            if self.combined_hiddens is None:
                # If either dim1 == d1 or dim2 == d2, add an additional hidden layer after concatenation
                if dim1 == self.d1 or dim2 == self.d2:
                    if dim1 != 0 and dim2 != 0:
                        combined_hidden_layers = [int((dim1 + dim2) / 2), int((dim1 + dim2) / 4)]  # Two hidden layers
                    elif dim1 != 0:
                        combined_hidden_layers = [int(dim1 / 2), int(dim1 / 4)]  # Two hidden layers
                    elif dim2 != 0:
                        combined_hidden_layers = [int(dim2 / 2), int(dim2 / 4)]  # Two hidden layers
                else:
                    if dim1 != 0 and dim2 != 0:
                        combined_hidden_layers = [int((dim1 + dim2) / 2)]  # One hidden layer
                    elif dim1 != 0:
                        combined_hidden_layers = [int(dim1 / 2)]  # One hidden layer
                    elif dim2 != 0:
                        combined_hidden_layers = [int(dim2 / 2)]  # One hidden layer
            else:
                combined_hidden_layers = self.combined_hiddens[i]

            # Create the Joint_Net model
            model = Joint_Net(arg1, arg2, MLP_Net(dim1 + dim2, combined_hidden_layers, self.output_dim))
            
            # Storing the model structure information
            model_structure = {
                'model_num': self.model_num + 1,
                'dim1': dim1,
                'dim2': dim2,
                'feature_extractor_1': 'full' if dim1 == self.d1 else f"MLP_Net({self.d1}, {hidden1}, {dim1})" if dim1 != 0 else 'None',
                'feature_extractor_2': 'full' if dim2 == self.d2 else f"MLP_Net({self.d2}, {hidden2}, {dim2})" if dim2 != 0 else 'None',
                'combined_net': combined_hidden_layers
            }
            self.model_structures.append(model_structure)
            self.cohort_models.append(model)
        
        return self.cohort_models
    


    def get_cohort_info(self):
        if not hasattr(self, 'model_num') or not hasattr(self, 'cohort_pairs'):
            
            self.cohort_pairs = []
            self.model_num = 0
            for dim1, dim2 in itertools.product(self.mod1_outs, self.mod2_outs):
                if dim1 == dim2 == 0:
                    continue
                self.model_num += 1
                self.cohort_pairs.append((dim1, dim2))

        return self.model_num, self.cohort_pairs



#####################
#    Graph Models   #
#####################

class GCN_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1, pooling='topK'):
        super(GCN_Net, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Pooling options
        self.pooling_type = pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'topK':
            self.pool = TopKPooling(hidden_dim)
        else:
            raise ValueError("Unsupported pooling type")

        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, edge_index, batch):
        # Initial embedding
        x = self.embedding(x)
        
        # GCN layers
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        
        # Pooling
        if self.pooling_type == 'topK':
            x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
            x = global_mean_pool(x, batch)
        else:
            x = self.pool(x, batch)
        
        # Final prediction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)



class GTN_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1, pooling='topK'):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling options
        self.pooling_type = pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'topK':
            self.pool = TopKPooling(hidden_dim)
        else:
            raise ValueError("Unsupported pooling type")
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch):
        # Initial embedding
        x = self.embedding(x)
        
        # Transformer layers
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        
        # Pooling
        if self.pooling_type == 'topK':
            x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
            x = global_mean_pool(x, batch)
        else:
            x = self.pool(x, batch)
        
        # Final prediction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)



# To be called by GraphExtractor to form submodels with different depths
class sliced_Graph_Net(nn.Module):
    def __init__(self, embedding, conv_layers, pool, pooling_type):
        super().__init__()
        self.embedding = embedding
        self.conv_layers = nn.ModuleList(conv_layers)
        self.pool = pool
        self.pooling_type = pooling_type


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        for conv in self.conv_layers:
            x = conv(x, edge_index)
        
        # Pooling
        if self.pooling_type == 'topK':
            x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
            x = global_mean_pool(x, batch)  # Additional global pooling after TopK
        elif self.pooling_type in ['mean', 'max']:
            x = self.pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        return x