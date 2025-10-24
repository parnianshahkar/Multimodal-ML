import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import random
from tqdm import tqdm
import pdb
import itertools



# Function to set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    



class CustomDataset(Dataset):
    def __init__(self, data, dim_modalities, classification=False):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.p1 = dim_modalities[0]
        self.p2 = dim_modalities[1]
        self.classification = classification
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X1 = self.data[idx, :self.p1] 
        X2 = self.data[idx, self.p1:(self.p1+self.p2)]  
        Y = self.data[idx, (self.p1+self.p2):] 
        
        # Make sure the label format is correct for classification task
        if self.classification:
            Y = Y.long().squeeze()

        return X1, X2, Y


# class CustomDataset(Dataset):
#     def __init__(self, data, dim_modalities, classification=False):
#         self.data = torch.tensor(data, dtype=torch.float32)
#         self.dim_modalities = dim_modalities
#         self.classification = classification

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Initialize starting index
#         start_idx = 0
#         modalities = []
        
#         # Iterate over each modality dimension
#         for dim in self.dim_modalities:
#             end_idx = start_idx + dim
#             modalities.append(self.data[idx, start_idx:end_idx])
#             start_idx = end_idx
        
#         # Remaining data is considered as labels
#         Y = self.data[idx, start_idx:]
        
#         # Ensure the label format is correct for classification task
#         if self.classification:
#             Y = Y.long().squeeze()

#         return tuple(modalities) + (Y,)


class AverageMeter():
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def calculate_accuracy(outputs, labels):
    """
    Calculates the accuracy of model predictions.

    Args:
        outputs (torch.Tensor): The model outputs (logits or probabilities).
        labels (torch.Tensor): The ground truth labels.

    Returns:
        float: The accuracy as a percentage of correct predictions.
    """
    _, predicted = torch.max(outputs, dim=1)
    
    # Compare predicted labels with true labels and compute the number of correct predictions
    correct = (predicted == labels).sum().item()

    # Calculate accuracy as the percentage of correct predictions
    total = labels.size(0)
    accuracy = correct / total

    return accuracy



def get_weights_by_task_loss(losses, scale=1):
    losses = torch.tensor(losses)

    # Inverse of the losses (lower loss -> higher weight)
    inverse_losses = (1.0 / losses ** scale) 

    # Normalize the weights to sum to 1
    weight_sum = torch.sum(inverse_losses)
    weights = inverse_losses / weight_sum  # Standardized weights

    return weights  # Shape: (len(losses),)



def get_weights_by_clustering(losses, max_k=10, optimal_k=None,
                              method='silhouette',
                              verbose=False, random_state=0):
    """
    Determines model weights based on clustering of task losses.
    
    Parameters:
    - losses: arraylike representing the task loss for each model.
    - max_k: Maximum number of clusters to test for the elbow method.
    - optimal_k: optimal number of clusters. If None, automatically choose by method
    - method: optimal cluster selection method, choose between 'silhouette' and 'elbow'
    - random_state: Random state for reproducibility in KMeans.

    Returns:
    - weights: numpy array of shape (n_models,) with normalized weights for each model.
    """
    # Step 1: Determine the optimal number of clusters
    losses = np.array(losses)
    if optimal_k is None: 
        max_k = np.min([max_k, len(losses)])
        k_list = np.arange(1,max_k + 1)

        # Automatically determine the elbow point where inertia starts to decrease slowly        
        if method == 'silhouette':
            optimal_k = compute_optimal_k_with_silhouette(losses.reshape(-1,1), k_list, random_state=random_state)
        elif method == 'elbow':
            optimal_k = compute_elbow_point(losses.reshape(-1,1), k_list, len(losses),random_state=random_state)

    # Step 2: Perform K-means clustering with the optimal number of clusters
    optimal_k = int(optimal_k)
    kmeans = KMeans(n_clusters=optimal_k, verbose=verbose, random_state=random_state)
    clusters = kmeans.fit_predict(losses.reshape(-1, 1))

    # Step 3: Identify the cluster with the smallest average task loss
    cluster_losses = [np.mean(losses[clusters == i]) for i in range(optimal_k)]
    best_cluster = np.argmin(cluster_losses)

    # Step 4: Assign weights based on cluster membership
    weights = np.zeros_like(losses, dtype=float)
    weights[clusters == best_cluster] = 1.0

    weights /= np.sum(weights)
    best_cluster_idxs = np.where(weights != 0)[0]
    return weights, optimal_k, best_cluster_idxs


# Computes the elbow point using cosine of intersection angles
# Reference: https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-021-01910-w#availability-of-data-and-materials
def calculate_euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(a, b, c):
    # Use the law of cosines to find the angle
    return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

def compute_elbow_point(data, k_list, n_points, random_state=0):
    inertia = []
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    normalized_inertia = np.array(inertia)/n_points
    inertia_range = np.max(normalized_inertia - np.min(normalized_inertia))
    normalized_inertia = (normalized_inertia - np.min(normalized_inertia)) * 10 /inertia_range

    PL = [(normalized_inertia[i], k_list[i]) for i in range(len(k_list))]

    alpha_min = np.pi
    optimal_k = None

    # Iterate over every three adjacent points
    for i in range(1, len(PL) - 1):
        Pi, Pj, Pk = PL[i - 1], PL[i], PL[i + 1]

        # Calculate distances a, b, c
        a = calculate_euclidean_distance(Pj, Pk)
        b = calculate_euclidean_distance(Pi, Pj)
        c = calculate_euclidean_distance(Pi, Pk)

        # Calculate the angle using the arccos formula
        angle = calculate_angle(a, b, c)

        # Find the minimal angle and corresponding index
        if angle < alpha_min:
            alpha_min = angle
            optimal_k = Pj[1]  # The cluster number corresponding to Pj

    return optimal_k



def compute_optimal_k_with_silhouette(data, k_list, random_state=0):
    best_k = None
    best_score = -1  # Silhouette scores range from -1 to 1

    # Iterate over each k in the list
    for k in k_list:
        # Silhouette scores are not defined for k=1 or k=len(data)
        if k == 1 or k == len(data):
            continue
        
        # Fit the KMeans model
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)

        # Compute the silhouette score
        score = silhouette_score(data, labels)

        # Update best_k if the current score is better
        if score > best_score:
            best_score = score
            best_k = k

    return best_k



def load_all_data(dataloader):
    # Load all data from the dataloader and convert then to numpy array
    dataset = dataloader.dataset
    X1 = dataset.data[:, :dataset.p1].numpy()  # First modality
    X2 = dataset.data[:, dataset.p1:(dataset.p1 + dataset.p2)].numpy()  # Second modality
    Y = dataset.data[:, (dataset.p1 + dataset.p2):].numpy()  # Target variable

    return X1, X2, Y


def load_all_data_general(dataloader):
    """
    Load all data from the dataloader and convert them to numpy arrays for multiple modalities.
    
    Args:
        dataloader: A DataLoader object containing the dataset.
        
    Returns:
        A tuple containing lists of numpy arrays for each modality and a numpy array for the target variable.
    """
    dataset = dataloader.dataset
    data = dataset.data.numpy()  # Convert entire dataset to a numpy array
    dim_modalities = dataset.dim_modalities  # Assume this is a list of indices defining modality partitions
    
    # Extract modalities based on partition indices
    modalities = []
    start_idx = 0

    for dim in dim_modalities:
        end_idx = start_idx + dim
        modalities.append(data[:, start_idx:end_idx])
        start_idx = end_idx
    
    # The last part of the data is assumed to be the target variable
    Y = data[:, start_idx:]
    
    return modalities, Y