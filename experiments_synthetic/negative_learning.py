import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import shutil

sys.path.append('../')
from meta_fusion.benchmarks import *
from meta_fusion.methods import *
from meta_fusion.models import *
from meta_fusion.utils import *
from meta_fusion.third_party import *
from meta_fusion.synthetic_data import PrepareSyntheticData
from meta_fusion.config import *


#########################
# Experiment parameters #
#########################
if True:
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 3:
        print("Error: incorrect number of parameters.")
        quit()

    large_cohort = int(sys.argv[1])
    seed = int(sys.argv[2])

# Fixed data parameters
repetition=1

# Data model parameters
n = 2000
dim_modalities = [2000, 400]
noise_ratios = [0.3, 0.3]
trans_type = ["linear", "linear", "quadratic"]
dim_latent = [50, 30, 20]
mod_prop = [0, 0, 1, 0]
interactive_prop = 0

if large_cohort:
    mod_outs = [[0, 50, 60, 80, 100],[0, 60, 80, 100]]
else:
    mod_outs = [[0, 50, 60, 80],[0, 60]]
num_modalities = 2
combined_hiddens = [300,200,100]
mod1_hiddens = mod2_hiddens = [[128],[128]]

# data parameters
data_name = 'regression'
exp_name = "negative_learning"
output_dim = 1  # specify the output dimension for regression

extractor_type = 'PCA'
if extractor_type == 'encoder':
    separate=False
    is_mod_static=[False]*num_modalities  
elif extractor_type == "separate":
    separate=True
    is_mod_static=[False]*num_modalities  
elif extractor_type == 'PCA':
    separate=False
    is_mod_static=[True]*num_modalities  
freeze_mod_extractors=[False]*num_modalities

# Load default model configurations 
config = load_config('../experiments_synthetic/config.json')

# Model files directory
if large_cohort:
    config['ckpt_dir'] = f"./checkpoints/{exp_name}/large_cohort_seed{seed}/"
else:
    config['ckpt_dir'] = f"./checkpoints/{exp_name}/seed{seed}/"

# Update other training parameters
config['divergence_weight_type'] = 'clustering'
config['rho_list'] = [0,1,3,5,10]
config['optimal_k'] = None
config['output_dim'] = output_dim
config["init_lr"] = 0.001
config["epochs"] = 30
#config["epochs"] = 2
config["ensemble_methods"] = [
        ]

#####################
#    Load Dataset   #
#####################
data_preparer = PrepareSyntheticData(data_name = data_name, test_size = 0.2, val_size = 0.2)
print(f"Finished generating {exp_name} dataset.")
sys.stdout.flush() 

###############
# Output file #
###############i:
outdir = f"./results/{exp_name}/"
os.makedirs(outdir, exist_ok=True)
if large_cohort:
    outfile_name = f"large_cohort_seed{seed}"
else:
    outfile_name = f"seed{seed}"
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()


# Header for results file
def add_header(df):
    results['extractor']=extractor_type
    results['weight_type'] = config['divergence_weight_type'] 
    return df

#####################
# Define Experiment #
#####################
def run_single_experiment(config, n, random_state):

    config['random_state'] = random_state
    res_list = []
    best_rho = {}
    cohort_pairs = {}
    cluster_idxs = {}


    #----------------#
    # Split dataset  #
    #----------------#
    train_loader, val_loader, test_loader, oracle_train_loader, oracle_val_loader, oracle_test_loader =\
    data_preparer.get_data_loaders(n, trans_type=trans_type, mod_prop=mod_prop, 
                                   interactive_prop = interactive_prop,
                                   dim_modalities=dim_modalities, dim_latent=dim_latent,
                                   noise_ratios=noise_ratios, random_state=random_state)
    # Get data info
    data_info = data_preparer.get_data_info()
    n = data_info[1]
    n_train = data_info[2]
    n_val = data_info[3]
    n_test = data_info[4]

    print(f"Finished splitting {data_name} dataset. Data information are summarized below:\n"
            f"Modality dimensions: {dim_modalities}\n"
            f"Data size: {n}\n"
            f"Train size: {n_train}\n"
            f"Val size: {n_val}\n"
            f"Test size: {n_test}")
    sys.stdout.flush() 


    #----------------------------#
    #  Adversarial Meta Fusion   #
    #----------------------------#
    meta_extractor = Extractors(mod_outs, dim_modalities, train_loader, val_loader)
    if (extractor_type == 'encoder') or (extractor_type == 'separate'):
        _ = meta_extractor.get_encoder_extractors(mod_hiddens, separate=separate, config=extractor_config)
    elif extractor_type == 'PCA':
        _ = meta_extractor.get_PCA_extractors()
    cohort = Cohorts(extractors=meta_extractor, combined_hidden_layers=combined_hiddens, output_dim=output_dim,
                     is_mod_static=is_mod_static, freeze_mod_extractors=freeze_mod_extractors)

    cohort_models = cohort.get_cohort_models()
    _, dim_pairs = cohort.get_cohort_info()
    trainer = AdversarialTrainer(config, cohort_models, [train_loader, val_loader])

    # Cohort with normal adaptive weights 
    trainer.train_adaptive() 
    res = trainer.test_adaptive(test_loader)
    res_list.append(res)
    
    best_rho['cohort'] = trainer.best_rho
    cohort_pairs['cohort'] = dim_pairs
    cluster_idxs['cohort'] = trainer.cluster_idxs
    print(f"Finished running adaptive meta fusion!")

    # Cohort with no mutual learning
    res = trainer.test_ablation(test_loader)
    res_list.append(res)
    
    cohort_pairs['indep_cohort'] = dim_pairs
    print(f"Finished testing indepedent cohort!")
    
    # Cohort with adversarial weights 
    trainer.train_adversarial() 
    res = trainer.test_adversarial(test_loader)
    res_list.append(res)
    
    best_rho['adversarial_cohort'] = trainer.fixed_rho
    cohort_pairs['adversarial_cohort'] = dim_pairs
    cluster_idxs['adversarial_cohort'] = trainer.cluster_idxs
    print(f"Finished running adversarial meta fusion!")

    results = []
    for res in res_list:
        for method, val in res.items():
            results.append({'Method': method, 'Test_metric': val, 
                            'best_rho':best_rho.get(method), 'cohort_pairs':cohort_pairs.get(method),
                            'cluster_idxs': cluster_idxs.get(method)})
    

    results = pd.DataFrame(results)

    results['random_state']=random_state
    results["dim_modalities"] = [dim_modalities] * len(results)
    results['n'] = n
    results['n_train'] = n_train
    results['n_val'] = n_val
    results['n_test'] = n_test 

    return results


#####################
#  Run Experiments  #
#####################
results = []

for i in tqdm(range(1, repetition+1), desc="Repetitions", leave=True, position=0):
    print(f'Running with repetition {i}...')
    random_state = repetition * (seed-1) + i
    set_random_seed(random_state)
    
    # Run experiment
    tmp = run_single_experiment(config, n, random_state)
    
    results.append(tmp)

results = pd.concat(results, ignore_index=True)
add_header(results)

#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# After the job is done, remove the model directory to free up space
if os.path.exists(config['ckpt_dir']):
    print(f"Deleting the model checkpoint directory: {config['ckpt_dir']}")
    shutil.rmtree(config['ckpt_dir'])
    print(f"Model checkpoint directory {config['ckpt_dir']} has been deleted.")
