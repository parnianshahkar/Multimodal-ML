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
    
    extractor_type = str(sys.argv[1])
    seed = int(sys.argv[2])

# Fixed data parameters
repetition=1

# Data model parameters
n = 2000
[d1, d2] = dim_modalities = [500, 400]
dim_latent = [50, 30, 20]
noise_ratios = [0.4, 0.4]
trans_type = ["linear", "linear", "linear"]
mod_prop = [0, 0, 1, 0]
interactive_prop = 0

mod_outs = [[0, 200, 300, 400, 500], [0, 100, 200, 300, 400]]
#mod_outs = [[0, 500], [0, 400]]
num_modalities = 2
combined_hiddens = [128, 64]
mod_hiddens = [[256], [256]]

# data parameters
data_name = 'regression'
exp_name = data_name + "_" + "linear_late"
output_dim = 1  # specify the output dimension for regression

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
extractor_config = load_config('../experiments_synthetic/config_extractor.json')

# Model files directory
ckpt_dir = f"./checkpoints/{exp_name}/{extractor_type}_seed{seed}/"
config['ckpt_dir'] = extractor_config['ckpt_dir'] = ckpt_dir

# Update other training parameters
config['divergence_weight_type'] = 'clustering'
config['optimal_k'] = None
config['output_dim'] = extractor_config['output_dim'] = output_dim
config["init_lr"] = 0.001
config["epochs"] = 30
config["ensemble_methods"] = [
        "simple_average",
        "weighted_average",
        "meta_learner",
        "greedy_ensemble"
        ]
extractor_config["init_lr"] = 0.001
extractor_config["epoch"] = 30


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
outfile_name = f"{extractor_type}_seed{seed}"
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
def run_single_experiment(config, extractor_config, n, random_state, 
                          run_oracle=False, run_coop=True, run_all_at_once=False):

    config['random_state'] = random_state
    extractor_config['random_state'] = random_state
    res_list = []
    best_rho = {}
    cohort_pairs = {}
    ens_idxs={}
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

    #------------------#
    # Benchmark models #
    #------------------#
    bm_extractor = Extractors([[d,0] for d in dim_modalities], dim_modalities, train_loader, val_loader)
    _ = bm_extractor.get_dummy_extractors()
    bm_cohort = Cohorts(extractors=bm_extractor, combined_hidden_layers=combined_hiddens, output_dim=output_dim)

    if run_oracle:
        oracle_dims = [dim_latent[0], dim_latent[1]+dim_latent[2]]
        oracle_extractor = Extractors([[d,0] for d in oracle_dims], oracle_dims, oracle_train_loader, oracle_val_loader)
        _ = oracle_extractor.get_dummy_extractors()
        oracle_cohort = Cohorts(extractors=oracle_extractor, combined_hidden_layers=combined_hiddens, output_dim=output_dim)

    #----------------------------#
    # Proposed model: Meta Fuse  #
    #----------------------------#
    meta_extractor = Extractors(mod_outs, dim_modalities, train_loader, val_loader)
    if (extractor_type == 'encoder') or (extractor_type == 'separate'):
        _ = meta_extractor.get_encoder_extractors(mod_hiddens, separate=separate, config=extractor_config)
    elif extractor_type == 'PCA':
        _ = meta_extractor.get_PCA_extractors()
    meta_cohort = Cohorts(extractors=meta_extractor, combined_hidden_layers=combined_hiddens, output_dim=output_dim,
                          is_mod_static=is_mod_static, freeze_mod_extractors=freeze_mod_extractors)

    #------------------------------#
    #  Train and test benchmarks   #
    #------------------------------#
    bm_models = bm_cohort.get_cohort_models()
    _, bm_dims = bm_cohort.get_cohort_info()
    bm = Benchmarks(config, bm_models, bm_dims, [train_loader, val_loader])
    bm.train()
    res = bm.test(test_loader)
    res_list.append(res)
    print(f"Finished running basic benchmarks!")
    
    if run_oracle:
        oracle_config = config
        oracle_config["init_lr"] = 0.001
        oracle_models = oracle_cohort.get_cohort_models()
        _, oracle_dims = oracle_cohort.get_cohort_info()
        oracle = Benchmarks(config, oracle_models, oracle_dims, [oracle_train_loader, oracle_val_loader])
        oracle.train()
        res = oracle.test(oracle_test_loader)
        res = {f"oracle_{key}": value for key, value in res.items()}
        res_list.append(res)
        print(f"Finished running oracle benchmarks!")
        
    if run_coop:
        bm_models = bm_cohort.get_cohort_models()
        _, bm_dims = bm_cohort.get_cohort_info()    
        coop = Coop(config, bm_models, bm_dims, [train_loader, val_loader])
        coop.train()
        res = coop.test(test_loader)
        res_list.append(res)
        best_rho['coop'] = coop.best_rho
        print(f"Finished running coop!")

    
    #------------------------------#
    #  Train and test Meta Fuse    #
    #------------------------------#
    cohort_models = meta_cohort.get_cohort_models()
    _, dim_pairs = meta_cohort.get_cohort_info()
    metafuse = Trainer(config, cohort_models, [train_loader, val_loader])
    metafuse.train() 
    res = metafuse.test(test_loader)
    res_list.append(res)
    
    metafuse.train_ablation() 
    res = metafuse.test_ablation(test_loader)
    res_list.append(res)
    
    best_rho['meta_learner'] = metafuse.best_rho
    cohort_pairs['cohort'] = dim_pairs
    cohort_pairs['indep_cohort'] = dim_pairs
    
    if "greedy_ensemble" in config["ensemble_methods"]:
        ens_idxs['greedy_ensemble'] = metafuse.ens_idxs  

    if config['divergence_weight_type'] == "clustering":
        cluster_idxs['cohort'] = metafuse.cluster_idxs
    print(f"Finished running meta fusion!")
    
    
    if run_all_at_once:
        cohort_models = meta_cohort.get_cohort_models()
        config['epochs']=30
        allin1 = Trainer_all_at_once(config, cohort_models, [train_loader, val_loader])
        allin1.train()
        res = allin1.test(test_loader)
        res_list.append(res)
        best_rho['all_at_once'] = allin1.best_rho
        print(f"Finished running meta fusion all in one model!")

    
    results = []
    for res in res_list:
        for method, val in res.items():
            results.append({'Method': method, 'Test_metric': val, 
                            'best_rho':best_rho.get(method), 'cohort_pairs':cohort_pairs.get(method),
                            'ensemble_idxs': ens_idxs.get(method), 'cluster_idxs': cluster_idxs.get(method)})
    

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
    tmp = run_single_experiment(config, extractor_config, n, random_state, 
                                run_oracle=False, run_coop=True, run_all_at_once=False)
    
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
if os.path.exists(ckpt_dir):
    print(f"Deleting the model checkpoint directory: {ckpt_dir}")
    shutil.rmtree(ckpt_dir)
    print(f"Model checkpoint directory {ckpt_dir} has been deleted.")

