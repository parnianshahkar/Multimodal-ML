import numpy as np
import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

sys.path.append('../')
from meta_fusion.benchmarks import *
from meta_fusion.methods import *
from meta_fusion.models import *
from meta_fusion.utils import *
from meta_fusion.third_party import *
from meta_fusion.config import load_config
from meta_fusion.real_data import PrepareNeuronData

os.environ['OMP_NUM_THREADS'] = '1'   # To handle the memory leak warning from MKL



#########################
# Experiment parameters #
#########################
if True:
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 4:
        print("Error: incorrect number of parameters.")
        quit()
    
    rat_name = str(sys.argv[1])
    extractor_type = str(sys.argv[2])
    seed = int(sys.argv[3])


# Fixed data parameters
repetition = 1
data_name = 'Neuron'
output_dim = 4

#mod_outs = [[100,70,50,30,20,15,0],[15,0]]
if rat_name == "Stella":
    mod_outs = [[300, 250, 200, 0],[20, 15, 10, 5, 0]]
elif rat_name ==  "Superchris":
    if extractor_type == "encoder":
        mod_outs = [[250, 200, 150, 0],[40, 35, 30, 20, 0]]
    else:
        mod_outs = [[100, 0],[39, 36, 33, 30, 27, 24, 0]]
elif rat_name ==  "Mitt":
    if extractor_type == "encoder":
        mod_outs = [[300, 250, 200, 0],[65, 60, 50, 40, 0]]
    else:
        mod_outs = [[100, 0],[70, 65, 60, 55, 50, 45, 0]]
elif rat_name ==  "Barat":
    mod_outs = [[300, 250, 200, 0],[40, 30, 20, 15, 0]]
elif rat_name ==  "Buchanan":
    mod_outs = [[250, 200, 150, 0],[38, 35, 30, 25, 0]]

num_modalities = 2
combined_hiddens = [64, 32]
mod_hiddens = [[], []]

if extractor_type == 'encoder':
    separate=False
    is_mod_static=[False]*num_modalities
    freeze_mod_extractors=[True]*num_modalities
elif extractor_type == "separate":
    separate=True
    is_mod_static=[False]*num_modalities
    freeze_mod_extractors=[True]*num_modalities
elif extractor_type == 'PCA':
    separate=False
    is_mod_static=[True]*num_modalities
    freeze_mod_extractors=[False]*num_modalities

# Load default model configurations 
config = load_config('../experiments_real/config.json')
extractor_config = load_config('../experiments_real/config_extractor.json')

ckpt_dir = f"./checkpoints/{data_name}/{rat_name}/{extractor_type}_seed{seed}/"
config['ckpt_dir'] = extractor_config['ckpt_dir'] = ckpt_dir

# Update other training parameters
config['output_dim'] = extractor_config['output_dim'] = output_dim
config["init_lr"] = 0.001
config["epochs"] = 200
#config["epochs"] = 2
config["ensemble_methods"] = [
        "simple_average",
        "weighted_average",
        "majority_voting",
        "weighted_voting",
        "meta_learner",
        "greedy_ensemble"
        ]
extractor_config["epochs"] = 200
extractor_config["init_lr"] = [0.0001, 0.001]
extractor_config["weight_decay"] = [0,0]
#extractor_config["epoch"] = 2
extractor_config["verbose"] = True



#####################
#    Load Dataset   #
#####################
data_preparer = PrepareNeuronData(rat_name,test_size = 0.1, val_size = 0.25, task_type="multiclass", 
                                  classes_to_remove=[4], oversample=False, 
                                  is_graph=False, reduce_dim=True, n_lfp_features=15)
print(f"Finished loading {data_name}:{rat_name}'s dataset.")
sys.stdout.flush() 



###############
# Output file #
###############i:
outdir = f"./results/{data_name}/{rat_name}/"
os.makedirs(outdir, exist_ok=True)
outfile_name = f"{extractor_type}_seed{seed}"
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()

# Header for results file
def add_header(df):
    results['extractor']=extractor_type
    results['rat_name']=rat_name
    return df



#####################
# Define Experiment #
#####################
def run_single_experiment(config, extractor_config, random_state, 
                          mod_outs, combined_hiddens, mod_hiddens,
                          is_mod_static, freeze_mod_extractors):

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
    train_loader, val_loader, test_loader = data_preparer.get_dataloaders(random_state=random_state)
    data_info = data_preparer.get_data_info()
    for key, value in data_info.items():
        print(f"{key}: {value}")
        sys.stdout.flush() 
    
    dim_modalities = data_info['dim_modalities']
    n = data_info['n_trials']
    n_train = data_info['train_num']
    n_val = data_info['val_num']
    n_test = data_info['test_num']

    #------------------#
    # Benchmark models #
    #------------------#
    bm_extractor = Extractors([[d,0] for d in dim_modalities], dim_modalities, train_loader, val_loader)
    _ = bm_extractor.get_dummy_extractors()
    bm_cohort = Cohorts(extractors=bm_extractor, combined_hidden_layers=combined_hiddens, output_dim=output_dim)

    #----------------------------#
    #  Proposed model: MetaFuse  #
    #----------------------------#
    mod_outs = [out+[dim] for out, dim in zip(mod_outs,dim_modalities)] # include the full modalities
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

    #------------------------------#
    #    Train and test MetaFuse   #
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
    
    best_rho['cohort'] = metafuse.best_rho
    cohort_pairs['cohort'] = dim_pairs
    cohort_pairs['indep_cohort'] = dim_pairs

    if "greedy_ensemble" in config["ensemble_methods"]:
        ens_idxs['greedy_ensemble'] = metafuse.ens_idxs  

    if config['divergence_weight_type'] == "clustering":
        cluster_idxs['cohort'] = metafuse.cluster_idxs

    print(f"Finished running meta fusion!")
    
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
    tmp = run_single_experiment(config, extractor_config, random_state, mod_outs, combined_hiddens, mod_hiddens,
                                is_mod_static, freeze_mod_extractors)
    
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
