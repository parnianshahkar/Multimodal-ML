import argparse
import json
import os



def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config



def update_config_with_args(config, args):
    """
    Update the configuration dictionary with command-line arguments if provided.
    """
    for key, value in vars(args).items():
        if value is not None:  # Only overwrite if a command-line argument is provided
            config[key] = value
    return config



def parse_arguments():
    """
    Parse command-line arguments to override default config values.
    """
    parser = argparse.ArgumentParser(description="Train a model using the Trainer class.")
    
    # data parameters
    parser.add_argument('--task_type', type=str, 
                        help="Task type (classification or regression)")
    parser.add_argument('--num_classes', type=int, 
                        help="Number of classes for classification")
    
    # training parameters
    parser.add_argument('--use_gpu', type=bool, 
                        help="Use GPU for training")
    parser.add_argument('--rho_list', type=float, nargs='+', 
                        help="List of rho values")
    parser.add_argument('--epochs', type=int, 
                        help="Number of training epochs")
    parser.add_argument('--init_lr', type=float, 
                        help="Initial learning rate")
    parser.add_argument('--weight_decay', type=float, 
                        help="Weight decay")
    parser.add_argument('--gamma', type=float, 
                        help="Learning rate decay factor")
    
    # ensemble parameters
    parser.add_argument('--ensemble_methods', type=str, nargs='+', 
                        help="List of ensemble methods")
    parser.add_argument('--epochs_meta_learner', type=int, 
                        help="Number of epochs for meta-learner training")

    # logging parameters
    parser.add_argument('--progress', type=bool, 
                        help="Show progress during training")
    parser.add_argument('--random_state', type=int, 
                        help="Random seed for reproducibility")
    parser.add_argument('--verbose', type=bool, 
                        help="Verbose output")
    parser.add_argument('--use_tensorboard', type=bool, 
                        help="Use TensorBoard for logging")
    parser.add_argument('--save_models', type=str, 
                        help="Directory to save model checkpoints")

    return parser.parse_args()
