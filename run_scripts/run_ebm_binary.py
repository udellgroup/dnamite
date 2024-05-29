import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
from utils import get_dataset, discretize, get_bin_counts, get_ebm_run_data
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from tqdm import tqdm
from interpret.utils import measure_interactions
from interpret.glassbox import ExplainableBoostingClassifier
from itertools import combinations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_name', 
    type=str, 
    help='Name of the dataset to process'
)
parser.add_argument(
    '--seed', 
    type=int, 
    default=10, 
    help='Seed to use for random number generation'
)
parser.add_argument(
    "--num_interactions",
    type=int,
    default=20,
    help="Number of interactions, obtained from FAST algorithm."
)
parser.add_argument(
    "--num_outer_bags",
    type=int,
    default=5,
    help="Number of different train/test splits to do."
)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"


print("----------------GETTING DATA ---------------")
data_dict = get_ebm_run_data(args.dataset_name, args.seed)
X_train = data_dict["X_train"]
X_test = data_dict["X_test"]
y_train = data_dict["y_train"]
y_test = data_dict["y_test"]
ebm_bags = data_dict["ebm_bags"]

# Set the random seed
np.random.seed(args.seed)

ebm = ExplainableBoostingClassifier(
    outer_bags=5,
    inner_bags=8,
    max_bins=32,
    max_rounds=1000,
    smoothing_rounds=1000,
    random_state=args.seed,
    interactions=args.num_interactions,
    validation_size=0.20
)

print("FITTING THE EBM MODEL...")
ebm.fit(X_train, y_train, bags=ebm_bags)

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/ebm_{args.dataset_name}_seed{args.seed}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)   
    
    
print("Saving model...")
with open(f"../model_saves/ebm_{args.dataset_name}_seed{args.seed}.pkl", "wb") as f:
    pickle.dump(ebm, f)
