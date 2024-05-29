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
from sksurv.linear_model import CoxnetSurvivalAnalysis
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
from time import time
sys.path.append("/home/jupyter/python_scripts/utils")
from surv_stack import SurvivalStacker


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
parser.add_argument(
    "--reg_param",
    type=float,
    default=0.0,
    help="Regularization parameter for Coxnet. Set to 0 to disable feature selection."
)
parser.add_argument(
    "--subsample_rate",
    type=float,
    default=1,
    help="Subsample rate for survival stacking."
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

if args.reg_param > 0:
    
    # Do feature selection using Coxnet
    print("Doing feature selection using Coxnet...")
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[args.reg_param], max_iter=1000, tol=1e-4, verbose=0)
    
    X_train_cox = X_train.copy()
    X_test_cox = X_test.copy()
    
    coxnet_preprocessor = make_column_transformer(
        (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            X_train.select_dtypes(include="category").columns
        ),
        (
            make_pipeline(
                StandardScaler(),
                SimpleImputer(strategy="constant", fill_value=0)
            ),
            X_train.select_dtypes(include="number").columns
        ),
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    X_train_cox = coxnet_preprocessor.fit_transform(X_train_cox)
    X_test_cox = coxnet_preprocessor.transform(X_test_cox)
    
    print("FITTING COXNET...")
    coxnet.fit(X_train_cox, y_train)
    
    selected_features = X_train_cox.columns[coxnet.coef_.reshape(-1) != 0]
    
    # For one-hot encoded features, if any are selected by coxnet, select original feature
    selected_features_fixed = []
    for feat in selected_features:
        if feat in X_train.columns:
            selected_features_fixed.append(feat)
        else:
            original_feat = feat.split("_")[0]
            if original_feat not in selected_features_fixed:
                selected_features_fixed.append(original_feat)
                
    selected_features = selected_features_fixed
    
    print(f"Selected {len(selected_features)} features.")
    
    # Save selected_features to yaml
    with open(f"../feature_sets/coxnet_{args.dataset_name}_seed{args.seed}.yaml", "w") as f:
        yaml.dump(selected_features, f)
        
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

# Set the random seed
np.random.seed(args.seed)

X_train_stacked = X_train.copy()

# Ordinal encode the categorical features
les = {}
for col in X_train_stacked.select_dtypes(include="category").columns:
    le = LabelEncoder()
    X_train_stacked[col] = le.fit_transform(X_train_stacked[col])
    les[col] = le

# Append ebm bags so keep track of train/val splits
X_train_stacked = pd.concat([
    X_train_stacked.reset_index(drop=True), 
    pd.DataFrame(ebm_bags.T, columns=[f"ebm_bag_{i}" for i in range(len(ebm_bags))])
], axis=1)


print("RUNNING SURVIVAL STACKING WITH SUBSAMPLE RATE ", args.subsample_rate)
stacker = SurvivalStacker(discrete_time=False, sampling_ratio=args.subsample_rate)
# stacker = SurvivalStacker(discrete_time=False, sampling_ratio=1)
stacker.fit(y_train["time"], ~y_train["event"])
X_train_stacked, y_train_stacked = stacker.transform(X_train_stacked)

# Transform the encoded features back to their original values
for col in X_train_stacked.select_dtypes(include="category").columns:
    X_train_stacked[col] = les[col].inverse_transform(X_train_stacked[col].astype(int))
    
# Remove the ebm bag columns
ebm_bags = X_train_stacked[[c for c in X_train_stacked.columns if "ebm_bag" in c]].values.T
X_train_stacked = X_train_stacked.drop(columns=[c for c in X_train_stacked.columns if "ebm_bag" in c], axis=1)


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
start_time = time()
ebm.fit(X_train_stacked, y_train_stacked, bags=ebm_bags)
fit_time = time() - start_time

ebm.fit_time = fit_time

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/ebm_{args.dataset_name}_seed{args.seed}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)   
    
    
print("Saving model...")
with open(f"../model_saves/ebm_{args.dataset_name}_seed{args.seed}.pkl", "wb") as f:
    pickle.dump(ebm, f)
