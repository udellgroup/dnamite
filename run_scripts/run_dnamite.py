import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import yaml
from utils import get_dataset

sys.path.append('../models')
from dnamite import DNAMiteSurvival

device = "cuda:4" if torch.cuda.is_available() else "cpu"

print("DEVICE:", device)

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
    '--n_eval_times', 
    type=int, 
    default=100, 
    help='Number of discrete times used during training'
)
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=128, 
    help="Batch size for data loader."
)
parser.add_argument(
    '--hidden_dim', 
    type=int, 
    default=128, 
    help="Hidden dimension of model."
)
parser.add_argument(
    '--n_embed', 
    type=int, 
    default=8, 
    help="Embedding dimension of model."
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help="Learning rate for optimizer."
)
args = parser.parse_args()

print("GETTING DATSET...")
X, y = get_dataset(args.dataset_name)

# Do train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

categorical_feat_idx = [X.columns.get_loc(col) for col in X.columns if X[col].dtype == "category"]

model = DNAMiteSurvival(
    n_features=X_train.shape[1], 
    n_hidden=args.hidden_dim, 
    n_output=args.n_eval_times, 
    n_embed=args.n_embed,
    validation_size=0.2,
    n_val_splits=5,
    learning_rate=args.lr,
    max_epochs=100,
    batch_size=args.batch_size,
    # pair_reg_param=0.1,
    # pair_entropy_param=0.001,
    # kernel_weight=1
    # kernel_size=10,
    # pair_kernel_weight=1,
    # pair_kernel_size=10,
    # gamma=0.5,
    device=device,
    categorical_feats=categorical_feat_idx
)

from time import time  

fit_start_time = time()
print("Fitting model...")
model.fit(X_train, y_train)
fit_time = time() - fit_start_time
model.fit_time = fit_time

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/dnamite_kw0_{args.dataset_name}_seed{args.seed}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)  
    
    
print("Saving model...")
model.params_id = param_id
torch.save(
    model, 
    f"../model_saves/dnamite_kw0_{args.dataset_name}_seed{args.seed}.pt"
)

print("Evaluating...")

test_times = model.eval_times.cpu().numpy()
test_times = np.unique(np.clip(
    test_times, 
    max(y_train["time"].min(), y_test[y_test["event"]]["time"].min()) + 1e-4, 
    min(y_train["time"].max(), y_test[y_test["event"]]["time"].max()) - 1e-4
))


cdf_preds = 1 / (1 + np.exp(-model.predict(X_test, y_test)))
cdf_preds = cdf_preds[:, np.searchsorted(model.eval_times.cpu().numpy(), test_times, side="right") - 1]

risk_preds = -1 * np.log(np.clip(1 - cdf_preds, 1e-5, 10 - 1e-5))

# Get time-dependent AUC
aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_preds, test_times)

# Get normal c-index (Harrell)
cindex = concordance_index_censored(
    y_test["event"],
    y_test["time"],
    risk_preds.sum(axis=1)
)[0]

# Get brier score
brier_score = integrated_brier_score(
    y_train,
    y_test,
    1 - cdf_preds,
    test_times
)

results = pd.DataFrame(
    [["DNAMite", args.dataset_name, args.seed, cindex, mean_auc, brier_score, param_id]],
    columns=["model", "dataset", "seed", "cindex", "mean_auc", "brier_score", "param_id"]
)

results.to_csv(
    f"../metric_saves/dnamite_kw0_{args.dataset_name}_seed{args.seed}.csv", index=False
)


