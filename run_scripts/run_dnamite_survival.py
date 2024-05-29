import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
from utils import get_dataset, discretize, get_bin_counts, get_discetized_run_data_survival
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sksurv.nonparametric import CensoringDistributionEstimator
from tqdm import tqdm
from interpret.utils import measure_interactions
from itertools import combinations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import yaml
from models import DNAMite
from epoch_functions import (
    train_epoch_nam_survival_mains,
    test_epoch_nam_survival_mains,
    train_epoch_nam_survival_pairs,
    test_epoch_nam_survival_pairs
)


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
    "--split",
    type=int,
    default=1,
    help="Split (seed) number for train/val split"
)
parser.add_argument(
    '--reg_param', 
    type=float, 
    default=0, 
    help="Penalty parameter for regularizer. 0 indicated no feature sparse penalty"
)
parser.add_argument(
    '--entropy_param', 
    type=float, 
    default=0, 
    help="Penalty parameter for entropy. Higher value makes smooth zs converge to {0, 1} faster."
)
parser.add_argument(
    '--gamma', 
    type=float, 
    default=1, 
    help="Gamma for smooth step function. Lower values make smooth step function in {0, 1} faster."
)
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=128, 
    help="Batch size for data loader."
)
parser.add_argument(
    '--lr', 
    type=float, 
    default=1e-4, 
    help="Learning rate to use."
)
parser.add_argument(
    '--max_bins',
    type=int,
    default=32,
    help="Maximum number of bins to use for discretization."
)
parser.add_argument(
    "--n_hidden",
    type=int,
    default=32,
    help="The dimension of the hidden layers."   
)
parser.add_argument(
    "--n_embed",
    type=int,
    default=32,
    help="The embedding dimension for each feature."
)
parser.add_argument(
    "--n_layers",
    type=int,
    default=2,
    help="Number of hidden layers in the network."
)
parser.add_argument(
    "--kernel_size",
    type=int,
    default=10,
    help="Kernel size for kernel smoothing."
)
parser.add_argument(
    "--kernel_weight",
    type=float,
    default=1,
    help="Kernel weight to control smoothness of interaction shape functions."
)
parser.add_argument(
    "--pair_kernel_size",
    type=int,
    default=10,
    help="Kernel size for kernel smoothing."
)
parser.add_argument(
    "--pair_kernel_weight",
    type=float,
    default=1,
    help="Kernel weight to control smoothness of interaction shape functions."
)
parser.add_argument(
    "--n_eval_times",
    type=int,
    default=100,
    help="Number of evaluation times to use for prediction."
)
parser.add_argument(
    "--use_feature_set",
    action="store_true",
    help="Use a pre-defined feature set for the model mains AND pairs (since FAST is not possible)."
)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"


print("----------------GETTING DATA ---------------")
# data_dict = get_discetized_run_data(args.dataset_name, args.seed, args.split, max_bins=args.max_bins)
data_dict = get_discetized_run_data_survival(
    args.dataset_name, 
    args.seed, 
    args.split, 
    args.max_bins, 
    args.n_eval_times,
    use_feature_set=args.use_feature_set
)

if args.use_feature_set:
    selected_feats = data_dict["selected_feats"]
    X_train_discrete = data_dict["X_train_discrete"].iloc[:, selected_feats]
    X_val_discrete = data_dict["X_val_discrete"].iloc[:, selected_feats]
    X_test_discrete = data_dict["X_test_discrete"].iloc[:, selected_feats]
else:
    X_train_discrete = data_dict["X_train_discrete"]
    X_val_discrete = data_dict["X_val_discrete"]
    X_test_discrete = data_dict["X_test_discrete"]


def fit(
    train_epoch_fn, 
    test_epoch_fn,
    train_loader,
    test_loader,
    model,
    optimizer,
    n_epochs,
    eval_times, 
    pcw_eval_times=None
):
    early_stopping_counter = 0
    best_test_loss = float('inf')
    prev_test_loss = float('inf')
    learning_rate_decreased = False  # Flag to track whether learning rate has been decreased

    for epoch in range(n_epochs):
        
        if pcw_eval_times is not None:
            train_loss = train_epoch_fn(model, optimizer, train_loader, eval_times, pcw_eval_times)
            test_loss, test_preds = test_epoch_fn(model, test_loader, eval_times, pcw_eval_times)
        else:
            train_loss = train_epoch_fn(model, optimizer, train_loader, eval_times)
            test_loss, test_preds = test_epoch_fn(model, test_loader, eval_times)
            
        print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f} | Num Feats: {len([z for z in model.get_smooth_z() if z > 0])}")
            
        
        # Do feature pruning
        if model.has_pairs:
            model.prune_parameters(mains=True, pairs=True)
        else:
            model.prune_parameters(mains=True)

        # Check if the test loss has improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stopping_counter = 0
            
            # Save the model at the best test loss
            torch.save(model.state_dict(), "tmp_run_saves/best_model_survival.pt")
            
        else:
            early_stopping_counter += 1

        # If test loss has not improved for 5 consecutive epochs, terminate training
        if early_stopping_counter >= 5:
            print("Early stopping: Test loss has not improved for 5 consecutive epochs.")
            break
        
    # Load the model from the best test loss
    model.load_state_dict(torch.load("tmp_run_saves/best_model_survival.pt"))

    return

# Set the random seed in numpy and pytorch
np.random.seed(args.seed)
torch.manual_seed(args.seed)
        

batch_size = args.batch_size

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train_discrete.values), 
    torch.BoolTensor(data_dict["y_train"]["event"]),
    torch.FloatTensor(data_dict["y_train"]["time"].copy()),
    torch.FloatTensor(data_dict["pcw_obs_times_train"])
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_val_discrete.values), 
    torch.BoolTensor(data_dict["y_val"]["event"]),
    torch.FloatTensor(data_dict["y_val"]["time"].copy()),
    torch.FloatTensor(data_dict["pcw_obs_times_val"])
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test_discrete.values), 
    torch.BoolTensor(data_dict["y_test"]["event"]),
    torch.FloatTensor(data_dict["y_test"]["time"].copy()),
    torch.FloatTensor(data_dict["pcw_obs_times_test"])
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = DiscreteNAM(
    n_features=X_train_discrete.shape[1], 
    n_embed=args.n_embed,
    n_hidden=args.n_hidden, 
    n_layers=args.n_layers,
    n_output=args.n_eval_times,
    feature_sizes=data_dict["n_bins"],
    gamma=args.gamma,
    entropy_param=args.entropy_param,
    reg_param=args.reg_param, 
    has_pairs=False,
    device=device,
    kernel_size=args.kernel_size,
    kernel_weight=args.kernel_weight
).to(device)

n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
eval_times = data_dict["eval_times"].to(device)
pcw_eval_times = data_dict["pcw_eval_times"].to(device)

from time import time
                     
fit_main_start_time = time()           
fit(    
    train_epoch_fn=train_epoch_dys_mains, 
    test_epoch_fn=test_epoch_dys_mains,
    train_loader=train_loader,
    test_loader=val_loader,
    model=model,
    optimizer=optimizer,
    n_epochs=n_epochs,
    eval_times=eval_times,
    pcw_eval_times=pcw_eval_times
)
fit_main_time = time() - fit_main_start_time

if args.use_feature_set:

    selected_pairs = data_dict["selected_pairs"]
    
    # Get interactions from full X, not subseted X
    X_train_interactions = data_dict["X_train_discrete"].values[:, selected_pairs]
    X_val_interactions = data_dict["X_val_discrete"].values[:, selected_pairs]
    X_test_interactions = data_dict["X_test_discrete"].values[:, selected_pairs]
    
else:
    active_feats = model.active_feats.cpu().numpy()
    selected_pairs = list(combinations(active_feats, 2))
    
    X_train_interactions = X_train_discrete.values[:, selected_pairs]
    X_val_interactions = X_val_discrete.values[:, selected_pairs]
    X_test_interactions = X_test_discrete.values[:, selected_pairs]
    
    # Add small amount of regularization
    # model.reg_param = 1e-2
    # model.entropy_param = 1e-3
    # model.gamma = 0.01
    # model.penalized = True


model.freeze_main_effects()
model.has_pairs = True
model.pair_kernel_size = args.pair_kernel_size
model.pair_kernel_weight = args.pair_kernel_weight
model.pairs_list = torch.IntTensor(selected_pairs).to(device)
model.n_pairs = len(selected_pairs)
model.init_pairs_params(model.n_pairs)
model.active_pairs = torch.arange(model.n_pairs).to(device)
model.to(device)

def fit_pairs(
    train_epoch_fn, 
    test_epoch_fn,
    train_loader,
    test_loader,
    model,
    optimizer,
    n_epochs,
    eval_times, 
    pcw_eval_times=None
):
    early_stopping_counter = 0
    best_test_loss = float('inf')
    prev_test_loss = float('inf')
    learning_rate_decreased = False  # Flag to track whether learning rate has been decreased

    for epoch in range(n_epochs):
        
        if pcw_eval_times is not None:
            train_loss = train_epoch_fn(model, optimizer, train_loader, eval_times, pcw_eval_times)
            test_loss, _ = test_epoch_fn(model, test_loader, eval_times, pcw_eval_times)
        else:
            train_loss = train_epoch_fn(model, optimizer, train_loader, eval_times)
            test_loss, _ = test_epoch_fn(model, test_loader, eval_times)
            
        print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f} | Num Feats: {len([z for z in model.get_smooth_z() if z > 0])} | Num Pairs: {len([z for z in model.get_smooth_z_pairs() if z > 0])}")            
        
        # Do feature pruning
        if model.has_pairs:
            model.prune_parameters(mains=True, pairs=True)
        else:
            model.prune_parameters(mains=True)

        # Check if the test loss has improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stopping_counter = 0
            
            # Save the model at the best test loss
            torch.save(model.state_dict(), "tmp_run_saves/best_model_survival.pt")
            
        else:
            early_stopping_counter += 1

        # If test loss has not improved for 5 consecutive epochs, terminate training
        if early_stopping_counter >= 5:
            print("Early stopping: Test loss has not improved for 5 consecutive epochs.")
            break
        
    # Load the model from the best test loss
    model.load_state_dict(torch.load("tmp_run_saves/best_model_survival.pt"))

    return


batch_size = args.batch_size

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train_discrete.values), 
    torch.FloatTensor(X_train_interactions),
    torch.BoolTensor(data_dict["y_train"]["event"]),
    torch.FloatTensor(data_dict["y_train"]["time"].copy()),
    torch.FloatTensor(data_dict["pcw_obs_times_train"])
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_val_discrete.values),
    torch.FloatTensor(X_val_interactions), 
    torch.BoolTensor(data_dict["y_val"]["event"]),
    torch.FloatTensor(data_dict["y_val"]["time"].copy()),
    torch.FloatTensor(data_dict["pcw_obs_times_val"])
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test_discrete.values),
    torch.FloatTensor(X_test_interactions), 
    torch.BoolTensor(data_dict["y_test"]["event"]),
    torch.FloatTensor(data_dict["y_test"]["time"].copy()),
    torch.FloatTensor(data_dict["pcw_obs_times_test"])
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from functools import partial

n_epochs = 100
# optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr / 5)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    
fit_pairs_start_time = time()                         
fit_pairs(    
    train_epoch_fn=partial(train_epoch_dys_pairs, model_mains=model), 
    test_epoch_fn=partial(test_epoch_dys_pairs, model_mains=model),
    train_loader=train_loader,
    test_loader=val_loader,
    model=model,
    optimizer=optimizer,
    n_epochs=n_epochs,
    eval_times=eval_times,
    pcw_eval_times=pcw_eval_times
)
fit_pairs_time = time() - fit_pairs_start_time

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/discrete_nam_survival_{args.dataset_name}_seed{args.seed}_split{args.split}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)  
    
    
print("Saving model...")
model.params_id = param_id
model.fit_time = fit_main_time + fit_pairs_time
torch.save(
    model, 
    f"../model_saves/discrete_nam_survival_{args.dataset_name}_seed{args.seed}_split{args.split}.pt"
)