import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
from utils import get_dataset, discretize, get_bin_counts, get_run_data
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from tqdm import tqdm
from interpret.utils import measure_interactions
from itertools import combinations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import yaml
from models import NAM
from epoch_functions import (
    train_epoch_nam_mains,
    test_epoch_nam_mains,
    train_epoch_nam_pairs,
    test_epoch_nam_pairs
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
    "--exu",
    action="store_true",
    help="Use the exu version of the NAM model."
)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"


print("----------------GETTING DATA ---------------")
data_dict = get_run_data(args.dataset_name, args.split, preprocess=True)
X_train = data_dict["X_train"]
X_val = data_dict["X_val"]
X_test = data_dict["X_test"]
y_train = data_dict["y_train"]
y_val = data_dict["y_val"]
y_test = data_dict["y_test"]

def fit(
    train_epoch_fn, 
    test_epoch_fn,
    train_loader,
    test_loader,
    model,
    optimizer,
    n_epochs,
):
    early_stopping_counter = 0
    best_test_loss = float('inf')

    for epoch in range(n_epochs):
        
        train_loss = train_epoch_fn(model, optimizer, train_loader)
        test_loss, _ = test_epoch_fn(model, test_loader)
            
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
        else:
            early_stopping_counter += 1

        # If test loss has not improved for 5 consecutive epochs, terminate training
        if early_stopping_counter >= 5:
            print("Early stopping: Test loss has not improved for 5 consecutive epochs.")
            break

    return

# Set the random seed in numpy and pytorch
np.random.seed(args.seed)
torch.manual_seed(args.seed)



batch_size = args.batch_size

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train.values), 
    torch.FloatTensor(y_train),
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_val.values), 
    torch.FloatTensor(y_val),
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test.values), 
    torch.FloatTensor(y_test),
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = NAM(
    n_features=X_train.shape[1], 
    n_hidden=args.n_hidden, 
    n_layers=args.n_layers,
    n_output=1,
    gamma=args.gamma,
    entropy_param=args.entropy_param,
    reg_param=args.reg_param, 
    has_pairs=False,
    exu=args.exu,
    device=device,
).to(device)

n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                                
fit(    
    train_epoch_fn=train_epoch_nam_mains, 
    test_epoch_fn=test_epoch_nam_mains,
    train_loader=train_loader,
    test_loader=val_loader,
    model=model,
    optimizer=optimizer,
    n_epochs=n_epochs,
)   

selected_feats = model.active_feats.cpu().numpy()
selected_feats = X_train.columns[selected_feats]
with open(f"../feature_sets/nam_exu{args.exu}_{args.dataset_name}_seed{args.seed}.yml", "w") as f:
    yaml.dump(selected_feats.tolist(), f)

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/nam_exu{args.exu}_feature_selection_{args.dataset_name}_seed{args.seed}_split{args.split}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)  