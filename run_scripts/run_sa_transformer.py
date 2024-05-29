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
import torchtuples as tt
import pickle
import yaml
from epoch_functions import (
    train_epoch_sa_transformer,
    test_epoch_sa_transformer,
)
from models import SATransformer
from utils import get_run_data, get_run_data_survival

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    '--split',
    type=int,
    default=1,
    help='Split to use for data'
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
    '--lr',
    type=float,
    default=1e-3,
    help="Learning rate for optimizer."
)
parser.add_argument(
    '--use_feature_set', 
    action="store_true",
    help="Whether (true) or not (false) to use pre-calculate feature set from Coxnet."
)
args = parser.parse_args()

print("GETTING DATSET...")
data_dict = get_run_data_survival(args.dataset_name, seed=args.seed, split=args.split, preprocess=True, use_feature_set=args.use_feature_set)
X_train = data_dict["X_train"]
X_val = data_dict["X_val"]
X_test = data_dict["X_test"]
y_train = data_dict["y_train"]
y_val = data_dict["y_val"]
y_test = data_dict["y_test"]

# subset features if configured
# if args.use_feature_set:
#     print("SUBSETTING DATA USING COXNET FEATURES...")
#     with open(f"../feature_sets/coxnet_{args.dataset_name}_seed{args.seed}.yaml", "r") as f:
#         feats = yaml.safe_load(f)
        
#     X_train = X_train[feats]
#     X_val = X_val[feats]
#     X_test = X_test[feats]


def fit(
    train_epoch_fn, 
    test_epoch_fn,
    train_loader,
    test_loader,
    model,
    optimizer,
    n_epochs,
    eval_times, 
):
    early_stopping_counter = 0
    best_test_loss = float('inf')
    prev_test_loss = float('inf')
    learning_rate_decreased = False  # Flag to track whether learning rate has been decreased

    for epoch in range(n_epochs):
        
        train_loss = train_epoch_fn(model, optimizer, train_loader, eval_times)
        test_loss, _ = test_epoch_fn(model, test_loader, eval_times)
            
        print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f}")
            

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


train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train.values), 
    torch.BoolTensor(y_train["event"]),
    torch.FloatTensor(y_train["time"].copy())
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_val.values), 
    torch.BoolTensor(y_val["event"]),
    torch.FloatTensor(y_val["time"].copy())
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test.values), 
    torch.BoolTensor(y_test["event"]),
    torch.FloatTensor(y_test["time"].copy())
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

n_feats = X_train.shape[1]
d_model = 64
num_heads = 4
d_ff = 256
drop_prob = 0.1
n_layers = 6
eval_times = data_dict["eval_times"].to(device)

model = SATransformer(
    n_feats=n_feats, 
    d_model=d_model, 
    n_eval_times=len(eval_times), 
    n_layers=n_layers, 
    num_heads=num_heads, 
    d_ff=d_ff, 
    drop_prob=drop_prob,
    device=device
).to(device)


n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

from time import time  

fit_start_time = time()
fit(    
    train_epoch_fn=train_epoch_sa_transformer, 
    test_epoch_fn=test_epoch_sa_transformer,
    train_loader=train_loader,
    test_loader=val_loader,
    model=model,
    optimizer=optimizer,
    n_epochs=n_epochs,
    eval_times=eval_times
)
fit_time = time() - fit_start_time
model.fit_time = fit_time

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/sa_transformer_{args.dataset_name}_seed{args.seed}_split{args.split}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)  
    
    
print("Saving model...")
model.params_id = param_id
torch.save(
    model, 
    f"../model_saves/sa_transformer_{args.dataset_name}_seed{args.seed}_split{args.split}.pt"
)
