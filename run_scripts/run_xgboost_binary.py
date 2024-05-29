import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import argparse
import pickle
import yaml
from xgboost import XGBClassifier
from utils import get_ebm_run_data


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

args = parser.parse_args()


print("----------------GETTING DATA ---------------")
data_dict = get_ebm_run_data(args.dataset_name, args.seed)
X_train = data_dict["X_train"]
X_test = data_dict["X_test"]
y_train = data_dict["y_train"]
y_test = data_dict["y_test"]

# Set the random seed
np.random.seed(args.seed)

xgb = XGBClassifier(tree_method="hist", enable_categorical=True)

print("FITTING THE XGBOOST MODEL...")
xgb.fit(X_train, y_train)

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/ebm_{args.dataset_name}_seed{args.seed}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)   
    
    
print("Saving model...")
with open(f"../model_saves/xgboost_{args.dataset_name}_seed{args.seed}.pkl", "wb") as f:
    pickle.dump(xgb, f)
