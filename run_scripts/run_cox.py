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
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis

sys.path.append('../models')
from coxnam import CoxNAM

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
    "--n_eval_times",
    type=int,
    default=100,
    help="Number of discrete times used during evaluation."
)
args = parser.parse_args()

print("GETTING DATSET...")
X, y = get_dataset(args.dataset_name)

# Do train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

preprocessor = make_column_transformer(
    (
        OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary"),
        X.select_dtypes(include="category").columns
    ),
    (
        make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="constant", fill_value=0)
        ),
        X.select_dtypes(include="number").columns
    ),
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

model = CoxPHSurvivalAnalysis(alpha=1e-2)

from time import time  

fit_start_time = time()
model.fit(X_train, y_train)
fit_time = time() - fit_start_time
model.fit_time = fit_time

# Generate unique id to save with parameters
import random
param_id = random.randint(100000, 999999)


# Save run parameters to file
with open(f"../run_parameters/coxph_{args.dataset_name}_seed{args.seed}_params{param_id}.yaml", "w") as f:
    yaml.dump(vars(args), f)  
    
    
print("Saving model...")
model.params_id = param_id
with open(f"../model_saves/coxph_{args.dataset_name}_seed{args.seed}_params{param_id}.pkl", "wb") as f:
    pickle.dump(model, f)

print("Evaluating...")

quantiles = torch.quantile(
    torch.FloatTensor(y_train["time"].copy()),
    torch.linspace(0, 1, args.n_eval_times+2)
)

test_times = quantiles[1:-1].cpu().numpy()

# Remove duplicates in eval times
if len(np.unique(test_times)) < len(test_times):
    test_times = np.unique(test_times)
    
# Clip to y_test min and max
test_times = np.unique(np.clip(
    test_times, 
    max(y_train["time"].min(), y_test[y_test["event"]]["time"].min()) + 1e-4, 
    min(y_train["time"].max(), y_test[y_test["event"]]["time"].max()) - 1e-4
))

surv_fns = model.predict_survival_function(X_test)
cdf_preds = 1 - np.array([fn(test_times) for fn in surv_fns])

risk_preds = -1 * np.log(np.clip(1 - cdf_preds, 1e-5, 10 - 1e-5))

# Get time-dependent AUC
aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_preds, test_times)

# Get normal c-index (Harrell)
cindex = concordance_index_censored(
    y_test["event"],    
    y_test["time"],
    model.predict(X_test)
)[0]

# Get brier score
brier_score = integrated_brier_score(
    y_train,
    y_test,
    1 - cdf_preds,
    test_times
)

results = pd.DataFrame(
    [["CoxPH", args.dataset_name, args.seed, cindex, mean_auc, brier_score, param_id]],
    columns=["model", "dataset", "seed", "cindex", "mean_auc", "brier_score", "param_id"]
)

results.to_csv(
    f"../metric_saves/coxph_{args.dataset_name}_seed{args.seed}.csv", index=False
)