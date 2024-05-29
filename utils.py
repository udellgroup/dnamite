import numpy as np
import pandas as pd
from interpret.utils._preprocessor import _cut_continuous
from interpret.utils._native import Native
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sksurv.nonparametric import CensoringDistributionEstimator
import yaml
import torch
import os


# Function to get a dataset given a name
def get_dataset(dataset_name):
    if dataset_name == "higgs":
        
        data = fetch_openml(data_id=23512, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)

        X = X.iloc[:-1, :] # last columns has NAs
        y = y[:-1]
        
        return X, y
        
    elif dataset_name == "heart_failure_survival":
        train_data = pd.read_parquet(
            f"/home/jupyter/omop_data/processed_data/med_paper_with_units_all_term_train.parquet", engine="fastparquet"
        )

        test_data = pd.read_parquet(
            f"/home/jupyter/omop_data/processed_data/med_paper_with_units_all_term_test.parquet", engine="fastparquet"
        )
        
        data = pd.concat([train_data, test_data], axis=0)
        
        cat_cols = [
            c for c in train_data.columns if "cat" in c \
                or c == "gender" or c == "race" or c == "smoking" \
                or "condition" in c or "drug" in c
        ]
        
        # cat_cols = [
        #     c for c in train_data.columns if "cat" in c \
        #         or c == "gender" or c == "race" or c == "smoking"
        # ]
        
        for col in cat_cols:
            data[col] = train_data[col].astype("category")
            
        non_feat_cols = [
            "birth_DATETIME",
            "death_DATE",
            "index_date",
            "last_encounter_date",
            "hf_date",
            "censor",
            "survival_date",
            "survival_time",
            "person_id",
        ]
        
        X = data.drop(non_feat_cols, axis=1)

        y = np.array(
            list(zip(~data["censor"], data["survival_time"])),
            dtype=[("event", "?"), ("time", "f8")]
        )
        
        return X, y
    
    elif dataset_name == "heart_failure_binary":
        train_data = pd.read_parquet(
            f"/home/jupyter/omop_data/processed_data/med_paper_with_units_all_term_train.parquet", engine="fastparquet"
        )

        test_data = pd.read_parquet(
            f"/home/jupyter/omop_data/processed_data/med_paper_with_units_all_term_test.parquet", engine="fastparquet"
        )
        
        data = pd.concat([train_data, test_data], axis=0)
        
        data = pd.concat([train_data, test_data], axis=0)
        
        cat_cols = [c for c in train_data.columns if "cat" in c or c == "gender" or c == "race" or c == "smoking"]
        for col in cat_cols:
            data[col] = train_data[col].astype("category")
            
        eval_time = 365 * 5
        
        # keep only rows which are not censored before eval_time
        rows_to_remove = data.loc[(data["censor"]) & (data["survival_time"] < eval_time)].index
        data = data.drop(rows_to_remove, axis=0)
            
        non_feat_cols = [
            "birth_DATETIME",
            "death_DATE",
            "index_date",
            "last_encounter_date",
            "hf_date",
            "censor",
            "survival_date",
            "survival_time",
            "person_id",
        ]
        
        X = data.drop(non_feat_cols, axis=1)
        y = (~data["censor"]).astype(int).values
        
        return X, y
    
    elif dataset_name == "support":
        if not os.path.exists("/home/jupyter/python_scripts/discrete_nam/data/support.csv"):
            print("Copying SUPPORT dataset from survml Github")
            
            df = pd.read_csv("https://raw.githubusercontent.com/survml/survml-deepsurv/main/data/support_parsed.csv")
            
            # Undo one-hot encoding
            prefixes = ["race", "dzgroup", "income"]
            
            def find_value(row):
                candidates = [
                    part for part in row[related_columns].index if row[part] == 1
                ]
                if len(candidates) == 0:
                    return "other"
                else:
                    return candidates[0].split(prefix)[1]

            for prefix in prefixes:
                related_columns = [col for col in df.columns if col.startswith(prefix)]
                df[prefix] = df.apply(find_value, axis=1)
                
            df.to_csv("/home/jupyter/python_scripts/discrete_nam/data/support.csv", index=False)
            
        data = pd.read_csv("/home/jupyter/python_scripts/discrete_nam/data/support.csv")
        X = data.drop(["time", "dead"], axis=1)
        y = np.array(list(zip(data["dead"], data["time"])), dtype=[('event', 'bool'), ('time', 'float32')]) 
        
        cat_cols = ["race", "dzgroup", "income"]
        for col in cat_cols:
            X[col] = X[col].astype("category")
        
        return X, y
    
    elif dataset_name == "unos":
        data = pd.read_parquet(
            "/home/jupyter/python_scripts/discrete_nam/data/kidpan_data.parquet",
            engine="fastparquet"
        )
        
        # Subset to only data where ftime is > 0
        data = data[data["ftime"] > 0]
        
        X = data.drop(["ftime", "event"], axis=1)
        y = np.array(list(zip(data["event"], data["ftime"])), dtype=[('event', 'bool'), ('time', 'float32')])
        
        cat_cols = ["ABO", "ON_DIALYSIS", "GENDER", "MALIG_TCR_KI", "PERIP_VASC", "DRUGTRT_COPD"]
        for col in cat_cols:
            X[col] = X[col].astype("category")
            
        return X, y
    
    elif dataset_name == "housing":
        data = fetch_openml(data_id=44090, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        
        return X, y
    
    elif dataset_name == "phoneme":
        data = fetch_openml(data_id=1489, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    elif dataset_name == "miniboone":
        data = fetch_openml(data_id=41150, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    elif dataset_name == "spam":
        data = fetch_openml(data_id=44, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    elif dataset_name == "adult":
        data = fetch_openml(data_id=1590, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    elif dataset_name == "churn":
        data = fetch_openml(data_id=42178, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        
        # Convert TotalCharges to Numeric
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        # Convert all object columns to category
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")

        return X, y
        
    elif dataset_name == "space":
        data = fetch_openml(data_id=737, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    elif dataset_name == "philippine":
        data = fetch_openml(data_id=41145, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    elif dataset_name == "albert":
        data = fetch_openml(data_id=41147, parser="auto")
        X, y = data["data"].copy(deep=False), data["target"].copy(deep=False)
        y = LabelEncoder().fit_transform(y)
        return X, y
    
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

# Function to discretize a feature into bins
# Based on the binning method used in EBMs in interpret package
# x: feature to discretize
# max_bins: maximum number of bins
# bins: optional bins to use
def discretize(x, max_bins, bins=None):
    
    # Convert x to numpy if pandas
    if isinstance(x, pd.Series):
        x = x.values
        
    # Convert to float if not float
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    
    # Get "native" from interpret package
    native = Native.get_native_singleton()
    
    if bins is None:
        
        # Compute bins based on quantiles
        bins = _cut_continuous(
            native=native,
            X_col=x,
            processing="quantile",
            binning=None,
            max_bins=max_bins,
            min_samples_bin=1,
        )
    
    # Discretize using interpret's native discretize function
    # Very similar to np.digitize, except extra bin for missing values at index 0
    x_discrete = native.discretize(x, bins)
    
    return x_discrete, bins

# Function to get bin counts from a discretized feature
# x: discretized feature
def get_bin_counts(x, nbins):
    
    # Convert to pandas series if numpy
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    
    # Get counts for each bin
    counts = x.value_counts().sort_index()
    
    # Add a count of 0 for any bins that are missing
    counts_corrected = np.zeros(nbins)
    for i in range(nbins):
        if i not in counts.index:
            counts_corrected[i] = 0
        else:
            counts_corrected[i] = counts[counts.index == i].values[0]
        
    return counts_corrected

# Function to get the bin counts for a pair of discretized features
# x, y: discretized features
def get_pair_bin_counts(x, y):
    
    # Get the value counts for each unique pair in x, y
    counts = pd.crosstab(x, y)
    
    # Add in 0 counts for the 0 index of both features
    for i in range(x.max() + 1):
        if i not in counts.index:
            counts.loc[i] = 0
                
    for i in range(y.max() + 1):
        if i not in counts.columns:
            counts[i] = 0
    
    # Sort and flatten     
    counts = counts.sort_index(axis=0).sort_index(axis=1).values.flatten()
    
    return counts

def get_discetized_run_data(dataset, seed=10, split=1, max_bins=32, use_feature_set=False):
    X, y = get_dataset(dataset)
    
    cat_cols = X.select_dtypes(include="category").columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    
    preprocessor = make_column_transformer(
        (
            OrdinalEncoder(dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan),
            X_train.select_dtypes(include="category").columns
        ),
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    feature_bins = []
    n_bins = [] 
    
    X_train_discrete = X_train.copy()
    X_test_discrete = X_test.copy()
    
    for col in X_train.columns:
        X_train_discrete[col], bins = discretize(X_train[col], max_bins=max_bins)
        X_test_discrete[col], _ = discretize(X_test[col], max_bins=max_bins, bins=bins)
        
        feature_bins.append(bins)
    
        # Number of bins is (maximum bin index) + 1 (accounting for missing bin)
        n_bins.append(X_train_discrete[col].max() + 1)
        
    # make train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_discrete)), test_size=0.20, random_state=10 + split
    )
    X_train_discrete, X_val_discrete = X_train_discrete.iloc[train_idx], X_train_discrete.iloc[val_idx]
    X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]
    
    cat_cols_indices = [X_train.columns.get_loc(col) for col in cat_cols]
    
    if use_feature_set:
        
        with open(f"/home/jupyter/python_scripts/discrete_nam/experimental/feature_sets/discrete_nam_{dataset}_seed{seed}.yml", "r") as f:
            selected_feats = yaml.safe_load(f)
            
        selected_feat_indices = [X_train_discrete.columns.get_loc(col) for col in selected_feats]
        
        X_train = X_train[selected_feats]
        X_val = X_val[selected_feats]
        X_test = X_test[selected_feats]

        X_train_discrete = X_train_discrete[selected_feats]
        X_val_discrete = X_val_discrete[selected_feats]
        X_test_discrete = X_test_discrete[selected_feats]

        feature_bins = [feature_bins[i] for i in selected_feat_indices]
        n_bins = [n_bins[i] for i in selected_feat_indices]
        
        cat_cols_indices = [selected_feat_indices.index(i) for i in cat_cols_indices if i in selected_feat_indices]
        
    # return X_train_discrete, X_val_discrete, X_test_discrete, y_train, y_val, y_test, feature_bins, n_bins
    return {
        "X_train_discrete": X_train_discrete,
        "X_val_discrete": X_val_discrete,
        "X_test_discrete": X_test_discrete,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_bins": feature_bins,
        "n_bins": n_bins,
        "cat_cols_indices": cat_cols_indices,
        "preprocessor": preprocessor
    }
    
def get_run_data(dataset, seed=10, split=1, exu=False, preprocess=True, use_feature_set=False):
    
    X, y = get_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    # Get validation data from train data
    # Use a different train/val split to mimic "outer_bags" from EBM
    # Do this after discretization so binning remains the same
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.20, random_state=10 + split
    )
    X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]

    if preprocess:
        preprocessor = make_column_transformer(
            (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary"),
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

        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)
    else:
        preprocessor = None
    
    # return X_train, X_val, X_test, y_train, y_val, y_test
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor
    }
    
def get_ebm_run_data(dataset, seed, num_outer_bags=5, use_feature_set=False):
    X, y = get_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    
    ebm_bags = np.ones((num_outer_bags, len(X_train)))

    for i in range(num_outer_bags):
        idx_train, idx_val = train_test_split(
            np.arange(len(X_train)), test_size=0.20, random_state=10 + (i+1)
        )
        ebm_bags[i, idx_val] = -1
        
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "ebm_bags": ebm_bags
    }
    
def get_discetized_run_data_survival(dataset, seed=10, split=1, max_bins=32, n_eval_times=100, use_feature_set=False):
    X, y = get_dataset(dataset)
    
    cat_cols = X.select_dtypes(include="category").columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    
    # Get evaluation times before train/val split
    quantiles = torch.quantile(
        torch.FloatTensor(y_train["time"].copy()),
        torch.linspace(0, 1, n_eval_times+2)
    )

    eval_times = quantiles[1:-1]
    
    preprocessor = make_column_transformer(
        (
            OrdinalEncoder(dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan),
            X_train.select_dtypes(include="category").columns
        ),
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    feature_bins = []
    n_bins = [] 
    
    X_train_discrete = X_train.copy()
    X_test_discrete = X_test.copy()
    
    for col in X_train.columns:
        X_train_discrete[col], bins = discretize(X_train[col], max_bins=max_bins)
        X_test_discrete[col], _ = discretize(X_test[col], max_bins=max_bins, bins=bins)
        
        feature_bins.append(bins)
    
        # Number of bins is (maximum bin index) + 1 (accounting for missing bin)
        n_bins.append(X_train_discrete[col].max() + 1)
        
    # make train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_discrete)), test_size=0.20, random_state=10 + split
    )
    X_train_discrete, X_val_discrete = X_train_discrete.iloc[train_idx], X_train_discrete.iloc[val_idx]
    X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]
    
    cat_cols_indices = [X_train.columns.get_loc(col) for col in cat_cols]
    
    if use_feature_set:
        
        with open(f"/home/jupyter/python_scripts/discrete_nam/experimental/feature_sets/discrete_nam_survival_{dataset}_seed{seed}", "r") as f:
            selected_feats_and_pairs = yaml.safe_load(f)
    
        selected_feats = selected_feats_and_pairs["selected_feats"]
        
        feature_bins = [feature_bins[i] for i in selected_feats]
        n_bins = [n_bins[i] for i in selected_feats]
        
        cat_cols_indices = [i for i in cat_cols_indices if i in selected_feats]
        
        selected_pairs = selected_feats_and_pairs["selected_pairs"]

        # Need to map the pairs (which contain indices from all columns) to subseted columns
        selected_pairs = np.searchsorted(selected_feats, selected_pairs)

    # print("Number of categorical features", len(cat_cols_indices))

    cde = CensoringDistributionEstimator()
    cde.fit(y_train)

    pcw_obs_times_train = cde.predict_proba(y_train["time"]) + 1e-5
    pcw_obs_times_val = cde.predict_proba(y_val["time"]) + 1e-5
    pcw_obs_times_test = cde.predict_proba(y_test["time"]) + 1e-5

    pcw_eval_times = torch.FloatTensor(cde.predict_proba(eval_times.cpu().numpy())) + 1e-5
        
    # return X_train_discrete, X_val_discrete, X_test_discrete, y_train, y_val, y_test, feature_bins, n_bins
    data_dict =  {
        "X_train_discrete": X_train_discrete,
        "X_val_discrete": X_val_discrete,
        "X_test_discrete": X_test_discrete,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_bins": feature_bins,
        "n_bins": n_bins,
        "cat_cols_indices": cat_cols_indices,
        "eval_times": eval_times,
        "pcw_obs_times_train": pcw_obs_times_train,
        "pcw_obs_times_val": pcw_obs_times_val,
        "pcw_obs_times_test": pcw_obs_times_test,
        "pcw_eval_times": pcw_eval_times,
    }
    
    if use_feature_set:
        data_dict["selected_feats"] = selected_feats
        data_dict["selected_pairs"] = selected_pairs
        
    return data_dict
    
    
def get_run_data_survival(dataset, seed=10, split=1, preprocess=True, n_eval_times=100, use_feature_set=False):
    
    X, y = get_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    
    # Get evaluation times before train/val split
    quantiles = torch.quantile(
        torch.FloatTensor(y_train["time"].copy()),
        torch.linspace(0, 1, n_eval_times+2)
    )

    eval_times = quantiles[1:-1]

    # Get validation data from train data
    # Use a different train/val split to mimic "outer_bags" from EBM
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.20, random_state=10 + split
    )
    X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train, y_val = y_train[train_idx], y_train[val_idx]
    
    if use_feature_set:
        
        with open(f"../feature_sets/coxnet_{dataset}_seed{seed}.yaml", "r") as f:
            selected_feats = yaml.safe_load(f)
        
        X_train = X_train[selected_feats]
        X_val = X_val[selected_feats]
        X_test = X_test[selected_feats]

    if preprocess:
        preprocessor = make_column_transformer(
            (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary"),
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

        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)
    else:
        preprocessor = None
    
    # return X_train, X_val, X_test, y_train, y_val, y_test
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "eval_times": eval_times
    }
    