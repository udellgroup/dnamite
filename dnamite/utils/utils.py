import numpy as np
import pandas as pd
from interpret.utils._preprocessor import _cut_continuous
from interpret.utils._native import Native
from sksurv.nonparametric import kaplan_meier_estimator
from tqdm import tqdm
import os



# Function to discretize a feature into bins
# Based on the binning method used in EBMs in interpret package
# x: feature to discretize
# max_bins: maximum number of bins
# min_samples_per_bin: minimum number of samples per bin
# bins: optional bins to use
def discretize(x, max_bins, bins=None, min_samples_per_bin=None):
    
    # Convert x to numpy if pandas
    if isinstance(x, pd.Series):
        x = x.values
        
    if min_samples_per_bin is None:
        min_samples_per_bin = min(x.shape[0] // 100, 50)
        
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
            min_samples_bin=min_samples_per_bin,
        )
    
    # Discretize using interpret's native discretize function
    # Very similar to np.digitize, except extra bin for missing values at index 0
    x_discrete = native.discretize(x, bins)
    
    return x_discrete, bins

def get_bin_counts(x, nbins):
    """
    # Function to get bin counts from a discretized feature
    # x: discretized feature
    """
    
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

def get_pair_bin_counts(x, y):
    """
    # Function to get the bin counts for a pair of discretized features
    # x, y: discretized features
    """
    
    # Get the value counts for each unique pair in x, y
    counts = pd.crosstab(x, y)
    
    # Add in 0 counts for the 0 index of both features
    for i in range(int(x.max()) + 1):
        if i not in counts.index:
            counts.loc[i] = 0
                
    for i in range(int(y.max()) + 1):
        if i not in counts.columns:
            counts[i] = 0
    
    # Sort and flatten     
    counts = counts.sort_index(axis=0).sort_index(axis=1).values.flatten()
    
    return counts

def get_pseudo_values(y, eval_times):
    """
    # Function to get pseudo-values for a dataset
    """

    full_km_times, full_km = kaplan_meier_estimator(y["event"], y["time"])
    
    full_km = full_km[np.clip(np.searchsorted(full_km_times, eval_times), 0, len(full_km_times)-1)]
    
    n = len(y)

    pseudo_values = []
    for i in tqdm(range(len(y))):
        jackknife_times, jackknife_km = kaplan_meier_estimator(
            np.delete(y["event"], i), 
            np.delete(y["time"], i)
        )
        
        # Insert missing time
        index = np.searchsorted(jackknife_times, y["time"][i], side="right")
        
        if index == len(jackknife_times):
            # Insert at end
            jackknife_km = np.insert(jackknife_km, index, jackknife_km[index-1])
        
        elif jackknife_times[index-1] != y["time"][i]:
            jackknife_km = np.insert(jackknife_km, index, jackknife_km[index-1])
            
            
        jackknife_km = jackknife_km[
            np.clip(np.searchsorted(jackknife_times, eval_times), 0, len(jackknife_times)-1)
        ]
        
        pseudo_values.append(n * full_km - (n-1) * jackknife_km)

    return np.stack(pseudo_values, axis=0)

def get_dataset(dataset_name):
    """
    Function to get a dataset given a name
    """
        
    if dataset_name == "support":
        if not os.path.exists("data/support.csv"):
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
                
            df.to_csv("data/support.csv", index=False)
            
        data = pd.read_csv("data/support.csv")
        X = data.drop(["time", "dead"], axis=1)
        y = np.array(list(zip(data["dead"], data["time"])), dtype=[('event', 'bool'), ('time', 'float32')]) 
        
        cat_cols = ["race", "dzgroup", "income"]
        for col in cat_cols:
            X[col] = X[col].astype("category")
        
        return X, y
    
    elif dataset_name == "metabric":
        
        if not os.path.exists("data/metabric.csv"):
            
            print("Copying METABRIC dataset from DeepHit Github")
            X = pd.read_csv(
                "https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/METABRIC/cleaned_features_final.csv"
            )

            y = pd.read_csv(
                "https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/METABRIC/label.csv"
            )

            pd.concat([X, y], axis=1).to_csv("data/metabric.csv", index=False)
            
        
        data = pd.read_csv("data/metabric.csv")
        X = data.drop(["event_time", "label"], axis=1)
        y = np.array(list(zip(data["label"], data["event_time"])), dtype=[('event', 'bool'), ('time', 'float32')])
        
        return X, y
    
    elif dataset_name == "flchain":
        
        from sksurv.datasets import load_flchain
        X, y = load_flchain()

        # Change y dtypes and names
        y.dtype = np.dtype([('event', '?'), ('time', '<f8')])

        # Impute chapter with new category
        X["chapter"] = X["chapter"].astype("category")
        X["chapter"] = X["chapter"].cat.add_categories("Unknown")
        X["chapter"].fillna("Unknown", inplace=True)
        
        # Clip Training data to avoid overflow errors
        X["creatinine"] = X["creatinine"].clip(-5, 5)
        
        return X, y
    
    elif dataset_name == "unos":
        
        data = pd.read_parquet("data/patients.parquet")
        X = data.drop(["event", "time"], axis=1)
        y = np.array(list(zip(data["event"], data["time"])), dtype=[('event', 'bool'), ('time', 'float32')])
        
        return X, y
    
    elif dataset_name == "heart_failure":
        
        data = pd.read_csv("data/med_paper_pcphf_data_all_term.csv")
        X = data.drop([
            "person_id",
            "censor",
            "survival_time",
            'birth_DATETIME', 
            'death_DATE', 
            'index_date', 
            'last_encounter_date',
            'hf_date', 
            'survival_date'
        ], axis=1)
        y = np.array(list(zip(~data["censor"], data["survival_time"])), dtype=[('event', 'bool'), ('time', 'float32')])
        
        cat_cols = [
            'gender', 
            'race',
            'smoking'
        ]
        for col in cat_cols:
            X[col] = X[col].astype("category")
            
        return X, y
    
    else:
        raise ValueError(f"Dataset {dataset_name} not found")