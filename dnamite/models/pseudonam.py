"""
SAMPLE BEGINNING OF FILE DOCSTRING
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from .nam import BaseSingleSplitNAM
from sksurv.nonparametric import kaplan_meier_estimator
from dnamite.loss_fns import pseudo_value_loss


class PseudoNAM(nn.Module):
    """   
    PseudoNAM
    """
    
    def __init__(
        self, 
        n_features, 
        n_hidden, 
        n_output, 
        validation_size=0.2,
        n_val_splits=5,
        learning_rate=1e-4,
        max_epochs=100,
        batch_size=128,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.validation_size = validation_size
        self.n_val_splits = n_val_splits
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        
        self.model_args = kwargs
        
    def preprocess_data(self, X):
        
        if not hasattr(self, 'preprocessor'):
            self.preprocessor = make_column_transformer(
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
            
            X = self.preprocessor.fit_transform(X)
            
        else:
            X = self.preprocessor.transform(X)
            
        return X
    
    def get_pseudo_values(self, y, eval_times):
        
        print("GETTING PSEUDO VALUES")

        full_km_times, full_km = kaplan_meier_estimator(y["event"], y["time"])
        
        full_km = full_km[np.clip(np.searchsorted(full_km_times, eval_times), 0, len(full_km_times)-1)]
        
        n = len(y)

        pseudo_values = []
        for i in tqdm(range(len(y)), leave=False):
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
                    
        
    
    def get_data_loader(self, X, pseudo_values, pairs=None, shuffle=True):
        
        # Convert X to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if pairs is not None:
            if hasattr(pairs, 'values'):
                pairs = pairs.values

        if pairs is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(pairs),
                torch.FloatTensor(pseudo_values),
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(pseudo_values),
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch_mains(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, pseudo_vals in tqdm(train_loader, leave=False):

            X_main, pseudo_vals = X_main.to(self.device), pseudo_vals.to(self.device)

            y_pred = model(mains=X_main)
            
            surv_preds = 1 - torch.sigmoid(y_pred)
            
            loss = pseudo_value_loss(
                surv_preds, 
                pseudo_vals
            )
            
            if model.penalized:
                loss += model.loss_penalty()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, pseudo_vals in tqdm(test_loader, leave=False):

                X_main, pseudo_vals = X_main.to(self.device), pseudo_vals.to(self.device)

                y_pred = model(mains=X_main)
        
                preds.append(y_pred.detach())
            
                surv_preds = 1 - torch.sigmoid(y_pred)
                
                loss = pseudo_value_loss(
                    surv_preds, 
                    pseudo_vals
                )
                
                if model.penalized:
                    loss += model.loss_penalty()
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def fit_one_split(self, X_train, pseudo_vals_train, X_val, pseudo_vals_val):
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            X_train = X_train[self.selected_feats]
            X_val = X_val[self.selected_feats]
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, pseudo_vals_train)
        val_loader = self.get_data_loader(X_val, pseudo_vals_val, shuffle=False)
        
        model = BaseSingleSplitNAM(
            n_features=self.n_features, 
            n_hidden=self.n_hidden, 
            n_output=self.n_output,
            device=self.device,
            **self.model_args
        ).to(self.device)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train_(
            train_epoch_fn=self.train_epoch_mains,
            test_epoch_fn=self.test_epoch_mains,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            n_epochs=self.max_epochs
        )
        
        return model
    
    def fit(self, X, y):
        
        # Preprocess features
        X = self.preprocess_data(X)
        
        self.feature_names_in_ = X.columns
        self.n_features = X.shape[1]
        
        # Get evaluation times before train/val split
        quantiles = torch.quantile(
            torch.FloatTensor(y["time"].copy()),
            torch.linspace(0, 1, self.n_output+2)
        )

        self.eval_times = quantiles[1:-1].to(self.device)
        
        # Remove duplicates in eval times
        if len(self.eval_times.unique()) < len(self.eval_times):
            self.eval_times = self.eval_times.unique()
            self.n_output = len(self.eval_times)
        
        
        # Fit several models, one for each validation split
        self.models = []
        for i in range(self.n_val_splits):
            print("SPlIT", i)
        
            # Split the data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size, random_state=10+i)
            
            # Get pseudo values
            pseudo_vals_train = self.get_pseudo_values(y_train, self.eval_times.cpu().numpy())
            pseudo_vals_val = self.get_pseudo_values(y_val, self.eval_times.cpu().numpy())
            
            # Fit to this split
            model = self.fit_one_split(X_train, pseudo_vals_train, X_val, pseudo_vals_val)
            model.feature_names_in_ = X_train.columns   
            
            self.models.append(model)
            
        return
    
    def predict(self, X_test):
        
        # Preprocess features
        X_test = self.preprocess_data(X_test)
        
        # Make placeholder y_test
        y_test = np.zeros((X_test.shape[0], len(self.eval_times)))
        
        test_loader = self.get_data_loader(X_test, y_test, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0], len(self.eval_times)))
        for i, model in enumerate(self.models):
            _, model_preds = self.test_epoch_mains(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def get_shape_function(self, feature_name, X, eval_time, feat_min=None, feat_max=None):
        
        # placeholder y
        y = np.zeros(X.shape[0])
        
        X = self.preprocess_data(X)
        
        dfs = []
        for i, model in enumerate(self.models):
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=self.validation_size, random_state=10+i)
            
            train_loader = self.get_data_loader(X_train, y_train, shuffle=False)
            model.compute_intercept(train_loader)
            
            feat_index = model.feature_names_in_.get_loc(feature_name)
            
            if feat_min is None:
                input_min = X_train.iloc[:, feat_index].min()
                input_max = X_train.iloc[:, feat_index].max()
            else:
                scaler = self.preprocessor.transformers_[1][1][0]
                feat_index_in_scaler = scaler.feature_names_in_.tolist().index(feature_name)
                feat_mean = scaler.mean_[feat_index_in_scaler]
                feat_std = scaler.scale_[feat_index_in_scaler]
                input_min = (feat_min - feat_mean) / feat_std
                input_max = (feat_max - feat_mean) / feat_std
                
            feat_shape, feat_inputs = model.get_shape_function(feat_index, input_min, input_max, center=True)
            
            if feat_min is not None:
                feat_inputs = feat_inputs * feat_std + feat_mean
            
            eval_time_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
            feat_shape = feat_shape.cpu().numpy()[:, eval_time_index]
            
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "shape": feat_shape,
                    "input": feat_inputs.cpu().numpy().round(3),
                    "split": i
                })
            )
            
        return pd.concat(dfs)
    
    def get_calibration_data(self, X, y, eval_time, n_bins=20, method="quantile"):
        
        # First get cdf preds
        cdf_preds = 1 / (1 + np.exp(-1 * self.predict(X)))
        eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        cdf_preds = cdf_preds[:, eval_index]
        
        
        if method == "quantile":
            # Get bins from quantiles of predictions
            quantiles = np.quantile(
                cdf_preds,
                np.linspace(0, 1, n_bins+2)
            )
        elif method == "uniform":
            # Get bins from uniform spacing
            quantiles = np.linspace(cdf_preds.min(), cdf_preds.max(), n_bins+2)
        else:
            raise ValueError("method must be 'quantile' or 'uniform'")
        
        predicted = []
        observed = []
        
        for bin_ in zip(quantiles[:-1], quantiles[1:]):
            
            if bin_[0] == bin_[1]:
                bin_data = cdf_preds[
                    cdf_preds == bin_[0]
                ]
                bin_y = y[
                    cdf_preds == bin_[0]
                ]
                
            else:
                bin_data = cdf_preds[
                    np.logical_and(
                        cdf_preds >= bin_[0],
                        cdf_preds < bin_[1]
                    )
                ]
                bin_y = y[
                    np.logical_and(
                        cdf_preds >= bin_[0],
                        cdf_preds < bin_[1]
                    )
                ]

            
            times, surv_prob = kaplan_meier_estimator(bin_y["event"], bin_y["time"])
            predicted.append(bin_data.mean())
            observed.append(
                1 - surv_prob[np.clip(
                    np.searchsorted(times, eval_time), 0, len(times)-1
                )]
            )
            
            
        return np.array(predicted), np.array(observed), quantiles