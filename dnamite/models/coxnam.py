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
from sksurv.linear_model.coxph import BreslowEstimator
from dnamite.loss_fns import coxph_loss


class CoxNAM(nn.Module):
    """ 
    CoxNAM
    """
    
    def __init__(
        self, 
        n_features, 
        n_hidden, 
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
        
    
    def get_data_loader(self, X, y, pairs=None, shuffle=True):
        
        assert "time" in y.dtype.names and "event" in y.dtype.names, \
            "y must be a structured array with 'time' and 'event' fields."
        
        # Convert X to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
            
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), 
            torch.BoolTensor(y["event"]),
            torch.FloatTensor(y["time"].copy()),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch_mains(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, events, times in tqdm(train_loader, leave=False):

            X_main, events, times = X_main.to(self.device), events.to(self.device), times.to(self.device)

            y_pred = model(mains=X_main).squeeze()
            
            loss = coxph_loss(
                y_pred,
                events,
                times
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
            for X_main, events, times in tqdm(test_loader, leave=False):

                X_main, events, times = X_main.to(self.device), events.to(self.device), times.to(self.device)

                y_pred = model(mains=X_main).squeeze()
        
                preds.append(y_pred.detach())
            
                loss = coxph_loss(
                    y_pred,
                    events,
                    times
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
            n_output=1,
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
    
    def fit(self, X, y, fit_baseline=True):
        
        # Preprocess features
        X_processed = self.preprocess_data(X)
        
        self.feature_names_in_ = X_processed.columns
        self.n_features = X_processed.shape[1]
        
        # Fit several models, one for each validation split
        self.models = []
        for i in range(self.n_val_splits):
            print("SPlIT", i)
        
            # Split the data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=self.validation_size, random_state=10+i)
            
            # Fit to this split
            model = self.fit_one_split(X_train, y_train, X_val, y_val)
            model.feature_names_in_ = X_train.columns   
            
            self.models.append(model)
            
        if fit_baseline:
            self.fit_baseline(X, y)
            
        return
    
    def predict(self, X_test):
        
        # Preprocess features
        X_test = self.preprocess_data(X_test)
        
        # Make placeholder y_test
        y_test = np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        test_loader = self.get_data_loader(X_test, y_test, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0]))
        for i, model in enumerate(self.models):
            _, model_preds = self.test_epoch_mains(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def fit_baseline(self, X, y):
        
        self.breslow = BreslowEstimator()
        pred = self.predict(X)
        self.breslow.fit(pred, y["event"], y["time"])
        
        return
    
    def predict_survival(self, X, eval_times):
        
        if not hasattr(self, 'breslow'):
            raise ValueError("Must fit baseline model first.")
        
        pred = self.predict(X)
        surv_fns = self.breslow.get_survival_function(pred)
        return np.array([fn(eval_times) for fn in surv_fns])

    
    def get_shape_function(self, feature_name, X, eval_time, feat_min=None, feat_max=None):
        
        # placeholder y
        y = np.zeros(X.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        X = self.preprocess_data(X)
        
        dfs = []
        for i, model in enumerate(self.models):
            
            print("SPLIT", i)
            print("X shape", X.shape)
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=self.validation_size, random_state=10+i)
            
            print("X_train shape", X_train.shape)
            
            train_loader = self.get_data_loader(X_train, y_train, shuffle=False)
            print("TRAIN LOADER", next(iter(train_loader))[0].shape)
            
            
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
            
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "shape": feat_shape.squeeze().cpu().numpy(),
                    "input": feat_inputs.cpu().numpy().round(3),
                    "split": i
                })
            )
            
        return pd.concat(dfs)
    
    def get_calibration_data(self, X, y, eval_time, n_bins=20, method="quantile"):
        
        # First get cdf preds
        surv_preds = self.predict_survival(X, eval_time)
        cdf_preds = 1 - surv_preds
        
        
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