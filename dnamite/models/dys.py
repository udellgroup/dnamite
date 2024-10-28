"""
SAMPLE BEGINNING OF FILE DOCSTRING
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from .nam import BaseSingleSplitNAM
from sksurv.nonparametric import kaplan_meier_estimator
from dnamite.loss_fns import rps_loss


class DyS(nn.Module):
    """  
    DyS
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
    
    def get_data_loader(self, X, y, pairs=None, shuffle=True):
        
        assert "time" in y.dtype.names and "event" in y.dtype.names, \
            "y must be a structured array with 'time' and 'event' fields."
        
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
                torch.BoolTensor(y["event"]),
                torch.FloatTensor(y["time"].copy()),
            )
        else:
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

            y_pred = model(mains=X_main)
            
            pmf_preds = torch.softmax(y_pred, dim=-1)
            surv_preds = 1 - torch.cumsum(pmf_preds, dim=-1)
            
            loss = rps_loss(
                surv_preds, 
                events,
                times,
                self.eval_times
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

                y_pred = model(mains=X_main)
        
                preds.append(y_pred.detach())
            
                pmf_preds = torch.softmax(y_pred, dim=-1)
                surv_preds = 1 - torch.cumsum(pmf_preds, dim=-1)
                
                loss = rps_loss(
                    surv_preds, 
                    events,
                    times,
                    self.eval_times
                )
                
                if model.penalized:
                    loss += model.loss_penalty()
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def train_epoch_pairs(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, X_pairs, events, times in tqdm(train_loader, leave=False):

            X_main, X_pairs, events, times = \
                X_main.to(self.device), X_pairs.to(self.device), events.to(self.device), times.to(self.device)

            with torch.no_grad():
                main_preds = model(mains=X_main)
            
            pair_preds = model(mains=None, pairs=X_pairs)
            
            y_pred = main_preds + pair_preds
            
            pmf_preds = torch.softmax(y_pred, dim=-1)
            surv_preds = 1 - torch.cumsum(pmf_preds, dim=-1)
            
            loss = rps_loss(
                surv_preds, 
                events,
                times,
                self.eval_times
            )
                
            
            if model.penalized:
                loss += model.loss_penalty()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def test_epoch_pairs(self, model, test_loader):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, X_pairs, events, times in tqdm(test_loader, leave=False):

                X_main, X_pairs, events, times = \
                    X_main.to(self.device), X_pairs.to(self.device), events.to(self.device), times.to(self.device)

                main_preds = model(mains=X_main)
                pair_preds = model(pairs=X_pairs)
                
                y_pred = main_preds + pair_preds
        
                preds.append(y_pred.detach())
            
                pmf_preds = torch.softmax(y_pred, dim=-1)
                surv_preds = 1 - torch.cumsum(pmf_preds, dim=-1)
                
                loss = rps_loss(
                    surv_preds, 
                    events,
                    times,
                    self.eval_times
                )
                
                if model.penalized:
                    loss += model.loss_penalty()
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def select_features(self, X, y):
        # Convert X to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Make one train/val split for feature selection
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size)
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False)
        
        model = BaseSingleSplitNAM(
            n_features=self.n_features, 
            n_hidden=self.n_hidden, 
            n_output=self.n_output,
            device=self.device,
            **self.model_args
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        model.train_(
            self.train_epoch_mains, 
            self.test_epoch_mains, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs
        )

        active_feats = model.active_feats.cpu().numpy()
        active_pairs = list(combinations(active_feats, 2))

        X_train_interactions = X_train[:, active_pairs]
        X_val_interactions = X_val[:, active_pairs]


        model.freeze_main_effects()
        model.fit_pairs = True
        model.pair_kernel_size = self.pair_kernel_size
        model.pair_kernel_weight = self.pair_kernel_weight
        model.pairs_list = torch.IntTensor(active_pairs).to(self.device)
        model.n_pairs = len(active_pairs)
        model.init_pairs_params(model.n_pairs)
        model.active_pairs = torch.arange(model.n_pairs).to(self.device)
        model.main_pair_strength = self.main_pair_strength
        model.to(self.device)
        
        train_loader = self.get_data_loader(X_train, y_train, pairs=X_train_interactions)
        val_loader = self.get_data_loader(X_val, y_val, pairs=X_val_interactions, shuffle=False)
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=self.learning_rate / 10
        )
        
        model.train_(
            self.train_epoch_pairs, 
            self.test_epoch_pairs, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs
        )
        
        self.selected_feats = self.feature_names_in_[model.active_feats.cpu().numpy()].tolist()
        self.selected_pairs = [
            [self.feature_names_in_[pair[0]], self.feature_names_in_[pair[1]]]
            for pair in model.pairs_list[model.active_pairs].cpu().numpy()
        ]
        
        return
    
    def fit_one_split(self, X_train, y_train, X_val, y_val):
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            X_train = X_train[self.selected_feats]
            X_val = X_val[self.selected_feats]
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False)
        
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
        
        if model.reg_param > 0:
            self.selected_pair_indices = [
                [self.selected_feats.index(feat) for feat in pair] for pair in self.selected_pairs
            ]
        else:
            self.selected_pairs = list(combinations(self.feature_names_in_, 2))
            self.selected_pair_indices = list(combinations(range(self.n_features), 2))
        
        model.freeze_main_effects()
        model.fit_pairs = True
        model.pairs_list = torch.IntTensor(self.selected_pair_indices).to(self.device)
        model.n_pairs = len(self.selected_pairs)
        model.init_pairs_params(model.n_pairs)
        model.active_pairs = torch.arange(model.n_pairs).to(self.device)
        model.to(self.device)
        
        X_train_interactions = X_train.values[:, self.selected_pair_indices]
        X_val_interactions = X_val.values[:, self.selected_pair_indices]
        
        train_loader = self.get_data_loader(X_train, y_train, pairs=X_train_interactions)
        val_loader = self.get_data_loader(X_val, y_val, pairs=X_val_interactions, shuffle=False)
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=self.learning_rate / 10
        )
        
        model.train_(
            self.train_epoch_pairs, 
            self.test_epoch_pairs, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs
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
            
            # Fit to this split
            model = self.fit_one_split(X_train, y_train, X_val, y_val)
            model.feature_names_in_ = X_train.columns   
            
            self.models.append(model)
            
        return
    
    def predict(self, X_test, y_test=None):
        
        # Preprocess features
        X_test = self.preprocess_data(X_test)
        
        X_test_interactions = X_test.values[:, self.selected_pair_indices]
        
        # Make placeholder y_test
        if y_test is None:
            y_test = np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        test_loader = self.get_data_loader(X_test, y_test, pairs=X_test_interactions, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0], len(self.eval_times)))
        for i, model in enumerate(self.models):
            test_loss, model_preds = self.test_epoch_pairs(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def predict_survival(self, X_test):
        
        # Preprocess features
        X_test = self.preprocess_data(X_test)
        
        X_test_interactions = X_test.values[:, self.selected_pair_indices]
        
        # Make placeholder y_test
        y_test = np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        test_loader = self.get_data_loader(X_test, y_test, pairs=X_test_interactions, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0], len(self.eval_times)))
        for i, model in enumerate(self.models):
            _, model_preds = self.test_epoch_pairs(model, test_loader)
            
            pmf_preds = torch.softmax(model_preds, dim=-1)
            surv_preds = 1 - torch.cumsum(pmf_preds, dim=-1)
            
            test_preds[i, ...] = surv_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def get_shape_function(self, feature_name, X, eval_time, feat_min=None, feat_max=None):
        
        # placeholder y
        y = np.zeros(X.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
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
        preds = self.predict(X)
        pmf_preds = np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True)
        cdf_preds = np.cumsum(pmf_preds, axis=-1)
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