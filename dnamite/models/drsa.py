"""
SAMPLE BEGINNING OF FILE DOCSTRING
"""

import torch
import torch.nn as nn
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.model_selection import train_test_split
from dnamite.loss_fns import drsa_loss

class DRSASingleSplit(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, eval_times, device="cpu"):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.eval_times = eval_times
        
        self.rnn = nn.LSTM(n_input+1, n_hidden, batch_first=True)
        self.output_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        
    def forward(self, x):
        # x is shape (N, p)
        
        # Add eval times to x
        inputs = x.unsqueeze(1).repeat(1, self.n_output, 1)
        inputs = torch.cat([
            inputs,
            self.eval_times.unsqueeze(0).repeat(x.shape[0], 1).unsqueeze(-1)
        ], dim=2)
        
        h, _ = self.rnn(inputs)
        
        return self.output_head(h)
    
    def train_(
        self, 
        train_epoch_fn, 
        test_epoch_fn,
        train_loader,
        test_loader,
        optimizer,
        n_epochs,
    ):
        
        early_stopping_counter = 0
        best_test_loss = float('inf')

        for epoch in range(n_epochs):
            train_epoch_fn(self, train_loader, optimizer)
            test_loss, test_preds = test_epoch_fn(self, test_loader)
            
            # print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f} | Num Feats: {len([z for z in self.get_smooth_z() if z > 0])}")

            # Check if the test loss has improved
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stopping_counter = 0
                
                # Save the model at the best test loss
                torch.save(self.state_dict(), "../model_saves/tmp_best_model.pt")
                
            else:
                early_stopping_counter += 1

            # If test loss has not improved for 5 consecutive epochs, terminate training
            if early_stopping_counter >= 5:
                print(f"Early stopping at {epoch+1} epochs: Test loss has not improved for 5 consecutive epochs.")
                break
            
        # Load the model from the best test loss
        self.load_state_dict(torch.load("../model_saves/tmp_best_model.pt"))

        return


class DRSA(nn.Module):
    """  
    DRSA
    """
    
    def __init__(
        self, 
        n_input, 
        n_hidden, 
        n_output, 
        n_eval_times,
        validation_size=0.2,
        n_val_splits=5,
        learning_rate=1e-4,
        max_epochs=100,
        batch_size=128,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_eval_times = n_eval_times
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
            
        self.n_feats = X.shape[1]
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
    
    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, events, times in tqdm(train_loader, leave=False):

            X_main, events, times = X_main.to(self.device), events.to(self.device), times.to(self.device)

            y_pred = model(X_main)
            y_pred = torch.sigmoid(y_pred).squeeze(-1)

            loss = drsa_loss(y_pred, events, times, self.eval_times)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def test_epoch(self, model, test_loader):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, events, times in tqdm(test_loader, leave=False):
                X_main, events, times = X_main.to(self.device), events.to(self.device), times.to(self.device)

                y_pred = model(X_main)
        
                preds.append(y_pred.detach().squeeze(-1))
            
                y_pred = model(X_main)
                y_pred = torch.sigmoid(y_pred).squeeze(-1)
                
                loss = drsa_loss(y_pred, events, times, self.eval_times)
                
                # surv_probs = torch.cumprod(1 - y_pred, dim=1)

                # evt_loss = event_time_loss(y_pred)
                # evr_loss = event_rate_loss(y_pred)
                # loss = (alpha * evt_loss) + ((1 - alpha) * evr_loss)
                
                # loss = bce_surv_loss(surv_probs, events, times, eval_times)

                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def fit_one_split(self, X_train, y_train, X_val, y_val):
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            X_train = X_train[self.selected_feats]
            X_val = X_val[self.selected_feats]
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False)
        
        model = DRSASingleSplit(
            n_input=self.n_input,
            n_hidden=self.n_hidden,
            n_output=self.n_eval_times,
            eval_times=self.eval_times,
            device=self.device
        ).to(self.device)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train_(
            train_epoch_fn=self.train_epoch,
            test_epoch_fn=self.test_epoch,
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
        self.n_input = X.shape[1]
        
        # Get evaluation times before train/val split
        quantiles = torch.quantile(
            torch.FloatTensor(y["time"].copy()),
            torch.linspace(0, 1, self.n_eval_times+2)
        )

        self.eval_times = quantiles[1:-1].to(self.device)
        
        # Remove duplicates in eval times
        if len(self.eval_times.unique()) < len(self.eval_times):
            self.eval_times = self.eval_times.unique()
            self.n_eval_times = len(self.eval_times)
        
        
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
    
    def predict(self, X_test):
        
        # Preprocess features
        X_test = self.preprocess_data(X_test)
        
        # Make placeholder y_test
        y_test = np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        test_loader = self.get_data_loader(X_test, y_test, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0], len(self.eval_times)))
        for i, model in enumerate(self.models):
            _, model_preds = self.test_epoch(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def get_calibration_data(self, X, y, eval_time, n_bins=20, method="quantile"):
        
        # First get cdf preds
        preds = self.predict(X)
        hazard_preds = 1 / (1 + np.exp(-preds))
        cdf_preds = 1 - np.cumprod(1 - hazard_preds, axis=1) 
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