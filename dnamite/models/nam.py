"""
SAMPLE BEGINNING OF FILE DOCSTRING
"""

from dnamite.loss_fns import ipcw_rps_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator

class BaseSingleSplitNAM(nn.Module):
    def __init__(
        self, 
        n_features, 
        n_hidden, 
        n_output, 
        n_layers=2, 
        exu=False,
        gamma=1, 
        reg_param=0.0, 
        main_pair_strength=1.0, 
        entropy_param=0.0, 
        has_pairs=True, 
        n_pairs=None, 
        device="cpu", 
        strong_hierarchy=False, 
        pairs_list=None, 
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.exu = exu
        self.gamma = gamma
        self.reg_param = reg_param
        self.main_pair_strength = main_pair_strength
        self.entropy_param = entropy_param 
        self.has_pairs = has_pairs
        self.main_effects_frozen = False
        self.device=device
        self.strong_hierarchy=strong_hierarchy
        self.fit_pairs = False
        
        if self.reg_param == 0:
            self.penalized = False
        else:
            self.penalized = True
        
        # Initialize empty parameters based on n_layers
        self.main_weights = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, 1, n_hidden))
        ])
        self.main_weights.extend([
            nn.Parameter(torch.empty(n_features, n_hidden, n_hidden))
            for _ in range(n_layers - 1)
        ])
        self.main_weights.append(
            nn.Parameter(torch.empty(n_features, n_hidden, n_output))
        )
        
        self.main_biases = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, n_hidden))
        ])
        self.main_biases.extend([
            nn.Parameter(torch.empty(n_features, n_hidden))
            for _ in range(n_layers - 1)
        ])
        self.main_biases.append(
            nn.Parameter(torch.empty(n_features, n_output))
        )
        
        if exu:
            self.main_biases[0] = nn.Parameter(torch.empty(n_features, 1))
        
        self.main_activations = [
            F.relu for _ in range(n_layers)
        ] + [nn.Identity()]
        
        self.z_main = nn.Parameter(torch.empty(n_features))
        
        self.active_feats = torch.arange(n_features).to(device)
        
        if has_pairs:
            if n_pairs is None:
                n_pairs = int((n_features * (n_features - 1)) / 2)
                self.pairs_list = list(combinations(range(n_features), 2))
            else:
                if self.strong_hierarchy:
                    assert pairs_list is not None, \
                        "Pairs list is None, but n_pairs is provided, meaning not all pairs are given"
                
                self.pairs_list = pairs_list
                
            self.init_pairs_params(n_pairs)
            
            self.active_pairs = torch.arange(n_pairs).to(device)
            
        self.reset_parameters()
        
    def init_pairs_params(self, n_pairs):
        
        # Set number of pairs to be |n_features choose 2|
        self.n_pairs = n_pairs

        self.pair_weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_pairs, 2, self.n_hidden))
        ])
        self.pair_weights.extend([
            nn.Parameter(torch.empty(self.n_pairs, self.n_hidden, self.n_hidden))
            for _ in range(self.n_layers - 1)
        ])
        self.pair_weights.append(
            nn.Parameter(torch.empty(self.n_pairs, self.n_hidden, self.n_output))
        )
        
        self.pair_biases = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_pairs, self.n_hidden))
        ])
        self.pair_biases.extend([
            nn.Parameter(torch.empty(self.n_pairs, self.n_hidden))
            for _ in range(self.n_layers - 1)
        ])
        self.pair_biases.append(
            nn.Parameter(torch.empty(self.n_pairs, self.n_output))
        )
        
        self.pair_activations = [
            F.relu for _ in range(self.n_layers)
        ] + [nn.Identity()]

        self.z_pairs = nn.Parameter(torch.empty(self.n_pairs))
        
        self.reset_pairs_parameters()
        
        
    def reset_parameters(self):
        for w, b in zip(self.main_weights, self.main_biases):
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)
        
        nn.init.uniform_(self.z_main, -self.gamma/100, self.gamma/100)
        
        if self.has_pairs:
            self.reset_pairs_parameters()
            
            
    def reset_pairs_parameters(self):
        for w, b in zip(self.pair_weights, self.pair_biases):
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)

        nn.init.uniform_(self.z_pairs, -self.gamma/100, self.gamma/100)
    
        
    def get_smooth_z(self):
        condition_1_main = self.z_main <= -self.gamma/2
        condition_2_main = self.z_main >= self.gamma/2
        
        smooth_zs_main = (-2 /(self.gamma**3)) * (self.z_main**3) + (3/(2 * self.gamma)) * self.z_main + 0.5
        z_main = torch.where(condition_1_main, torch.zeros_like(self.z_main), 
                              torch.where(condition_2_main, torch.ones_like(self.z_main), smooth_zs_main))  
        return z_main
    
    def get_smooth_z_pairs(self):
        condition_1_pairs = self.z_pairs <= -self.gamma/2
        condition_2_pairs = self.z_pairs >= self.gamma/2
        
        smooth_zs_pairs = (-2 /(self.gamma**3)) * (self.z_pairs**3) + (3/(2 * self.gamma)) * self.z_pairs + 0.5
        z_pairs = torch.where(condition_1_pairs, torch.zeros_like(self.z_pairs), 
                              torch.where(condition_2_pairs, torch.ones_like(self.z_pairs), smooth_zs_pairs))
        
        if self.strong_hierarchy:
        
            z_mains = self.get_smooth_z()
            z_pairs = z_pairs * z_mains[torch.tensor(self.pairs_list)].prod(dim=1)

        return z_pairs

    def forward(self, mains=None, pairs=None):
        # mains of shape (batch_size, n_features)
        # pairs of shape (batch_size, |n_features choose 2|, 2)
        
        output_main = 0
        output_pairs = 0
        
        if mains is not None:
            
            mains = mains[:, self.active_feats].unsqueeze(-1)

            # Only apply ExU to first layer
            # following code from official implementation
            # https://github.com/AmrMKayid/nam/blob/fc2da75ba008c4ef02a83747f2116036fa6fec46/nam/models/featurenn.py#L41
            if self.exu:
                mains = torch.einsum(
                    'ijk,jkl->ijl', 
                    mains - self.main_biases[0][self.active_feats, :].reshape(1, -1, 1),
                    torch.exp(self.main_weights[0][self.active_feats, :])
                )
                mains = torch.clip(mains, 0, 1)
                
                for w, b, a in zip(self.main_weights[1:], self.main_biases[1:], self.main_activations[1:]):
                    mains = torch.einsum(
                        'ijk,jkl->ijl', 
                        mains, 
                        w[self.active_feats, :, :]
                    ) + b[self.active_feats, :]
                    mains = a(mains)
            
            else:

                for w, b, a in zip(self.main_weights, self.main_biases, self.main_activations):
                    mains = torch.einsum(
                        'ijk,jkl->ijl', 
                        mains, 
                        w[self.active_feats, :, :]
                    ) + b[self.active_feats, :]
                    mains = a(mains)

            # Get smoothed z
            z_main = self.get_smooth_z()[self.active_feats]

            output_main = torch.einsum('ijk,j->ik', mains, z_main)
        
        if pairs is not None:
            
            pairs = pairs[:, self.active_pairs, :]
            
            for w, b, a in zip(self.pair_weights, self.pair_biases, self.pair_activations):
                pairs = torch.einsum(
                    'ijk,jkl->ijl', 
                    pairs, 
                    w[self.active_pairs, :, :]
                ) + b[self.active_pairs, :]
                pairs = a(pairs)
            
            z_pairs = self.get_smooth_z_pairs()[self.active_pairs]
            
            output_pairs = torch.einsum('ijk,j->ik', pairs, z_pairs)
        
        
        return output_main + output_pairs
    
    def forward_per_feature(self, mains=None, pairs=None):
        
        output_main = 0
        output_pairs = 0
        
        if mains is not None:
            
            mains = mains[:, self.active_feats].unsqueeze(-1)
                     
            # Only apply ExU to first layer
            # following code from official implementation
            # https://github.com/AmrMKayid/nam/blob/fc2da75ba008c4ef02a83747f2116036fa6fec46/nam/models/featurenn.py#L41
            if self.exu:
                mains = torch.einsum(
                    'ijk,jkl->ijl', 
                    mains - self.main_biases[0].reshape(1, -1, 1),
                    torch.exp(self.main_weights[0])
                )
                mains = torch.clip(mains, 0, 1)
                
                for w, b, a in zip(self.main_weights[1:], self.main_biases[1:], self.main_activations[1:]):
                    mains = torch.einsum(
                        'ijk,jkl->ijl', 
                        mains, 
                        w[self.active_feats, :, :]
                    ) + b[self.active_feats, :]
                    mains = a(mains)
            
            else:

                for w, b, a in zip(self.main_weights, self.main_biases, self.main_activations):
                    mains = torch.einsum(
                        'ijk,jkl->ijl', 
                        mains, 
                        w[self.active_feats, :, :]
                    ) + b[self.active_feats, :]
                    mains = a(mains)

            # Get smoothed z
            z_main = self.get_smooth_z()
            
            # Note difference: don't sum over j to keep each feature separate
            output_main = torch.einsum('ijk,j->ijk', mains, z_main)
        
        if pairs is not None:
            
            pairs = pairs[:, self.active_pairs, :]
            
            for w, b, a in zip(self.pair_weights, self.pair_biases, self.pair_activations):
                pairs = torch.einsum(
                    'ijk,jkl->ijl', 
                    pairs, 
                    w[self.active_pairs, :, :]
                ) + b[self.active_pairs, :]
                pairs = a(pairs)
            
            z_pairs = self.get_smooth_z_pairs()
            
            # Note difference: don't sum over j to keep each feature separate
            output_pairs = torch.einsum('ijk,j->ijk', pairs, z_pairs)
        
        
        return output_main, output_pairs
        
    
    
    def loss_penalty(self):
        z_main = self.get_smooth_z()[self.active_feats]
        
        if self.has_pairs:
            
            z_pairs = self.get_smooth_z_pairs()[self.active_pairs]
            
            if self.main_effects_frozen:
                return self.reg_param * self.main_pair_strength * z_pairs.sum() + self.get_entropy_reg()
            else:
                return self.reg_param \
                    * (z_main.sum() + self.main_pair_strength * z_pairs.sum()) \
                    + self.get_entropy_reg()
        
        else:
            
            return self.reg_param * z_main.sum() + self.get_entropy_reg()
        
    def get_entropy_reg(self):
        z_main = self.get_smooth_z()[self.active_feats]
        
        # Control consider the smooth zs which are not already 0 or 1
        z_main_active = z_main[
            (z_main > 0) & (z_main < 1)
        ]
        
        omegas_main = -(z_main_active * torch.log(z_main_active) + \
                        (1 - z_main_active) * torch.log(1 - z_main_active))
        
        if self.has_pairs:
            z_pairs = self.get_smooth_z_pairs()[self.active_pairs]

            z_pairs_active = z_pairs[
                (z_pairs > 0) & (z_pairs < 1)
            ]

            omegas_pairs = -(z_pairs_active * torch.log(z_pairs_active) + \
                                (1 - z_pairs_active) * torch.log(1 - z_pairs_active))

            if self.main_effects_frozen:
                return self.entropy_param * self.main_pair_strength * omegas_pairs.sum()
            
            else:
                return self.entropy_param * (omegas_main.sum() + self.main_pair_strength * omegas_pairs.sum())
        
        else:
        
            return self.entropy_param * omegas_main.sum()
        
    def freeze_main_effects(self):
        self.main_effects_frozen = True
        for w, b in zip(self.main_weights, self.main_biases):
            w.requires_grad = False
            b.requires_grad = False
        
        self.z_main.requires_grad = False
        
        
        return
    
    def get_shape_function(self, feat_index, feat_min, feat_max, center=True):
        inputs = torch.linspace(feat_min, feat_max, 1000).to(self.device)
        
        self.eval()
        with torch.no_grad():
            mains = inputs.unsqueeze(-1)
            if self.exu:
                mains = torch.einsum(
                    'ik,kl->il', 
                    mains - self.main_biases[0][feat_index, :].reshape(1, -1),
                    torch.exp(self.main_weights[0][feat_index, :, :])
                )
                mains = torch.clip(mains, 0, 1)
                
                for w, b, a in zip(self.main_weights[1:], self.main_biases[1:], self.main_activations[1:]):
                    mains = torch.einsum(
                        'ik,kl->il', 
                        mains, 
                        w[feat_index, :, :]
                    ) + b[feat_index, :]
                    mains = a(mains)
            
            else:

                for w, b, a in zip(self.main_weights, self.main_biases, self.main_activations):
                    mains = torch.einsum(
                        'ik,kl->il', 
                        mains, 
                        w[feat_index, :, :]
                    ) + b[feat_index, :]
                    mains = a(mains)


            # Get smoothed z
            z_main = self.get_smooth_z()
            
            prediction = mains * z_main[feat_index]
        
            if hasattr(self, 'feat_offsets') and center:
                return prediction - self.feat_offsets[feat_index, :], inputs
            elif center:
                print("Intercept not computed. Shape function will not be centered.")
                return prediction, inputs
            else:
                return prediction, inputs
    
    def get_interaction_function(
        self, pair_index, feat1_min, feat1_max, feat2_min, feat2_max, center=True
    ):
        
        inputs_feat1 = torch.linspace(feat1_min, feat1_max, 100)
        inputs_feat2 = torch.linspace(feat2_min, feat2_max, 100)
        
        # Create 2D grids for x and y
        grid_feat1, grid_feat2 = torch.meshgrid(inputs_feat1, inputs_feat2, indexing='ij')

        # Reshape the grids to 1D tensors to get pairwise combinations
        grid_feat1.reshape(-1)
        grid_feat2.reshape(-1)
        
        inputs = torch.stack([grid_feat1.reshape(-1), grid_feat2.reshape(-1)], dim=1).to(self.device)
        
        pairs = inputs
        for w, b, a in zip(self.pair_weights, self.pair_biases, self.pair_activations):
            pairs = torch.einsum(
                'ik,kl->il', 
                pairs, 
                w[pair_index, :, :]
            ) + b[pair_index, :]
            pairs = a(pairs)

        # Get smoothed z
        z_pairs = self.get_smooth_z_pairs()

        # output_pairs = torch.einsum('ijk,j->ik', output_pairs, z_pairs)
        
        # return pairs.squeeze() * z_pairs[pair_index], inputs
        
        prediction = pairs * z_pairs[pair_index]
        
        if hasattr(self, 'pair_offsets') and center:
            return prediction - self.pair_offsets[pair_index, :], inputs
        elif center:
            print("Intercept not computed. Shape function will not be centered.")
            return prediction, inputs
        else:
            return prediction, inputs
        
        
    def prune_parameters(self, mains=True, pairs=False):
        
        if mains:
        
            z_main = self.get_smooth_z()
            self.active_feats = torch.where(z_main > 0)[0]
        
        if pairs:
            assert self.has_pairs, "Tried to prune pairs parameters without model having pairs"
            
            z_pairs = self.get_smooth_z_pairs()
            self.active_pairs = torch.where(z_pairs > 0)[0]
            
    def compute_intercept(self, train_loader):
        
        self.feat_offsets = torch.zeros(self.n_features, self.n_output).to(self.device)
        n_samples = 0
        for batch in train_loader:
            mains = batch[0].to(self.device)
            
            main_preds, _ = self.forward_per_feature(mains)
            self.feat_offsets += main_preds.sum(dim=0)
            n_samples += mains.shape[0]
        
        self.feat_offsets = self.feat_offsets / n_samples
        self.intercept = self.feat_offsets.sum(dim=0)
            
        return
    
    def compute_pair_intercept(self, train_loader):
        
        self.pair_offsets = torch.zeros(len(self.pairs_list), self.n_output).to(self.device)
        n_samples = 0
        for batch in train_loader:
            pairs = batch[1].to(self.device)
            
            _, pair_preds = self.forward_per_feature(pairs=pairs)
            self.pair_offsets += pair_preds.sum(dim=0)
            n_samples += pairs.shape[0]
        
        self.pair_offsets = self.pair_offsets / n_samples
        self.pair_intercept = self.pair_offsets.sum(dim=0)
            
        return
    
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
            
            # print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f}")
            
            # Do feature pruning
            if self.fit_pairs:
                self.prune_parameters(mains=True, pairs=True)
            else:
                self.prune_parameters(mains=True)

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
    
class BaseNAM(nn.Module):
    
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
    
    def fit_one_split(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError("fit_one_split is only implmented for child NAM classes.")
    
    def fit(self, X, y):
        
        self.feature_names_in_ = X.columns
        
        # Preprocess features
        X = self.preprocess_data(X)
        
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
        
        X_test_interactions = X_test.values[:, self.selected_pair_indices]
        
        # Make placeholder y_test
        y_test = np.zeros(X_test.shape[0])
        
        test_loader = self.get_data_loader(X_test, y_test, pairs=X_test_interactions, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0]))
        for i, model in enumerate(self.models):
            _, model_preds = self.test_epoch_pairs(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
class NAMRegressor(BaseNAM):
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
        super().__init__(
            n_features=n_features,
            n_hidden=n_hidden,
            n_output=1,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            **kwargs    
        )
            
        
    def get_data_loader(self, X, y, pairs=None, shuffle=True):
        
        # Convert X to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        if pairs is not None:
            if hasattr(pairs, 'values'):
                pairs = pairs.values

        if pairs is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(pairs),
                torch.FloatTensor(y),
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(y),
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch_mains(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, labels in tqdm(train_loader, leave=False):

            X_main, labels = X_main.to(self.device), labels.to(self.device)

            y_pred = model(mains=X_main).squeeze(-1)
            
            loss = F.mse_loss(y_pred, labels)
            
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
            for X_main, labels in tqdm(test_loader, leave=False):

                X_main, labels = X_main.to(self.device), labels.to(self.device)

                y_pred = model(mains=X_main).squeeze(-1)
        
                preds.append(y_pred.detach())
            
                loss = F.mse_loss(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty()
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def train_epoch_pairs(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, X_pairs, labels in tqdm(train_loader, leave=False):

            X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                main_preds = model(mains=X_main).squeeze(-1)
            
            pair_preds = model(mains=None, pairs=X_pairs).squeeze(-1)
            
            y_pred = main_preds + pair_preds
            
            loss = F.mse_loss(y_pred, labels)
            
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
            for X_main, X_pairs, labels in tqdm(test_loader, leave=False):

                X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

                main_preds = model(mains=X_main).squeeze(-1)
                pair_preds = model(pairs=X_pairs).squeeze(-1)
                
                y_pred = main_preds + pair_preds
        
                preds.append(y_pred.detach())
            
                loss = F.mse_loss(y_pred, labels)
                
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
            n_output=1,
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
            lr=self.learning_rate / 5
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
            lr=self.learning_rate / 5
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
    
    def get_shape_function(self, feature_name, X, feat_min=None, feat_max=None):
        
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
                feat_mean = scaler.mean_[feat_index]
                feat_std = scaler.scale_[feat_index]
                input_min = (feat_min - feat_mean) / feat_std
                input_max = (feat_max - feat_mean) / feat_std
            
            feat_shape, feat_inputs = model.get_shape_function(feat_index, input_min, input_max, center=True)
            
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "shape": feat_shape.squeeze().cpu().numpy(),
                    "input": feat_inputs.cpu().numpy().round(3),
                    "split": i
                })
            )
            
        return pd.concat(dfs)
    

class NAMSurvival(nn.Module):
    """
    # Survival analysis NAM
    # Using the same structure as DNAMite
    # but without discretization/embedding
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

        pcw_obs_times = self.cde.predict_proba(y["time"]) + 1e-5

        if pairs is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(pairs),
                torch.BoolTensor(y["event"]),
                torch.FloatTensor(y["time"].copy()),
                torch.FloatTensor(pcw_obs_times)
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.BoolTensor(y["event"]),
                torch.FloatTensor(y["time"].copy()),
                torch.FloatTensor(pcw_obs_times)
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch_mains(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, events, times, pcw_obs_times in tqdm(train_loader, leave=False):

            X_main, events, times, pcw_obs_times = X_main.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

            y_pred = model(mains=X_main)
            
            cdf_preds = torch.sigmoid(y_pred)
            
            loss = ipcw_rps_loss(
                cdf_preds, 
                self.pcw_eval_times,
                pcw_obs_times, 
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
            for X_main, events, times, pcw_obs_times in tqdm(test_loader, leave=False):

                X_main, events, times, pcw_obs_times = X_main.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

                y_pred = model(mains=X_main)
        
                preds.append(y_pred.detach())
            
                cdf_preds = torch.sigmoid(y_pred)
            
                loss = ipcw_rps_loss(
                    cdf_preds, 
                    self.pcw_eval_times,
                    pcw_obs_times, 
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

        for X_main, X_pairs, events, times, pcw_obs_times in tqdm(train_loader, leave=False):

            X_main, X_pairs, events, times, pcw_obs_times = \
                X_main.to(self.device), X_pairs.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

            with torch.no_grad():
                main_preds = model(mains=X_main)
            
            pair_preds = model(mains=None, pairs=X_pairs)
            
            y_pred = main_preds + pair_preds
            
            cdf_preds = torch.sigmoid(y_pred)
            
            loss = ipcw_rps_loss(
                cdf_preds, 
                self.pcw_eval_times,
                pcw_obs_times, 
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
            for X_main, X_pairs, events, times, pcw_obs_times in tqdm(test_loader, leave=False):

                X_main, X_pairs, events, times, pcw_obs_times = \
                    X_main.to(self.device), X_pairs.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

                main_preds = model(mains=X_main)
                pair_preds = model(pairs=X_pairs)
                
                y_pred = main_preds + pair_preds
        
                preds.append(y_pred.detach())
            
                cdf_preds = torch.sigmoid(y_pred)
                
                loss = ipcw_rps_loss(
                    cdf_preds, 
                    self.pcw_eval_times,
                    pcw_obs_times, 
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

        self.cde = CensoringDistributionEstimator()
        self.cde.fit(y)

        # Get evaluation times before train/val split
        quantiles = torch.quantile(
            torch.FloatTensor(y["time"].copy()),
            torch.linspace(0, 1, self.n_output+2)
        )
        
        self.eval_times = quantiles[1:-1].to(self.device)
        self.pcw_eval_times = torch.FloatTensor(self.cde.predict_proba(self.eval_times.cpu().numpy())).to(self.device) + 1e-5
        
        self.feature_names_in_ = X.columns
        
        # Preprocess features
        X = self.preprocess_data(X)
        
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
            print("LOSS IN PREDICT", test_loss)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def get_shape_function(self, feature_name, X, eval_time, feat_min=None, feat_max=None):
        
        # placeholder y
        y = np.zeros(X.shape[0], dtype=[('time', 'float32'), ('event', 'bool')])
        
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
                feat_mean = scaler.mean_[feat_index]
                feat_std = scaler.scale_[feat_index]
                input_min = (feat_min - feat_mean) / feat_std
                input_max = (feat_max - feat_mean) / feat_std
            
            feat_shape, feat_inputs = model.get_shape_function(feat_index, input_min, input_max, center=True)
            
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
            # print("BIN", bin_, "MEAN", bin_data.mean(), "MEDIAN", np.median(bin_data))
            # predicted.append(bin_data.mean())
            predicted.append(np.mean(bin_data))
            observed.append(
                1 - surv_prob[np.clip(
                    np.searchsorted(times, eval_time), 0, len(times)-1
                )]
            )
            
            
        return np.array(predicted), np.array(observed), quantiles