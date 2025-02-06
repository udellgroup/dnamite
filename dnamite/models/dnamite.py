"""
SAMPLE BEGINNING OF FILE DOCSTRING
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm
import os
from functools import partial
from dnamite.utils import discretize, get_bin_counts, get_pair_bin_counts
from dnamite.loss_fns import ipcw_rps_loss
import textwrap
from contextlib import nullcontext
import copy
from dnamite.utils import LoggingMixin

# Class for a DNAMite model fit to one train/validation split
# Full DNAMite model averages several of these single-split models
# n_features: number of features in the data
# n_embed: embedding dimension
# n_hidden: hidden layer dimension
# n_output: output dimension
# feature_sizes: list of number of bins for each feature
# categorical_feats: list of indices of categorical features
# n_layers: number of hidden layers
# gamma: parameter for smooth-step function
# reg_param: regularization parameter for feature sparsity
# pair_reg_param: regularization parameter for pair sparsity
# entropy_param: regularization parameter for entropy of smoothed binary gates
# fit_pairs: whether to fit pair interactions
# device: device to run the model on
# pairs_list: list of pairs of indices of features to fit interactions for
# kernel_size: window size for kernel smoothing
# kernel_weight: weight for kernel smoothing
# pair_kernel_size: window size for kernel smoothing for pairs
# pair_kernel_weight: weight for kernel smoothing for pairs
class _BaseSingleSplitDNAMiteModel(nn.Module, LoggingMixin):
    """
    _BaseSingleSplitDNAMiteModel is a private base class for DNAMite models trained on a single train/val split.

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_output : int
        The number of output features.
    feature_sizes : list of int
        The sizes of the features.
    n_embed : int, optional (default=32)
        The size of the embedding layer.
    n_hidden : int, optional (default=32)
        The number of hidden units in the hidden layers.
    n_layers : int, optional (default=2)
        The number of hidden layers.
    gamma : float, optional (default=1)
        The gamma parameter use in the smooth-step function. 
    reg_param : float, optional (default=0)
        The regularization parameter for feature selection.
        Setting to 0 indicates no feature selection. 
        Larger values indicate stronger feature selection, i.e. less features.
    pair_reg_param : float, optional (default=0)
        The regularization parameter for pairs.
        Setting to 0 indicates no pair selection.
        Larger values indicate stronger pair selection, i.e. less pairs.
    entropy_param : float, optional (default=0)
        The entropy parameter for feature selection.
        Setting to 0 indicates no entropy regularization.
        Larger values indicate stronger entropy regularization.
    fit_pairs : bool, optional (default=True)
        Whether to fit pairs.
    device : str, optional (default="cpu")
        The device to run the model on ("cpu" or "cuda").
    pairs_list : list, optional (default=None)
        The list of pairs for the model.
    kernel_size : int, optional (default=5)
        The size of the kernel for smoothing.
    kernel_weight : float, optional (default=3)
        The weight for kernel smoothing.
    pair_kernel_size : int, optional (default=3)
        The size of the kernel for smoothing pairs.
    pair_kernel_weight : float, optional (default=3)
        The weight for kernel smoothing for pairs.
    cat_feat_mask : list or None, optional (default=None)
        The mask for categorical features.
    monotone_constraints : list or None, optional (default=None)
        The monotonic constraints for the features.
        0 indicates no constraint, 1 indicates increasing, -1 indicates decreasing.
        None means no constraints.
    verbosity : int, optional (default=0)
        The verbosity level of the model.
        0: warning, 1: info, 2: debug.
    """
    def __init__(self, 
                 n_features, 
                 n_output,
                 feature_sizes, 
                 n_embed=32,
                 n_hidden=32, 
                 n_layers=2,
                 gamma=1, 
                 pair_gamma=None,
                 reg_param=0, 
                 pair_reg_param=0, 
                 entropy_param=0, 
                 fit_pairs=True, 
                 device="cpu", 
                 pairs_list=None, 
                 kernel_size=5,
                 kernel_weight=3,
                 pair_kernel_size=3,
                 pair_kernel_weight=3,
                 cat_feat_mask=None,
                 monotone_constraints=None,
                 verbosity=0
                 ):
        nn.Module.__init__(self)
        LoggingMixin.__init__(self, verbosity=verbosity)
        self.n_features = n_features
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.feature_sizes = torch.tensor(feature_sizes).to(device)
        self.n_layers = n_layers
        self.gamma = gamma
        self.pair_gamma = pair_gamma if pair_gamma is not None else gamma
        self.reg_param = reg_param
        self.pair_reg_param = pair_reg_param
        self.entropy_param = entropy_param 
        self.fit_pairs = fit_pairs
        self.main_effects_frozen = False
        self.device=device
        self.kernel_size = kernel_size
        self.kernel_weight = kernel_weight
        self.pair_kernel_size = pair_kernel_size
        self.pair_kernel_weight = pair_kernel_weight
        self.pairs_list = pairs_list
        if pairs_list is not None:
            self.selected_pair_indices = pairs_list # list version
            self.pairs_list = torch.LongTensor(pairs_list).to(device) # tensor version
            self.n_pairs = len(pairs_list)
        else:
            self.selected_pair_indices = list(combinations(range(n_features), 2))
            self.pairs_list = torch.LongTensor(self.selected_pair_indices).to(device)
            self.n_pairs = len(self.selected_pair_indices)
            
        self.init_pairs_params(self.n_pairs)
            
        if cat_feat_mask is not None:
            self.cat_feat_mask = cat_feat_mask.to(device)
        else:
            self.cat_feat_mask = torch.zeros(n_features).astype(bool).to(device)
            
        self.monotone_constraints = monotone_constraints
        if monotone_constraints is not None:
            self.monotone_constraints_tensor = torch.tensor(monotone_constraints, device=device).view(1, n_features, 1)
            self.monotone_params = nn.Parameter((torch.rand(n_features, n_output) * 0.2) - 0.1)
        
        
        if self.reg_param == 0:
            self.penalized = False
        else:
            self.penalized = True
            
        # Initialize empty parameters based on n_layers
        self.main_weights = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, n_embed, n_hidden))
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
        
        self.main_activations = [
            F.relu for _ in range(n_layers)
        ] + [nn.Identity()]
        
        
        self.z_main = nn.Parameter(torch.empty(n_features))
        
        self.active_feats = torch.arange(n_features).to(device)
            
        self.reset_parameters()


        self.embedding_offsets = torch.cat([
            torch.tensor([0]), 
            torch.cumsum(torch.tensor(feature_sizes), dim=0)[:-1]
        ]).to(device)
        self.total_feature_size = sum(feature_sizes)
        self.embedding = nn.Embedding(self.total_feature_size, n_embed)
        
        # Set the pair embedding to be the same as the main embedding
        # Later, if the main effects are frozen, the pair embedding will be copied
        self.pair_embedding = self.embedding
            
            
        
    def init_pairs_params(self, n_pairs):
        
        self.n_pairs = n_pairs
        

        self.pair_weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_pairs, 2 * self.n_embed, self.n_hidden))
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
        
        self.active_pairs = torch.arange(self.n_pairs).to(self.device)
        
        self.reset_pairs_parameters()
        
        
    def reset_parameters(self):
        for w, b in zip(self.main_weights, self.main_biases):
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)
        
        nn.init.uniform_(self.z_main, -self.gamma/100, self.gamma/100)
            
            
    def reset_pairs_parameters(self):
        for w, b in zip(self.pair_weights, self.pair_biases):
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)

        nn.init.uniform_(self.z_pairs, -self.pair_gamma/100, self.pair_gamma/100)
    
        
    def get_smooth_z(self):
        condition_1_main = self.z_main <= -self.gamma/2
        condition_2_main = self.z_main >= self.gamma/2
        
        smooth_zs_main = (-2 /(self.gamma**3)) * (self.z_main**3) + (3/(2 * self.gamma)) * self.z_main + 0.5
        z_main = torch.where(condition_1_main, torch.zeros_like(self.z_main), 
                              torch.where(condition_2_main, torch.ones_like(self.z_main), smooth_zs_main))  
        return z_main
    
    def get_smooth_z_pairs(self):
        condition_1_pairs = self.z_pairs <= -self.pair_gamma/2
        condition_2_pairs = self.z_pairs >= self.pair_gamma/2
        
        smooth_zs_pairs = (-2 /(self.gamma**3)) * (self.z_pairs**3) + (3/(2 * self.gamma)) * self.z_pairs + 0.5
        z_pairs = torch.where(condition_1_pairs, torch.zeros_like(self.z_pairs), 
                              torch.where(condition_2_pairs, torch.ones_like(self.z_pairs), smooth_zs_pairs))

        return z_pairs
    
    def _get_pairs_neighbors(self, pairs):
        
        # Add neighboring indices to pairs in new dimension
        # where neighbors come from square around the pair
        # pair kernel size comes from relationship (2x+1)^2 = 2k + 1
        # pair_kernel_size = (np.sqrt(2 * self.pair_kernel_size + 1) - 1) // 2
        pair_kernel_size = self.pair_kernel_size
        
        # Create coordinate grids
        x = torch.arange(2*pair_kernel_size + 1)
        y = torch.arange(2*pair_kernel_size + 1)
        
        # Create meshgrid
        X, Y = torch.meshgrid(x, y, indexing="ij")
        
        # Stack X and Y to form the coordinate tensor
        coordinate_tensor = torch.stack((X, Y), dim=2)
        
        # Get the kernel offset
        kernel_offset = coordinate_tensor - pair_kernel_size
        
        return pairs.unsqueeze(-2).unsqueeze(-2) + kernel_offset.to(self.device)
    
    def _get_pair_weights(self):
        
        # Add neighboring indices to pairs in new dimension
        # where neighbors come from square around the pair
        # pair kernel size comes from relationship (2x+1)^2 = 2k + 1
        # pair_kernel_size = (np.sqrt(2 * self.pair_kernel_size + 1) - 1) // 2
        
        pair_kernel_size = self.pair_kernel_size
        
        # Create coordinate grids
        x = torch.arange(2*pair_kernel_size + 1)
        y = torch.arange(2*pair_kernel_size + 1)
        
        # Create meshgrid
        X, Y = torch.meshgrid(x, y, indexing="ij")
        
        # Stack X and Y to form the coordinate tensor
        coordinate_tensor = torch.stack((X, Y), dim=2)
        
        # Get the kernel offset
        kernel_offset = coordinate_tensor - pair_kernel_size
        
        weights = torch.exp(
            -torch.square(kernel_offset).sum(dim=2).to(self.device) / (2 * self.pair_kernel_weight)
        )
        
        return weights
        
        

    def forward(self, mains=None, pairs=None, on_partialed_set=False):
        # mains of shape (batch_size, n_features)
        # pairs of shape (batch_size, |n_features choose 2|, 2)
        
        output_main = 0
        output_pairs = 0
        
        if mains is not None:
            
            inputs = mains.clone()
            
            if not on_partialed_set:
                mains = mains[:, self.active_feats]
            
            # Add offsets to features to get the correct indices
            offsets = self.embedding_offsets.unsqueeze(0).expand(mains.size(0), -1).to(self.device)
            mains = mains + offsets[:, self.active_feats]
            
            if self.kernel_size > 0 and self.kernel_weight > 0:
            
                # Add neighboring indices to mains in new dimension
                # mains is now of shape (batch_size, n_features, 2 * kernel_size + 1)
                mains = mains.unsqueeze(-1) + torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)
                
                # Add the embeddings for each feature using weight function
                # weights will be of shape (2 * kernel_size + 1)
                weights = torch.exp(-torch.square(torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)) / (2 * self.kernel_weight))
                
                # weights now of shape (batch_size, n_features, 2 * kernel_size + 1)
                # Add a <= instead of < on left side to eliminate weight on missing bin
                weights = torch.where(
                    (mains <= self.embedding_offsets[self.active_feats].reshape(1, -1, 1)) | (mains >= self.embedding_offsets[self.active_feats].reshape(1, -1, 1) + self.feature_sizes[self.active_feats].reshape(1, -1, 1)),
                    torch.zeros_like(weights),
                    weights
                )
                
                # Zero out weights for categorical feats
                weights = torch.where(
                    torch.logical_and(
                        self.cat_feat_mask[self.active_feats].reshape(1, -1, 1),
                        weights != 1
                    ),
                    torch.zeros_like(weights),
                    weights
                )
                
                mains = torch.sum(
                    self.embedding(
                        torch.clamp(mains.long(), 0, self.total_feature_size-1)
                    ) * weights.unsqueeze(-1), 
                    dim=2
                )
                
            else:
                mains = self.embedding(mains.long())
            
            for w, b, a in zip(self.main_weights, self.main_biases, self.main_activations):
                mains = torch.einsum(
                    'ijk,jkl->ijl', 
                    mains, 
                    w[self.active_feats, :, :]
                ) + b[self.active_feats, :]
                mains = a(mains)
            
            # Apply monotonic constraints
            if self.monotone_constraints is not None:
                mains_squared = torch.square(mains)
                mains = torch.where(self.monotone_constraints_tensor == 1, mains_squared, mains)
                mains = torch.where(self.monotone_constraints_tensor == -1, -1 * mains_squared, mains)
                
                for feat_idx in range(len(self.monotone_constraints)):
                    if self.monotone_constraints[feat_idx] != 0:
                        bin_scores = self.get_bin_scores(feat_idx, center=False, backprop=True) # (feature_size, n_output)
                        feat_input = inputs[:, feat_idx].long()
                        mains[:, feat_idx, :] = bin_scores[feat_input - 1, :] + mains[:, feat_idx, :] # (batch_size, n_output)
                
            # Get smoothed z
            if self.reg_param > 0:
                z_main = self.get_smooth_z()[self.active_feats]
                output_main = torch.einsum('ijk,j->ik', mains, z_main)
            else:
                output_main = mains.sum(dim=1)
        
        if pairs is not None:
            
            if not on_partialed_set:
                pairs = pairs[:, self.active_pairs, :]
            
            # Get cat_feat_mask
            pairs_cat_feat_mask = self.cat_feat_mask[self.pairs_list[self.active_pairs]]
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (n_pairs, 2)
            offsets = self.embedding_offsets[self.pairs_list[self.active_pairs]].to(self.device)
            pairs = pairs + offsets
            
            if self.pair_kernel_size > 0 and self.pair_kernel_weight > 0:
            
                # Pairs will now be shape (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2)
                pairs = self._get_pairs_neighbors(pairs)
                
                # Get the kernel weights using gaussian kernel
                # Will be shape (2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = self._get_pair_weights()
                
                pair_sizes = self.feature_sizes[self.pairs_list[self.active_pairs]].to(self.device)
                
                # Add a <= instead of < on left side so eliminate weight on missing bin
                # weights is now (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = torch.where(
                    (pairs[:, :, :, :, 0] <= offsets[:, 0].reshape(1, -1, 1, 1)) | (pairs[:, :, :, :, 1] <= offsets[:, 1].reshape(1, -1, 1, 1)) | \
                        (pairs[:, :, :, :, 0] >= offsets[:, 0].reshape(1, -1, 1, 1) + pair_sizes[:, 0].reshape(1, -1, 1, 1)) | (pairs[:, :, :, :, 1] >= offsets[:, 1].reshape(1, -1, 1, 1) + pair_sizes[:, 1].reshape(1, -1, 1, 1)),
                    torch.zeros_like(weights), 
                    weights
                )
                
                # Zero out weights completely where both features are categorical
                weights = torch.where(
                    torch.logical_and(
                        (pairs_cat_feat_mask.sum(dim=-1) == 2).reshape(1, -1, 1, 1),
                        weights != 1
                    ),
                    torch.zeros_like(weights),
                    weights
                )
                
                # When only second feature is categorical
                weights = torch.where(
                    torch.logical_and(
                        torch.logical_and(
                            ~pairs_cat_feat_mask[:, 0],
                            pairs_cat_feat_mask[:, 1],
                        ).reshape(-1, 1, 1),
                        torch.arange(2*self.pair_kernel_size + 1).reshape(1, 1, -1).to(self.device) != self.pair_kernel_size
                    ).unsqueeze(0),
                    torch.zeros_like(weights),
                    weights
                )
                
                # When only first feature is categorical
                weights = torch.where(
                    torch.logical_and(
                        torch.logical_and(
                            pairs_cat_feat_mask[:, 0],
                            ~pairs_cat_feat_mask[:, 1],
                        ).reshape(-1, 1, 1),
                        torch.arange(2*self.pair_kernel_size + 1).reshape(1, -1, 1).to(self.device) != self.pair_kernel_size
                    ).unsqueeze(0),
                    torch.zeros_like(weights),
                    weights
                )
                
                # pairs has shape (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2, n_embed)
                pairs = self.pair_embedding(
                    torch.clamp(pairs.long(), 0, self.total_feature_size-1)
                )
                
                
                # pairs now has shape (batch_size, n_pairs, 2, n_embed)
                pairs = torch.sum(
                    pairs * weights.unsqueeze(-1).unsqueeze(-1), 
                    dim=[2, 3]
                )
                
                # Concat embeddings of both features in pair
                # pairs now has shape (batch_size, n_pairs, 2 * n_embed)
                pairs = pairs.reshape(pairs.size(0), pairs.size(1), 2*self.n_embed)
                
                
            else:
                pairs = self.pair_embedding(pairs.long())
                pairs = pairs.reshape(pairs.size(0), pairs.size(1), 2*self.n_embed)
        

            for w, b, a in zip(self.pair_weights, self.pair_biases, self.pair_activations):
                pairs = torch.einsum(
                    'ijk,jkl->ijl', 
                    pairs, 
                    w[self.active_pairs, :, :]
                ) + b[self.active_pairs, :]
                pairs = a(pairs)
                
            if self.pair_reg_param > 0:
                z_pairs = self.get_smooth_z_pairs()[self.active_pairs]
                output_pairs = torch.einsum('ijk,j->ik', pairs, z_pairs)
            else:
                output_pairs = pairs.sum(dim=1)
        
        
        return output_main + output_pairs
    
    
    def loss_penalty(self, mains=True):
        
        
        # if mains = True then mains are being fit
        # else pairs are being fit
        if mains:
            z_main = self.get_smooth_z()[self.active_feats]
            return self.reg_param * z_main.sum() + self.get_entropy_reg(mains=mains)
        else:
            z_pairs = self.get_smooth_z_pairs()[self.active_pairs]
            return self.pair_reg_param * z_pairs.sum() + self.get_entropy_reg(mains=mains)

        
    def get_entropy_reg(self, mains=True):
        
        # if mains = True then mains are being fit
        # else pairs are being fit
        if mains:
            z_main = self.get_smooth_z()[self.active_feats]
            
            # Control consider the smooth zs which are not already 0 or 1
            z_main_active = z_main[
                (z_main > 0) & (z_main < 1)
            ]
            
            omegas_main = -(z_main_active * torch.log(z_main_active) + \
                            (1 - z_main_active) * torch.log(1 - z_main_active))
            
            return self.entropy_param * omegas_main.sum()
        
        else:
            z_pairs = self.get_smooth_z_pairs()[self.active_pairs]
            
            z_pairs_active = z_pairs[
                (z_pairs > 0) & (z_pairs < 1)
            ]
            
            omegas_pairs = -(z_pairs_active * torch.log(z_pairs_active) + \
                             (1 - z_pairs_active) * torch.log(1 - z_pairs_active))
            
            return self.entropy_param * omegas_pairs.sum()
        
    def freeze_main_effects(self):
        self.main_effects_frozen = True
        for w, b in zip(self.main_weights, self.main_biases):
            w.requires_grad = False
            b.requires_grad = False
        
        self.z_main.requires_grad = False
        
        
        # Copy the current embeddings into new pair embeddings
        import copy
        self.pair_embedding = nn.Embedding(self.total_feature_size, self.n_embed)
        self.pair_embedding.weight = copy.deepcopy(self.embedding.weight)
        
        # Remove grad from main embedding
        self.embedding.weight.requires_grad = False
        
        return
    
    # Get the predicted score for a feature in each bin
    # feat_index: index of the feature
    # center: whether to center the bin scores so that sum over training set is 0
    def get_bin_scores(self, feat_index, center=True, backprop=False):
        
        context = torch.no_grad() if not backprop else nullcontext()
        
        with context:
        
            feat_inputs = torch.arange(0, self.feature_sizes[feat_index]).to(self.device)
            
            # Add offsets to features to get the correct indices
            # offsets = torch.tensor(self.embedding_offsets).unsqueeze(0).expand(inputs.size(0), -1).to(self.device)
            offset = self.embedding_offsets[feat_index]
            inputs = feat_inputs + offset
            
            if self.kernel_size > 0 and self.kernel_weight > 0 and not self.cat_feat_mask[feat_index]:
                
                # Apply kernel smoothing to the embeddings
                inputs = inputs.unsqueeze(-1) + torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)
                weights = torch.exp(-torch.square(torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)) / (2 * self.kernel_weight))
                
                weights = torch.where(
                    (inputs <= offset) | (inputs >= offset + self.feature_sizes[feat_index]),
                    torch.zeros_like(weights),
                    weights
                )
                
                mains = torch.sum(
                    self.embedding(
                        torch.clamp(inputs.long(), 0, self.total_feature_size-1)
                    ) * weights.unsqueeze(-1),
                    dim=1
                )
                
            else:
                # inputs = self.embeddings[feat_index](inputs.long())
                mains = self.embedding(inputs.long())
                
                
            for w, b, a in zip(self.main_weights, self.main_biases, self.main_activations):
                mains = torch.einsum(
                    'ik,kl->il', 
                    mains, 
                    w[feat_index, :, :]
                ) + b[feat_index, :]
                mains = a(mains)
                
            # Apply monotonic constraints
            if self.monotone_constraints is not None and self.monotone_constraints[feat_index] != 0:
                mains_squared = torch.square(mains)
                if self.monotone_constraints[feat_index] == 1:
                    mains = mains_squared
                elif self.monotone_constraints[feat_index] == -1:
                    mains = -1 * mains_squared
                
                # bin_scores = self.get_bin_scores(feat_idx, center=False) # (feature_size, n_output)
                # cum_bin_scores = torch.cumsum(mains, dim=0)
                # feat_input = feat_inputs.long()
                # mains = self.monotone_params[[feat_index], :] + cum_bin_scores[feat_input - 1, :] + mains # (batch_size, n_output)
                mains = torch.cumsum(mains, dim=0) + self.monotone_params[[feat_index], :]


            # Get prediction using the smoothed z
            if self.reg_param > 0:
                z_main = self.get_smooth_z()
                prediction = mains * z_main[feat_index]
            else:
                prediction = mains
            
            if hasattr(self, 'feat_offsets') and center:
                return prediction - self.feat_offsets[feat_index, :]
            elif center:
                self.logger.warning("Intercept not computed. Shape function will not be centered.")
                return prediction
            else:
                return prediction
            
    def get_pair_bin_scores(self, pair_index, center=True):
        
        with torch.no_grad():
        
            # Get the pair
            pair = self.pairs_list[pair_index]
            
            # Get the sizes of the features in the pair
            size1 = self.feature_sizes[pair[0]]
            size2 = self.feature_sizes[pair[1]]
            
            # Get all pairs of indices for the pair
            # pairs is of shape (batch_size, 2)
            pairs = torch.stack(torch.meshgrid(
                torch.arange(size1), torch.arange(size2), indexing="ij"
            ), dim=-1).reshape(-1, 2).to(self.device)
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (2)
            offsets = self.embedding_offsets[self.pairs_list[pair_index]].to(self.device)
            pairs = pairs + offsets
            
            if self.pair_kernel_size > 0 and self.pair_kernel_weight > 0 and not (self.cat_feat_mask[pair[0]] and self.cat_feat_mask[pair[1]]):
            
                # Pairs will now be shape (batch_size, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2)
                pairs = self._get_pairs_neighbors(pairs)
                
                # Get the kernel weights using gaussian kernel
                # Will be shape (2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = self._get_pair_weights()
                
                pair_sizes = self.feature_sizes[self.pairs_list[pair_index]].to(self.device)
                
                # weights is now (batch_size, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = torch.where(
                    (pairs[:, :, :, 0] <= offsets[0]) | (pairs[:, :, :, 1] <= offsets[1]) | \
                        (pairs[:, :, :, 0] >= offsets[0] + pair_sizes[0]) | (pairs[:, :, :, 1] >= offsets[1] + pair_sizes[1]),
                    torch.zeros_like(weights), 
                    weights
                )
                
                if self.cat_feat_mask[pair[0]] and not self.cat_feat_mask[pair[1]]:
                    weights = weights[:, :, self.pair_kernel_size] # (batch_size, 2*pair_kernel_size + 1)
                    pairs = pairs[:, :, self.pair_kernel_size, :] # (batch_size, 2*pair_kernel_size + 1, 2)
                    
                    # pairs has shape (batch_size, 2 * pair_kernel_size + 1, 2, n_embed)
                    pairs = self.pair_embedding(
                        torch.clamp(pairs.long(), 0, self.total_feature_size-1)
                    )    
                    pairs = torch.sum(
                        pairs * weights.unsqueeze(-1).unsqueeze(-1), 
                        dim=1
                    )
                elif not self.cat_feat_mask[pair[0]] and self.cat_feat_mask[pair[1]]:
                    weights = weights[:, self.pair_kernel_size, :]
                    pairs = pairs[:, self.pair_kernel_size, :, :]
                    
                    # pairs has shape (batch_size, 2 * pair_kernel_size + 1, 2, n_embed)
                    pairs = self.pair_embedding(
                        torch.clamp(pairs.long(), 0, self.total_feature_size-1)
                    )    
                    pairs = torch.sum(
                        pairs * weights.unsqueeze(-1).unsqueeze(-1), 
                        dim=1
                    )
                else:
                
                    # pairs has shape (batch_size, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2, n_embed)
                    pairs = self.pair_embedding(
                        torch.clamp(pairs.long(), 0, self.total_feature_size-1)
                    )    
                    
                    # pairs now has shape (batch_size, 2, n_embed)
                    pairs = torch.sum(
                        pairs * weights.unsqueeze(-1).unsqueeze(-1), 
                        dim=[1, 2]
                    )
                    
                # Concat embeddings of both features in pair
                # pairs now has shape (batch_size, 2 * n_embed)
                pairs = pairs.reshape(pairs.size(0), 2*self.n_embed)
                
                
            else:
                
                pairs = self.pair_embedding(pairs.long())
                pairs = pairs.reshape(pairs.size(0), 2*self.n_embed)
            
            for w, b, a in zip(self.pair_weights, self.pair_biases, self.pair_activations):
                pairs = torch.einsum(
                    'ik,kl->il', 
                    pairs, 
                    w[pair_index, :, :]
                ) + b[pair_index, :]
                pairs = a(pairs)
            
            z_pairs = self.get_smooth_z_pairs()
            prediction = pairs * z_pairs[pair_index]
            
            if hasattr(self, 'pair_offsets') and center:
                return prediction - self.pair_offsets[pair_index, :]
            elif center:
                self.logger.warning("Intercept not computed. Shape function will not be centered.")
                return prediction
            else:
                return prediction
        
    def prune_parameters(self, mains=True, pairs=False):
        
        if mains:
        
            z_main = self.get_smooth_z()
            self.active_feats = torch.where(z_main > 0)[0]
        
        if pairs:
            assert self.fit_pairs, "Tried to prune pairs parameters without model having pairs"
            
            z_pairs = self.get_smooth_z_pairs()
            self.active_pairs = torch.where(z_pairs > 0)[0] 
            
    def compute_intercept(self, bin_counts, ignore_missing_bin=False, has_missing_bin=None):
        
        # bin counts is a list of bin counts for each feature
        if has_missing_bin is None:
            has_missing_bin = [True] * self.n_features
        
        self.bin_counts = bin_counts
        self.feat_offsets = torch.zeros(self.n_features, self.n_output).to(self.device)
        
        for feat_idx in self.active_feats:
            bin_preds = self.get_bin_scores(feat_idx, center=False)
            feat_bin_counts = torch.tensor(bin_counts[feat_idx]).to(self.device).unsqueeze(-1)
            
            if ignore_missing_bin and has_missing_bin[feat_idx]:
                self.logger.debug(f"Ignoring missing bin for feature {feat_idx.item()} in Intercept Calculation")
                feat_bin_counts = feat_bin_counts[1:]
                bin_preds = bin_preds[1:, :]
            
            feat_pred = torch.sum(
                bin_preds * feat_bin_counts, dim=0
            ) / torch.sum(feat_bin_counts)
            self.feat_offsets[feat_idx, :] = feat_pred
            
        self.intercept = torch.sum(self.feat_offsets, dim=0)
        
        return
    
    def compute_pairs_intercept(self, pair_bin_counts):
            
        # pair_bin_counts is a list of bin counts for each pair
        
        self.pair_bin_counts = pair_bin_counts
        self.pair_offsets = torch.zeros(self.n_pairs, self.n_output).to(self.device)
        
        for pair_idx in range(self.n_pairs):
            bin_preds = self.get_pair_bin_scores(pair_idx, center=False)
            bin_counts = torch.tensor(pair_bin_counts[pair_idx]).to(self.device).unsqueeze(-1)
            pair_pred = torch.sum(
                bin_preds * bin_counts, dim=0
            ) / torch.sum(bin_counts)
            self.pair_offsets[pair_idx, :] = pair_pred
            
        self.pairs_intercept = torch.sum(self.pair_offsets, dim=0)    
        
        return
    
    def train_(
        self, 
        train_epoch_fn, 
        test_epoch_fn,
        train_loader,
        test_loader,
        optimizer,
        n_epochs,
        mains=True,
        verbose=False,
        show_feat_count=False,
        show_pair_count=False,
    ):
        

        early_stopping_counter = 0
        best_test_loss = float('inf')
        best_model_state = None

        for epoch in range(n_epochs):
            train_loss = train_epoch_fn(self, train_loader, optimizer)
            test_loss, _ = test_epoch_fn(self, test_loader)
            
            # Do feature pruning
            if self.penalized:
                if mains:
                    self.prune_parameters(mains=True, pairs=False)
                else:
                    self.prune_parameters(mains=False, pairs=True)
            
            if show_feat_count:
                self.logger.info(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f} | Active features: {len(self.active_feats)}")
            elif show_pair_count:
                self.logger.info(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f} | Active pairs: {len(self.active_pairs)}")
            else:
                self.logger.info(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f}")
                
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stopping_counter = 0
            
                # Save the model state in memory (CPU)
                best_model_state = {k: v.cpu() for k, v in self.state_dict().items()}
            
            else:
                early_stopping_counter += 1

            # If test loss has not improved for 5 consecutive epochs, terminate training
            if early_stopping_counter >= 5:
                self.logger.info(f"Early stopping at {epoch+1} epochs: Test loss has not improved for 5 consecutive epochs.")
                break
            
        # Load the model from the best test loss
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return best_test_loss

class BaseDNAMiteModel(nn.Module, LoggingMixin):
    """
    BaseDNAMiteModel is the parent class for DNAMite models.

    This class provides the foundational architecture for DNAMite models, including embedding layers, hidden layers, and output layers. It also manages training and validation processes.

    Parameters
    ----------
    n_features : int
        Number of input features in the data.
    n_output : int
        Number of output features.
    n_embed : int, default=32
        Dimensionality of the embedding layer.
    n_hidden : int, default=32
        Dimension of each hidden layer.
    n_layers : int, default=2
        Number of hidden layers in the model.
    max_bins : int, default=32
        Maximum number of bins for feature discretization.
    min_samples_per_bin : int, default=None
        Minimum number of samples required in each bin.
        Default is None which sets it to min(n_train_samples / 100, 50).
    validation_size : float, default=0.2
        Proportion of data to use for validation.
    n_val_splits : int, default=5
        Number of validation splits to average over.
    learning_rate : float, default=1e-4
        Learning rate for the optimizer.
    max_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=128
        Batch size for training.
    device : str, default="cpu"
        Device for model computation, either "cpu" or "cuda".
    pairs_list : list of tuple[int, int], optional
        List of feature pairs to consider for interactions; if None, no specific pairs are used.
    kernel_size : int, default=5
        Size of the kernel used in smoothing for single features.
    kernel_weight : float, default=3
        Weight applied to the smoothing kernel for single features.
    pair_kernel_size : int, default=3
        Kernel size for smoothing pairwise feature interactions.
    pair_kernel_weight : float, default=3
        Weight applied to the smoothing kernel for feature pairs.
    monotone_constraints : list or None, optional (default=None)
        The monotonic constraints for the features.
        0 indicates no constraint, 1 indicates increasing, -1 indicates decreasing.
        None means no constraints.
    num_pairs : int, default=0.
        Number of pairwise interactions to use in the model.
    verbosity : int, default=0
        Level of verbosity for logging.
        0: Warning, 1: Info, 2: Debug
    random_state: int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, 
                 n_features, 
                 n_output, 
                 n_embed=32, 
                 n_hidden=32,
                 n_layers=2, 
                 max_bins=32, 
                 min_samples_per_bin=None,
                 validation_size=0.2, 
                 n_val_splits=5, 
                 learning_rate=1e-4, 
                 max_epochs=100, 
                 batch_size=128, 
                 device="cpu", 
                 pairs_list=None,
                 kernel_size=5, 
                 kernel_weight=3, 
                 pair_kernel_size=3, 
                 pair_kernel_weight=3, 
                 monotone_constraints=None,
                 num_pairs=0,
                 verbosity=0,
                 random_state=None,
    ):
        nn.Module.__init__(self)
        LoggingMixin.__init__(self, verbosity=verbosity)
        self.n_features = n_features
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.max_bins = max_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.validation_size = validation_size
        self.n_val_splits = n_val_splits
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.pairs_list = pairs_list
        self.kernel_size = kernel_size
        self.kernel_weight = kernel_weight
        self.pair_kernel_size = pair_kernel_size
        self.pair_kernel_weight = pair_kernel_weight
        self.monotone_constraints = monotone_constraints
        self.num_pairs = num_pairs
        self.verbosity = verbosity
        self.random_state = random_state if random_state is not None else np.random.randint(0, 1000)
        
        self.models = nn.ModuleList()
        self.fit_pairs = self.num_pairs > 0
        
    def _set_gamma(self, X):
        if self.gamma is None:
            self.gamma = min(((X.shape[0] / self.batch_size) / 250) / (self.n_hidden / 16), 1.0)
        if self.pair_gamma is None:
            self.pair_gamma = self.gamma / 4
    
    def _infer_data_types(self, X):
        
        self.feature_dtypes = []
        for i in range(X.shape[1]):
            if set(X.iloc[:, i].unique()).issubset({0, 1}):
                self.feature_dtypes.append('binary')
            elif X.iloc[:, i].dtype.name == 'category' or \
                X.iloc[:, i].dtype.name == 'object':
                self.feature_dtypes.append('categorical')
            else:
                self.feature_dtypes.append('continuous')
                
        self.cat_feat_mask = np.array(self.feature_dtypes) != 'continuous'
        self.cat_feat_mask = torch.tensor(self.cat_feat_mask).to(self.device)
    
    def _discretize_data(self, X):
        X_discrete = X.copy()
        
        if not hasattr(self, 'feature_dtypes'):
            self._infer_data_types(X_discrete)
        
        # If X is pandas, store column names
        if hasattr(X, 'columns'):
            col_names = X.columns
            X_discrete = X_discrete.values
        
        # If feature_bins is already set, use existing bins
        if hasattr(self, 'feature_bins'):
            for i in range(self.n_features):
                if self.feature_dtypes[i] == 'continuous':
                    X_discrete[:, i], _ = discretize(np.ascontiguousarray(X_discrete[:, i]), max_bins=self.max_bins, bins=self.feature_bins[i])
                elif self.feature_dtypes[i] == 'binary':
                    ordinal_map = {val: float(j) for j, val in enumerate(self.feature_bins[i])}
                    X_discrete[:, i] = np.vectorize(ordinal_map.get)(X_discrete[:, i].astype(float))
                else:
                    ordinal_map = {val: float(j) for j, val in enumerate(self.feature_bins[i])}
                    
                    # Replace the NAs because they are handled incorrectly in dict
                    X_discrete[:, i] = np.where(X_discrete[:, i] == X_discrete[:, i], X_discrete[:, i], "NA")
                    ordinal_map["NA"] = 0.0
                    
                    for val in pd.Series(X_discrete[:, i]).unique():
                        if val not in ordinal_map:
                            # This means that the category was either not seen in training
                            # Or was an infrequent category
                            if self.feature_bins[i][-1] == "Other":
                                ordinal_map[val] = self.feature_sizes[i] - 1.0
                            else:
                                ordinal_map[val] = 0.0
                    X_discrete[:, i] = np.vectorize(ordinal_map.get)(X_discrete[:, i])
                
        else:
            self.feature_bins = []
            self.feature_sizes = []
            self.has_missing_bin = []
            self.logger.debug("Discretizing features...")
            from time import time
            for i in range(self.n_features):
                if self.feature_dtypes[i] == 'continuous':
                    X_discrete[:, i], bins = discretize(
                        np.ascontiguousarray(X_discrete[:, i]), 
                        max_bins=self.max_bins, 
                        min_samples_per_bin=self.min_samples_per_bin
                    )
                    self.has_missing_bin.append(True)
                elif self.feature_dtypes[i] == 'binary':
                    bins = [0, 1]
                    self.has_missing_bin.append(False)
                else:
                    # Ordinal encoder and force missing/unknown value to be 0
                    ordinal_encoder = OrdinalEncoder(
                        dtype=float, 
                        min_frequency=min(0.01*X.shape[0], 50),
                        handle_unknown="use_encoded_value", 
                        unknown_value=-1, 
                        encoded_missing_value=-1
                    )
                    
                    X_discrete[:, i] = ordinal_encoder.fit_transform(X_discrete[:, i].reshape(-1, 1)).flatten() # + 1.0
                    
                    infrequent_categories = ordinal_encoder.infrequent_categories_[0]
                    if infrequent_categories is None:
                        infrequent_categories = []
                    
                    if X_discrete[:, i].min() == -1:
                        # offset and add np.nan to bins
                        X_discrete[:, i] += 1.0
                        bins = [np.nan] + [c for c in ordinal_encoder.categories_[0] if c is not None and c == c and c not in infrequent_categories]
                        self.has_missing_bin.append(True)
                    else:
                        bins = [c for c in ordinal_encoder.categories_[0] if c is not None and c == c and c not in infrequent_categories]
                        self.has_missing_bin.append(False)
                        
                    if len(infrequent_categories) > 0:
                        bins.append("Other")
                
                self.feature_bins.append(bins)
                
                # Number of bins is (maximum bin index) + 1 (accounting for missing bin)
                self.feature_sizes.append(int(X_discrete[:, i].max() + 1))
                # self.feature_sizes.append(len(bins))
            
        if hasattr(X, 'columns'):
            X_discrete = pd.DataFrame(X_discrete, columns=col_names, dtype=float)
            
        return X_discrete
    
    def _compute_bin_scores(self, ignore_missing_bin_in_intercept=True):
        
        # has_missing_bins = [bins[0] != bins[0] if len(bins) > 0 else True for bins in self.feature_bins]
        
        for model in self.models:
        
            model.compute_intercept(
                model.bin_counts, 
                ignore_missing_bin=ignore_missing_bin_in_intercept,
                has_missing_bin=self.has_missing_bin
            )
            
            model.bin_scores = []
            for i in range(model.n_features):
                model.bin_scores.append(
                    model.get_bin_scores(feat_index=i, center=True).cpu().numpy()
                )
            
            if model.fit_pairs:
    
                pairs_list = model.pairs_list.cpu().numpy()
                model.compute_pairs_intercept(model.pair_bin_counts)
                
                model.pair_bin_scores = []
                for i in range(len(pairs_list)):
                    model.pair_bin_scores.append(
                        model.get_pair_bin_scores(pair_index=i, center=True).cpu().numpy()
                    )
                
        return
    
    def _fit_one_split(
        self, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        random_state=None, 
        partialed_feats=None, 
        train_loader_args={},
        val_loader_args={},
    ):
        
        # Set random seed if one is provided
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            self.logger.debug("Found selected features. Using only those features.")
            X_train = X_train[self.selected_feats]
            X_val = X_val[self.selected_feats]
            
            if self.fit_pairs:
                pairs_list = [
                    [X_train.columns.get_loc(feat1), X_train.columns.get_loc(feat2)] for feat1, feat2 in self.selected_pairs
                ]
            else:
                pairs_list = None
                
            feature_sizes = [self.feature_sizes[self.feature_names_in_.get_loc(feat)] for feat in self.selected_feats]
        else:
            pairs_list = self.pairs_list
            # pairs_list = None
            feature_sizes = self.feature_sizes
            
        if partialed_feats is not None:
            partialed_indices = [X_train.columns.get_loc(feat) for feat in partialed_feats]
        else:
            partialed_indices = None
        
        # Get data loaders
        self.logger.debug("GETTING DATA LOADERS")
        train_loader = self.get_data_loader(X_train, y_train, **train_loader_args)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False, **val_loader_args)
        
        model = _BaseSingleSplitDNAMiteModel(
            n_features=X_train.shape[1], 
            n_output=self.n_output,
            feature_sizes=feature_sizes, 
            n_embed=self.n_embed,
            n_hidden=self.n_hidden, 
            n_layers=self.n_layers,
            fit_pairs=self.fit_pairs, 
            device=self.device, 
            pairs_list=pairs_list, 
            kernel_size=self.kernel_size,
            kernel_weight=self.kernel_weight,
            pair_kernel_size=self.pair_kernel_size,
            pair_kernel_weight=self.pair_kernel_weight,
            cat_feat_mask=self.cat_feat_mask,
            monotone_constraints=self.monotone_constraints,
            verbosity=self.verbosity
        ).to(self.device)
        model.feature_names_in_ = X_train.columns
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        if partialed_indices is not None:
            self.logger.debug("TRAINING PARTIALED FEATS")
            model.active_feats = torch.tensor(partialed_indices).to(self.device)
            
            # Get data loaders only for partialed feats
            train_loader = self.get_data_loader(X_train.iloc[:, partialed_indices], y_train, **train_loader_args)
            val_loader = self.get_data_loader(X_val.iloc[:, partialed_indices], y_val, shuffle=False, **val_loader_args)
            
            model.train_(
                partial(self.train_epoch_mains, on_partialed_set=True), 
                partial(self.test_epoch_mains, on_partialed_set=True), 
                train_loader, 
                val_loader, 
                optimizer, 
                self.max_epochs,
                mains=True,
            )
            model.active_feats = torch.arange(len(X_train.columns)).to(self.device)
            
            # Get data loaders
            train_loader = self.get_data_loader(X_train, y_train, **train_loader_args)
            val_loader = self.get_data_loader(X_val, y_val, shuffle=False, **val_loader_args)
        
        self.logger.debug("TRAINING MAINS")
        model.train_(
            train_epoch_fn=partial(self.train_epoch_mains, partialed_indices=partialed_indices),
            test_epoch_fn=self.test_epoch_mains,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            n_epochs=self.max_epochs,
            mains=True,
        )
        
        if hasattr(self, 'selected_feats'):
            model.feature_bins = [self.feature_bins[self.feature_names_in_.get_loc(feat)] for feat in self.selected_feats]
            model.bin_counts = [
                get_bin_counts(X_train[col], nb) for col, nb in zip(X_train.columns, feature_sizes)
            ]
        else:
            model.feature_bins = self.feature_bins
                
            # Compute the bin counts
            model.bin_counts = [
                get_bin_counts(X_train[col], nb) for col, nb in zip(X_train.columns, self.feature_sizes)
            ]
        
        if not self.fit_pairs:
            return model
        
        model.freeze_main_effects()
        
        if pairs_list is None:
            # No pairs have been selected already, so do a round of pair selection
            self._select_pairs(model, X_train, y_train, X_val, y_val, self.num_pairs, train_loader_args, val_loader_args)
            pairs_list = self.pairs_list
            model.selected_pair_indices = pairs_list # list version
            model.pairs_list = torch.LongTensor(pairs_list).to(self.device) # tensor version
            model.n_pairs = len(pairs_list)
            model.init_pairs_params(model.n_pairs)
            model.to(self.device)
            
        X_train_interactions = X_train.values[:, pairs_list]
        X_val_interactions = X_val.values[:, pairs_list]
        
        train_loader = self.get_data_loader(X_train, y_train, pairs=X_train_interactions, **train_loader_args)
        val_loader = self.get_data_loader(X_val, y_val, pairs=X_val_interactions, shuffle=False, **val_loader_args)
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=self.learning_rate / 5
        )
        
        self.logger.debug("TRAINING PAIRS")
        model.train_(
            self.train_epoch_pairs, 
            self.test_epoch_pairs, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs,
            mains=False
        )
        
        if self.fit_pairs:
            pairs_list = model.pairs_list.cpu().numpy()
            model.pair_bin_counts = [
                get_pair_bin_counts(X_train.iloc[:, col1], X_train.iloc[:, col2]) 
                for col1, col2 in pairs_list
            ]
    
        return model
    
    def select_features(
        self, 
        X, 
        y, 
        reg_param,
        select_pairs=False, 
        partialed_feats=None,
        gamma=None,
        pair_gamma=None, 
        pair_reg_param=0, 
        entropy_param=0, 
    ):
        """
        Perform feature selection. Selected features and pairs will be stored in model.selected_feats
        and model.selected_pairs, respectively. Should be called before fit if feature selection is desired.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for the model.   
        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable. 
        reg_param : float
            Regularization parameter for feature-level regularization.
        select_pairs : bool, default=False
            Whether to select feature pairs in addition to individual features.
        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        gamma : float, default=1
            Regularization or scaling parameter; purpose depends on specific use in the model.
        pair_gamma : float or None, default=None
            Regularization or scaling parameter for feature pairs.
            If None, defaults to gamma.
        pair_reg_param : float, default=0
            Regularization parameter for feature pairs.
        entropy_param : float, default=0
            Entropy parameter to control the diversity or uncertainty.
        """
        self.gamma = gamma
        self.pair_gamma = pair_gamma if pair_gamma is not None else gamma
        self.reg_param = reg_param
        self.pair_reg_param = pair_reg_param
        self.entropy_param = entropy_param
        
        if self.reg_param <= 0 and self.pair_reg_param <= 0:
            raise ValueError("Regularization parameters must be greater than 0 for feature selection.")
        
        if select_pairs:
            self.fit_pairs = True
        
        _ = self._select_features(X, y, partialed_feats=partialed_feats)
        return
    
    def _select_features(self, X, y, partialed_feats=None, model=None, train_loader_args={}, val_loader_args={}):
        self.logger.debug("STARTING SELECT FEATURES...")
        
        # Set random seed if one is provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
        if partialed_feats is not None:
            partialed_indices = [X.columns.get_loc(feat) for feat in partialed_feats]
        else:
            partialed_indices = None
        
        self.feature_names_in_ = X.columns
        
        # First discretize the data
        X_discrete = self._discretize_data(X)
        
        if hasattr(X, 'values'):
            X_discrete = X_discrete.values
        
        # Make one train/val split for feature selection
        X_train, X_val, y_train, y_val = train_test_split(X_discrete, y, test_size=self.validation_size, random_state=self.random_state)
        
        # Set gamma using size of training data
        self._set_gamma(X_train)
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train, **train_loader_args)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False, **val_loader_args)
        
        if model is None:
            model = _BaseSingleSplitDNAMiteModel(
                n_features=self.n_features, 
                n_output=self.n_output,
                feature_sizes=self.feature_sizes, 
                n_embed=self.n_embed,
                n_hidden=self.n_hidden, 
                n_layers=self.n_layers,
                gamma=self.gamma, 
                pair_gamma=self.pair_gamma,
                reg_param=self.reg_param, 
                pair_reg_param=self.pair_reg_param, 
                entropy_param=self.entropy_param, 
                device=self.device, 
                kernel_size=self.kernel_size,
                kernel_weight=self.kernel_weight,
                pair_kernel_size=self.pair_kernel_size,
                pair_kernel_weight=self.pair_kernel_weight,
                cat_feat_mask=self.cat_feat_mask,
                monotone_constraints=self.monotone_constraints,
                verbosity=self.verbosity
            ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        if partialed_indices is not None:
            self.logger.debug("TRAINING ON PARTIALED FEATS")
            
            model.active_feats = torch.tensor(partialed_indices).to(self.device)
            reg_param = model.reg_param
            entropy_param = model.entropy_param
            penalized = model.penalized
            model.reg_param = 0
            model.entropy_param = 0
            model.penalized = False
            
            # Get data loaders only for partialed feats
            train_loader = self.get_data_loader(X_train[:, partialed_indices], y_train, **train_loader_args)
            val_loader = self.get_data_loader(X_val[:, partialed_indices], y_val, shuffle=False, **val_loader_args)
            
            model.train_(
                partial(self.train_epoch_mains, on_partialed_set=True), 
                partial(self.test_epoch_mains, on_partialed_set=True), 
                train_loader, 
                val_loader, 
                optimizer, 
                self.max_epochs,
                mains=True,
                verbose=True,
                show_feat_count=True
            )
            
            model.active_feats = torch.arange(X.shape[1]).to(self.device)
            model.reg_param = reg_param
            model.entropy_param = entropy_param
            model.penalized = penalized
            
            # Go back to full loaders
            train_loader = self.get_data_loader(X_train, y_train, **train_loader_args)
            val_loader = self.get_data_loader(X_val, y_val, shuffle=False, **val_loader_args)
        
        best_val_loss = model.train_(
            partial(self.train_epoch_mains, partialed_indices=partialed_indices), 
            self.test_epoch_mains, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs,
            mains=True,
            verbose=True,
            show_feat_count=True
        )
        
        self.logger.info(f"Number of main features selected: {len(model.active_feats)}")
        
        
        if not self.fit_pairs:
            self.selected_feats = self.feature_names_in_[model.active_feats.cpu().numpy()].tolist()
            val_score = self._score(pd.DataFrame(X_val, columns=self.feature_names_in_), y_val, y_train, model=model)
            # val_score = {"Test": 0}
            return best_val_loss, val_score, model

        selected_feat_indices = model.active_feats.cpu().numpy()
        selected_pair_indices = list(combinations(selected_feat_indices, 2))

        X_train_interactions = X_train[:, selected_pair_indices]
        X_val_interactions = X_val[:, selected_pair_indices]

        model.freeze_main_effects()
        model.selected_pair_indices = selected_pair_indices
        model.pairs_list = torch.LongTensor(selected_pair_indices).to(self.device)
        model.n_pairs = len(selected_pair_indices)
        model.init_pairs_params(model.n_pairs)
        model.active_pairs = torch.arange(model.n_pairs).to(self.device)
        model.to(self.device)
        
        train_loader = self.get_data_loader(X_train, y_train, pairs=X_train_interactions, **train_loader_args)
        val_loader = self.get_data_loader(X_val, y_val, pairs=X_val_interactions, shuffle=False, **val_loader_args)
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=self.learning_rate / 5
        )
        
        best_val_loss = model.train_(
            self.train_epoch_pairs, 
            self.test_epoch_pairs, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs,
            mains=False,
            verbose=True,
            show_pair_count=True
        )
        # val_score = self.score(pd.DataFrame(X_val, columns=self.feature_names_in_), y_val, y_train, model=model)
        val_score = {}
        
        self.selected_feats = self.feature_names_in_[model.active_feats.cpu().numpy()].tolist()
        self.selected_pairs = [
            [self.feature_names_in_[pair[0]], self.feature_names_in_[pair[1]]]
            for pair in model.pairs_list[model.active_pairs].cpu().numpy()
        ]
        self.pairs_list = [selected_pair_indices[i] for i in model.active_pairs.cpu().numpy()]
        
        self.logger.info(f"Number of interaction features selected: {len(self.selected_pairs)}")
        
        return best_val_loss, val_score, model
    
    def get_regularization_path(self, X, y, init_reg_param, partialed_feats=None):
        
        original_fit_pairs = self.fit_pairs
        
        self.reg_param = init_reg_param
        self.fit_pairs = False
        self.gamma = None # will be set in _select_features
        self.pair_gamma = None
        self.pair_reg_param = 0
        self.entropy_param = 0
        
        n_feat_selected = np.inf
        losses = []
        num_feats = []
        feats = {}
        scores = []
        model = None
        
        while n_feat_selected > 1:
            if model is None:
                self.logger.info(f"SELECTING FEATURES WITH REG PARAM: {self.reg_param}")
            else:
                self.logger.info(f"SELECTING FEATURES WITH REG PARAM: {model.reg_param}")
            
            # best_val_loss, model = self._select_features(X, y, partialed_feats=partialed_feats, model=model)
            best_val_loss, val_score, _ = self._select_features(X, y, partialed_feats=partialed_feats)
            n_feat_selected = len(self.selected_feats)
            num_feats.append(n_feat_selected)
            losses.append(best_val_loss)
            feats[n_feat_selected] = self.selected_feats
            scores.append(list(val_score.values())[0])
            
            self.logger.debug(f"Number of features selected: {n_feat_selected}")
            self.logger.debug(f"Selected features: {self.selected_feats}")
            self.logger.debug(f"Validation loss: {best_val_loss}")
            self.logger.debug(f"Validation score: {val_score}")
            
            # model.reg_param = model.reg_param * 2
            self.reg_param = self.reg_param * 2
            del self.selected_feats
            
        self.fit_pairs = original_fit_pairs
        
        val_score_name = list(val_score.keys())[0]
            
        return feats, pd.DataFrame({"num_feats": num_feats, "val_loss": losses, f"val_{val_score_name}": scores})
    
    def _select_pairs(
        self, 
        model, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        num_pairs=10,
        train_loader_args={},
        val_loader_args={},
    ):
        
        # Set random seed if one is provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        pair_reg_param = 0.01
        pair_gamma = min(((X_train.shape[0] / self.batch_size) / 250) / (self.n_hidden / 16), 1.0)
        
        self.logger.debug("SELECTING PAIRS")
        self.logger.debug(f"WITH PAIR REG PARAM: {pair_reg_param}")
        
        pairs_list = list(combinations(range(X_train.shape[1]), 2))
        
        pair_scores = self._score_pairs(
            model, X_train, y_train, X_val, y_val, pair_reg_param, pair_gamma, train_loader_args, val_loader_args
        )
        
        # Check number of non-zeros in pair scores
        while np.sum(pair_scores > 0) < num_pairs:
            pair_reg_param /= 2
            self.logger.debug(f"CHANGING PAIR REG PARAM TO {pair_reg_param}")
            pair_scores = self._score_pairs(
                model, X_train, y_train, X_val, y_val, pair_reg_param, pair_gamma
            )
        
        # Select the pairs with the largest k scores
        top_pairs = np.argsort(pair_scores)[::-1][:num_pairs]
        
        self.pairs_list = [pairs_list[i] for i in top_pairs]
        
        self.selected_pairs = [
            [self.feature_names_in_[pair[0]], self.feature_names_in_[pair[1]]]
            for pair in self.pairs_list
        ]
        
        return
        
    def _score_pairs(
        self, 
        model, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        pair_reg_param, 
        pair_gamma,
        train_loader_args={},
        val_loader_args={},
    ):
        
        model = copy.deepcopy(model)
        
        model.freeze_main_effects()
        
        # Set parameters so that pairs will be regularized
        model.penalized = True
        model.pair_reg_param = pair_reg_param
        model.pair_gamma = pair_gamma
        
        # Start initially with all pairs
        pairs_list = list(combinations(range(X_train.shape[1]), 2))
        
        X_train_interactions = X_train.values[:, pairs_list]
        X_val_interactions = X_val.values[:, pairs_list]
        
        train_loader = self.get_data_loader(X_train, y_train, pairs=X_train_interactions, **train_loader_args)
        val_loader = self.get_data_loader(X_val, y_val, pairs=X_val_interactions, shuffle=False, **val_loader_args)
        
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
            self.max_epochs,
            mains=False,
            # verbose=True,
            # show_pair_count=True
        )
        
        # Get the pair scores
        pair_scores = model.get_smooth_z_pairs().detach().cpu().numpy()
        
        return pair_scores
    
    def fit(self, X, y, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for training.

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable or labels.

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        self.logger.debug("STARTING FIT...")
        
        if hasattr(self, 'selected_feats'):
            if hasattr(self, 'selected_pairs') and self.fit_pairs:
                self.logger.warning("Found selected features and pairs. Using only those features and pairs.")
            else:
                self.logger.warning("Found selected features. Using only those features.")
        
        self.feature_names_in_ = X.columns
        
        # First discretize the data
        X_discrete = self._discretize_data(X)
        
        # Fit several models, one for each validation split
        for i in range(self.n_val_splits):
            self.logger.info(f"SPlIT: {i}")

            # Split the data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X_discrete, y, test_size=self.validation_size, random_state=self.random_state+i)
            
            # Fit to this split
            model = self._fit_one_split(X_train, y_train, X_val, y_val, random_state=self.random_state+i, partialed_feats=partialed_feats)
            
            self.models.append(model)
            
        self._compute_bin_scores()
            
        return
    
    def _predict(self, X_test_discrete, models):
        self.logger.debug("STARTING _PREDICT...")
        
        # # If selected_feats is set, only use those features
        # if hasattr(self, 'selected_feats'):
        #     X_test_discrete = X_test_discrete[self.selected_feats]
        
        if hasattr(self, 'selected_feats'):
            if self.fit_pairs:
                pairs_list = [
                    [X_test_discrete.columns.get_loc(feat1), X_test_discrete.columns.get_loc(feat2)] for feat1, feat2 in self.selected_pairs
                ]    
        elif self.fit_pairs:
            pairs_list = self.pairs_list
            
        # Make placeholder y_test
        y_test = np.zeros(X_test_discrete.shape[0])
        
        self.logger.debug("GETTING THE TEST PREDICTIONS")
        if not self.fit_pairs:
            test_loader = self.get_data_loader(X_test_discrete, y_test, shuffle=False)
            test_preds = np.zeros((len(models), X_test_discrete.shape[0]))
            for i, model in enumerate(models):
                _, model_preds = self.test_epoch_mains(model, test_loader)
                test_preds[i, ...] = model_preds.cpu().numpy()
        
        else:
            X_test_interactions = X_test_discrete.values[:, pairs_list]
            test_loader = self.get_data_loader(X_test_discrete, y_test, pairs=X_test_interactions, shuffle=False)
            
            test_preds = np.zeros((len(models), X_test_discrete.shape[0]))
            for i, model in enumerate(models):
                _, model_preds = self.test_epoch_pairs(model, test_loader)
                test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
        
    
    def predict(self, X_test):
        """
        Predict labels using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        X_test_discrete = self._discretize_data(X_test)
                
        if hasattr(self, 'selected_feats'):
            self.logger.debug("Found selected features. Using only those features.")
            X_test_discrete = X_test_discrete[self.selected_feats]
        
        return self._predict(X_test_discrete, self.models)
    
    def _score(self, X, y, y_train=None):
        raise NotImplementedError("Scoring is only implemented for child classes.")
    
    def get_feature_importances(self, missing_bin="include"):
        """
        Get the feature importance scores for all features in the model.
        
        Parameters
        ----------
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the feature importance scores for each feature.
        """
        
        importances = []
        for split, model in enumerate(self.models):
            
            for i, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[i].squeeze()
                bin_counts = model.bin_counts[i]
                
                if missing_bin != "include" and self.has_missing_bin[i]:
                    missing_bin_score = feat_bin_scores[0]
                    feat_bin_scores = feat_bin_scores[1:]
                    bin_counts = bin_counts[1:]
                
                feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(bin_counts)) / np.sum(bin_counts)
                
                if missing_bin == "stratify":
                    if self.has_missing_bin[i]:
                        missing_importance = np.abs(missing_bin_score)
                    else:
                        missing_importance = np.nan
                    importances.append([split, col, feat_importance, "observed", len(feat_bin_scores)])
                    importances.append([split, col, missing_importance, "missing", 1])
                else:
                    importances.append([split, col, feat_importance, len(feat_bin_scores)])
            
            if model.fit_pairs:
                pairs_list = model.pairs_list.cpu().numpy()
                for i, pair in enumerate(pairs_list):
                    # To do: incorporate ignore missing bin here
                    pair_bin_scores = model.pair_bin_scores[i].squeeze()
                    pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[i])) / np.sum(model.pair_bin_counts[i])
                    
                    if missing_bin == "stratify":
                        importances.append([split, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, "observed", len(pair_bin_scores)])
                    else:
                        importances.append([split, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])
        
        if missing_bin == "stratify":
            self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "bin_type", "n_bins"])
        else:
            self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    
    def plot_feature_importances(self, n_features=10, missing_bin="include"):
        """
        Plot a bar plot with the importance score for the top k features.

        Parameters
        ----------
        n_features : int, default=10
            Number of features to plot.
            
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
        """
        
        plt.figure(figsize=(8 if missing_bin == "stratify" else 6, 4))
        
        feat_imps = self.get_feature_importances(missing_bin=missing_bin)

        if missing_bin == "stratify":
            feat_means = feat_imps.groupby(["feature", "bin_type"])["importance"].mean().reset_index()
            feat_means = feat_means.pivot(index="feature", columns="bin_type", values="importance").reset_index()
            feat_means = feat_means.sort_values(by="observed", ascending=True).tail(n_features)
            
            if self.n_val_splits == 1:
                plt.barh(feat_means["feature"], feat_means["observed"], color=sns.color_palette()[0])
                plt.barh(feat_means["feature"], -1*feat_means["missing"], color=sns.color_palette()[1])
            else:
                feat_sems = feat_imps.groupby(["feature", "bin_type"])["importance"].agg(["sem"]).reset_index()
                feat_sems = feat_sems.pivot(index="feature", columns="bin_type", values="sem").reset_index()
                feat_imps = feat_means.merge(feat_sems, on="feature", suffixes=("", "_sem"))
                plt.barh(
                    y=feat_imps["feature"],
                    width=feat_imps["observed"],
                    xerr=1.96 * feat_imps["observed_sem"],
                )
                plt.barh(
                    y=feat_imps["feature"],
                    width=-1*feat_imps["missing"],
                    xerr=1.96 * feat_imps["missing_sem"],
                    color=sns.color_palette()[1]
                )
                
            plt.ylabel("")

            # Add legend
            plt.legend(["Observed", "Missing"], title="Importance Type")
            
            ax = plt.gca()
            xticks = ax.get_xticks()
            xticklabels = [round(abs(x), 2) for x in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            
        else:
        
            top_k_features = feat_imps.groupby("feature")["importance"].agg(["mean", "sem"])
            top_k_features = top_k_features.sort_values("mean", ascending=True).tail(n_features)

            if self.n_val_splits == 1:
                plt.barh(top_k_features.index, top_k_features["mean"])
            else:
                plt.barh(top_k_features.index, top_k_features["mean"], xerr=1.96*top_k_features["sem"])
            
        plt.xlabel("Importance")
        plt.title(f"Feature Importances ({missing_bin})")
        plt.tight_layout()
    
    def get_shape_function(self, feature_name):
        """
        Get the shape function data, i.e. the bin scores, for a given feature.
        
        Parameters
        ----------
        feature_name : str
            The name of the feature.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the bin scores for the feature.
        """
        
        is_cat_col = self.feature_dtypes[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
        
        dfs = []
        for i, model in enumerate(self.models):
        
            col_index = model.feature_names_in_.get_loc(feature_name)
            feat_bin_scores = model.bin_scores[col_index].squeeze()
            
            
            if not is_cat_col:
                
                feat_bin_values = np.concatenate([
                    [np.nan],
                    [model.feature_bins[col_index].min() - 0.01],
                    model.feature_bins[col_index]
                ])
                
            else:
                
                feat_bin_values = model.feature_bins[col_index]
                
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "bin": feat_bin_values,
                    "score": feat_bin_scores,
                    "count": model.bin_counts[col_index],
                    "split": i
                })
            )
            
        df = pd.concat(dfs)
            
        return df.reset_index()
    
    def plot_shape_function(self, feature_names, plot_missing_bin=False, yaxis_label="Contribution", axes=None):
        """
        Plot the shape function for given feature(s).
        
        Parameters
        ----------
        feature_names : str or list of str
            The name of the feature(s) to plot.
            
        plot_missing_bin : bool, default=False
            Whether to plot the missing bin.
            Only applicable for continuous features.
        """
        
        def set_font_sizes(axes):
            LABEL_FONT_SIZE = 14 * (1.5 if plot_missing_bin else 1)
            TICK_FONT_SIZE = 12 * (1.5 if plot_missing_bin else 1)
            for ax in axes:
                ax.xaxis.label.set_fontsize(LABEL_FONT_SIZE)
                ax.yaxis.label.set_fontsize(LABEL_FONT_SIZE)
                ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            
        is_cat_cols = [self.feature_dtypes[self.feature_names_in_.get_loc(f)] != 'continuous' for f in feature_names]

        num_axes = sum([2 if not c and plot_missing_bin else 1 for _, c in zip(feature_names, is_cat_cols)])

        if axes is None:
            if plot_missing_bin:
            
                fig = plt.figure(figsize=(4*num_axes, 4))

                gs = gridspec.GridSpec(1, 2*len(feature_names) + len(feature_names)-1, width_ratios=[10, 1, 4] * (len(feature_names)-1) + [10, 1])  # Add an extra column for the space
                def generate_indices(n_times):
                    indices = []
                    for i in range(n_times):
                        indices.extend([i*3, i*3+1])
                    return indices

                axes = [fig.add_subplot(gs[i]) for i in generate_indices(len(feature_names))]
                
            else:
                fig, axes = plt.subplots(1, num_axes, figsize=(4*num_axes, 4))
                if num_axes == 1:
                    axes = [axes]
                    
            set_font_sizes(axes)

        ax_idx = 0

        for feature_name in feature_names:
            shape_data = self.get_shape_function(feature_name)
            is_cat_col = self.feature_dtypes[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
            
            if not is_cat_col:
                if not plot_missing_bin:
                    line_plot_data = shape_data[shape_data["bin"].notna()].groupby("bin")["score"]
                    axes[ax_idx].plot(line_plot_data.mean().index, line_plot_data.mean(), drawstyle='steps-post')
                    axes[ax_idx].fill_between(
                        line_plot_data.mean().index,
                        line_plot_data.mean() - 1.96 * line_plot_data.sem(),
                        line_plot_data.mean() + 1.96 * line_plot_data.sem(),
                        alpha=0.3,
                        step='post'
                    )
                    axes[ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                    if ax_idx == 0:
                        axes[ax_idx].set_ylabel("\n".join(textwrap.wrap(yaxis_label, width=25)))
                    else:
                        axes[ax_idx].set_ylabel("")
                    ax_idx += 1
                else:
                        
                    # Make the line plot
                    line_plot_data = shape_data[shape_data["bin"].notna()].groupby("bin")["score"]
                    axes[ax_idx].plot(line_plot_data.mean().index, line_plot_data.mean(), drawstyle='steps-post')
                    axes[ax_idx].fill_between(
                        line_plot_data.mean().index,
                        line_plot_data.mean() - 1.96 * line_plot_data.sem(),
                        line_plot_data.mean() + 1.96 * line_plot_data.sem(),
                        alpha=0.3,
                        step='post'
                    )
                    axes[ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                    if ax_idx == 0:
                        axes[ax_idx].set_ylabel("\n".join(textwrap.wrap(yaxis_label, width=25)))
                    else:
                        axes[ax_idx].set_ylabel("")
                    
                    ax_idx += 1

                    # Make the bar plot
                    bar_data = shape_data[shape_data["bin"].isna()]
                    axes[ax_idx].bar("NA", bar_data["score"].mean())
                    axes[ax_idx].errorbar("NA", bar_data["score"].mean(), yerr=1.96*bar_data["score"].sem(), fmt='none', color='black')
                    axes[ax_idx].set_ylim(axes[0].get_ylim())
                    axes[ax_idx].set_yticklabels([])
                    axes[ax_idx].set_xlim(-0.5, 0.5)
                    
                    ax_idx += 1
                    
            else:
                sns.barplot(x="bin", y="score", data=shape_data, errorbar=('ci', 95), ax=axes[ax_idx])
                axes[ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                axes[ax_idx].tick_params(axis='x', rotation=45)
                for label in axes[ax_idx].get_xticklabels():
                    label.set_ha('right')
                if ax_idx == 0:
                    axes[ax_idx].set_ylabel("\n".join(textwrap.wrap(yaxis_label, width=25)))
                else:
                    axes[ax_idx].set_ylabel("")
                ax_idx += 1
        
        if not plot_missing_bin:
            plt.tight_layout()
        
    def _make_grid(self, x, y):
        grid = np.meshgrid(x, y, indexing="ij")
        grid = np.stack(grid, axis=-1)
        return grid.reshape(-1, 2)
        
    def get_pair_shape_function(self, feat1_name, feat2_name):
        """
        Get the shape function data for an interaction affect.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
        
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the shape function data for the interaction effect.
        """
        
        bin_scores = []
        bin_values = []
        for i, model in enumerate(self.models):
        
            col1_index = model.feature_names_in_.get_loc(feat1_name)
            col2_index = model.feature_names_in_.get_loc(feat2_name)
            
            pair_index = [
                i for i, p in enumerate(model.pairs_list) if (p[0] == col1_index and p[1] == col2_index) or (p[0] == col2_index and p[1] == col1_index)
            ]
            
            if len(pair_index) == 0:
                raise ValueError("Pair not found")
            
            pair_index = pair_index[0]
            
            pair_bin_scores = model.pair_bin_scores[pair_index].squeeze()
            
            
            # Remove scores associated with missing bins
            grid = self._make_grid(
                np.arange(0, self.feature_sizes[col1_index]), 
                np.arange(0, self.feature_sizes[col2_index])
            )
            if self.has_missing_bin[col1_index] and not self.has_missing_bin[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 0] > 0]
            elif not self.has_missing_bin[col1_index] and self.has_missing_bin[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 1] > 0]
            else:
                pair_bin_scores = pair_bin_scores[grid.prod(axis=1) > 0]
            
            # Get bin values
            if self.feature_dtypes[col1_index] != 'continuous':
                # feat1_bin_values = np.arange(-1, len(model.feature_bins[col1_index])+1)
                feat1_bin_values = model.feature_bins[col1_index]
            else:
                feat1_bin_values = np.concatenate([
                    [model.feature_bins[col1_index].min() - 0.01],
                    model.feature_bins[col1_index]
                ])
                
            if self.feature_dtypes[col2_index] != 'continuous':
                # feat2_bin_values = np.arange(-1, len(model.feature_bins[col2_index])+1)
                feat2_bin_values = model.feature_bins[col2_index]
            else:
                feat2_bin_values = np.concatenate([
                    [model.feature_bins[col2_index].min() - 0.01],
                    model.feature_bins[col2_index]
                ])

            pair_bin_values = self._make_grid(feat1_bin_values, feat2_bin_values)
            
            bin_scores.append(pair_bin_scores)
            bin_values.append(pair_bin_values)
                
        # return bin_scores, bin_values
        
        pair_scores = np.mean(np.stack(bin_scores), axis=0)
        pair_values = bin_values[0]

        pair_data_dnamite = pd.DataFrame({
            feat1_name: pair_values[:, 0],
            feat2_name: pair_values[:, 1],
            "score": pair_scores
        })

        pair_data_dnamite = pair_data_dnamite.pivot(
            index=feat1_name, 
            columns=feat2_name, 
            values="score"
        )

        # Round the values in the columns and index
        # Round the index and columns to 3 decimal places and reassign
        pair_data_dnamite.index = pair_data_dnamite.index.to_series().round(3)
        pair_data_dnamite.columns = pair_data_dnamite.columns.to_series().round(3)
        
        return pair_data_dnamite
    
    def plot_pair_shape_function(self, feat1_name, feat2_name):
        """
        Plot a heatmap for an interaction shape function.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
            
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        """
        
        plt.figure(figsize=(4, 4))
        
        pair_data_dnamite = self.get_pair_shape_function(feat1_name, feat2_name)

        sns.heatmap(pair_data_dnamite)
        
        plt.tight_layout()
      
class DNAMiteRegressor(BaseDNAMiteModel):
    """
    DNAMiteRegressor is a model for regression using the DNAMite architecture.

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_embed : int, optional (default=32)
        The size of the embedding layer.
    n_hidden : int, optional (default=32)
        The number of hidden units in the hidden layers.
    max_bins : int, optional (default=32)
        The maximum number of bins for discretizing continuous features.
    min_samples_per_bin : int, default=None
        Minimum number of samples required in each bin.
        Default is None which sets it to min(n_train_samples / 100, 50).
    validation_size : float, optional (default=0.2)
        The proportion of the dataset to include in the validation split.
    n_val_splits : int, optional (default=5)
        The number of validation splits for cross-validation.
    learning_rate : float, optional (default=5e-4)
        The learning rate for the optimizer.
    max_epochs : int, optional (default=100)
        The maximum number of epochs for training.
    batch_size : int, optional (default=128)
        The batch size for training.
    device : str, optional (default="cpu")
        The device to run the model on ("cpu" or "cuda").
    pairs_list : list of tuple[int, int], optional
        List of pairs to use in the model.
        Each entry should be a pair of indices corresponding to the features.
        Should only be set if manual pair selection is desired.
        Set num_pairs instead for automatic pair selection.
    kernel_size : int, optional (default=5)
        The size of the kernel in convolutional layers for single features.
    kernel_weight : float, optional (default=3)
        The weight of the kernel for single feature convolutional layers.
    pair_kernel_size : int, optional (default=3)
        The size of the kernel for pairwise convolutional layers.
    pair_kernel_weight : float, optional (default=3)
        The weight of the kernel for pairwise convolutional layers.
    monotone_constraints : list or None, optional (default=None)
        The monotonic constraints for the features.
        0 indicates no constraint, 1 indicates increasing, -1 indicates decreasing.
        None means no constraints.
    num_pairs : int, default=0
        Number of pairwise interactions to use in the model.
    verbosity : int, default=0
        Level of verbosity for logging.
        0: Warning, 1: Info, 2: Debug
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self,
                 n_features, 
                 n_embed=32, 
                 n_hidden=32, 
                 max_bins=32, 
                 min_samples_per_bin=None,
                 validation_size=0.2, 
                 n_val_splits=5, 
                 learning_rate=5e-4, 
                 max_epochs=100, 
                 batch_size=128, 
                 device="cpu", 
                 pairs_list=None,
                 kernel_size=5, 
                 kernel_weight=3, 
                 pair_kernel_size=3, 
                 pair_kernel_weight=3, 
                 monotone_constraints=None,
                 num_pairs=0,
                 verbosity=0,
                 random_state=None
    ):
        super().__init__(
            n_features=n_features,
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=1,
            max_bins=max_bins,
            min_samples_per_bin=min_samples_per_bin,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            pairs_list=pairs_list,
            kernel_size=kernel_size,
            kernel_weight=kernel_weight,
            pair_kernel_size=pair_kernel_size,
            pair_kernel_weight=pair_kernel_weight,
            monotone_constraints=monotone_constraints,
            num_pairs=num_pairs,
            verbosity=verbosity,
            random_state=random_state
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None, on_partialed_set=False):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, labels in train_loader:

            X_main, labels = X_main.to(self.device), labels.to(self.device)

            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main).squeeze(-1)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main, on_partialed_set=on_partialed_set).squeeze(-1)
            y_pred += y_pred_init
            
            loss = F.mse_loss(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        # Reset active feats
        model.active_feats = active_feats
            
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader, on_partialed_set=False):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, labels in test_loader:

                X_main, labels = X_main.to(self.device), labels.to(self.device)

                y_pred = model(mains=X_main, on_partialed_set=on_partialed_set).squeeze(-1)
        
                preds.append(y_pred.detach())
            
                loss = F.mse_loss(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=True)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def train_epoch_pairs(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, X_pairs, labels in train_loader:

            X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                main_preds = model(mains=X_main).squeeze(-1)
            
            pair_preds = model(mains=None, pairs=X_pairs).squeeze(-1)
            
            y_pred = main_preds + pair_preds
            
            loss = F.mse_loss(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=False)

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
            for X_main, X_pairs, labels in test_loader:

                X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

                main_preds = model(mains=X_main).squeeze(-1)
                pair_preds = model(pairs=X_pairs).squeeze(-1)
                
                y_pred = main_preds + pair_preds
        
                preds.append(y_pred.detach())
            
                loss = F.mse_loss(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=False)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def fit(self, X, y, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for training. 
            Missing values should be encoded as np.nan.
            Categorical features will automatically be detected as all columns with dtype "object" or "category".

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The labels, should be floats in (-inf, inf).

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        return super().fit(X, y, partialed_feats=partialed_feats)
    
    def get_feature_importances(self, missing_bin="include"):
        """
        Get the feature importance scores for all features in the model.
        
        Parameters
        ----------
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the feature importance scores for each feature.
        """
        
        return super().get_feature_importances(missing_bin=missing_bin)
    
    def get_pair_shape_function(self, feat1_name, feat2_name):
        """
        Get the shape function data for an interaction affect.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
        
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the shape function data for the interaction effect.
        """
        
        return super().get_pair_shape_function(feat1_name, feat2_name)
    
    def get_shape_function(self, feature_name):
        """
        Get the shape function data, i.e. the bin scores, for a given feature.
        
        Parameters
        ----------
        feature_name : str
            The name of the feature.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the bin scores for the feature.
        """
        
        return super().get_shape_function(feature_name)
    
    def plot_feature_importances(self, n_features=10, missing_bin="include"):
        """
        Plot a bar plot with the importance score for the top k features.
        
        Parameters
        ----------
        n_features : int, default=10
            Number of features to plot.
            
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
        """
        
        return super().plot_feature_importances(n_features, missing_bin)
    
    def plot_pair_shape_function(self, feat1_name, feat2_name):
        """
        Plot a heatmap for an interaction shape function.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
            
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        """
        
        return super().plot_pair_shape_function(feat1_name, feat2_name)
    
    def plot_shape_function(self, feature_names, plot_missing_bin=False, axes=None):
        """
        Plot the shape function for given feature(s).
        
        Parameters
        ----------
        feature_names : str or list of str
            The name of the feature(s) to plot.
            
        plot_missing_bin : bool, default=False
            Whether to plot the missing bin.
            Only applicable for continuous features.
        """
        
        return super().plot_shape_function(feature_names, plot_missing_bin, yaxis_label="Contribution to Prediction", axes=axes)
    
    def predict(self, X_test):
        """
        Predict labels using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        
        return super().predict(X_test)
    
    def select_features(
        self, 
        X, 
        y, 
        reg_param,
        select_pairs=False, 
        partialed_feats=None,
        gamma=None,
        pair_gamma=None, 
        pair_reg_param=0, 
        entropy_param=0, 
    ):
        """
        Perform feature selection. Selected features and pairs will be stored in model.selected_feats
        and model.selected_pairs, respectively. Should be called before fit if feature selection is desired.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for the model.   
        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable. 
        reg_param : float
            Regularization parameter for feature-level regularization.
        select_pairs : bool, default=False
            Whether to select feature pairs in addition to individual features.
        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        gamma : float, default=1
            Regularization or scaling parameter; purpose depends on specific use in the model.
        pair_gamma : float or None, default=None
            Regularization or scaling parameter for feature pairs.
            If None, defaults to gamma.
        pair_reg_param : float, default=0
            Regularization parameter for feature pairs.
        entropy_param : float, default=0
            Entropy parameter to control the diversity or uncertainty.
        """
        
        return super().select_features(
            X, 
            y, 
            select_pairs=select_pairs, 
            partialed_feats=partialed_feats,
            gamma=gamma,
            pair_gamma=pair_gamma,
            reg_param=reg_param,
            pair_reg_param=pair_reg_param,
            entropy_param=entropy_param
        )
    
    def _score(self, X, y, y_train=None, model=None):
        from sklearn.metrics import mean_squared_error
        preds = self._predict(X, models=[model])
                
        return {"RMSE": np.sqrt(mean_squared_error(y, preds))}
    

class DNAMiteBinaryClassifier(BaseDNAMiteModel):
    """
    DNAMiteClassifier is a model for binary classification using the DNAMite architecture.

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_embed : int, optional (default=32)
        The size of the embedding layer.
    n_hidden : int, optional (default=32)
        The number of hidden units in the hidden layers.
    max_bins : int, optional (default=32)
        The maximum number of bins for discretizing continuous features.
    min_samples_per_bin : int, default=None
        Minimum number of samples required in each bin.
        Default is None which sets it to min(n_train_samples / 100, 50).
    validation_size : float, optional (default=0.2)
        The proportion of the dataset to include in the validation split.
    n_val_splits : int, optional (default=5)
        The number of validation splits for cross-validation.
    learning_rate : float, optional (default=5e-4)
        The learning rate for the optimizer.
    max_epochs : int, optional (default=100)
        The maximum number of epochs for training.
    batch_size : int, optional (default=128)
        The batch size for training.
    device : str, optional (default="cpu")
        The device to run the model on ("cpu" or "cuda").
    pairs_list : list of tuple[int, int], optional
        List of pairs to use in the model.
        Each entry should be a pair of indices corresponding to the features.
        Should only be set if manual pair selection is desired.
        Set num_pairs instead for automatic pair selection.
    kernel_size : int, optional (default=5)
        The size of the kernel in convolutional layers for single features.
    kernel_weight : float, optional (default=3)
        The weight of the kernel for single feature convolutional layers.
    pair_kernel_size : int, optional (default=3)
        The size of the kernel for pairwise convolutional layers.
    pair_kernel_weight : float, optional (default=3)
        The weight of the kernel for pairwise convolutional layers.
    monotone_constraints : list or None, optional (default=None)
        The monotonic constraints for the features.
        0 indicates no constraint, 1 indicates increasing, -1 indicates decreasing.
        None means no constraints.
    num_pairs : int, default=0
        Number of pairwise interactions to use in the model.
    verbosity : int, default=0
        Level of verbosity for logging.
        0: Warning, 1: Info, 2: Debug
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    
    def __init__(self,
                 n_features, 
                 n_embed=32, 
                 n_hidden=32, 
                 max_bins=32, 
                 min_samples_per_bin=None,
                 validation_size=0.2, 
                 n_val_splits=5, 
                 learning_rate=5e-4, 
                 max_epochs=100, 
                 batch_size=128, 
                 device="cpu", 
                 pairs_list=None,
                 kernel_size=5, 
                 kernel_weight=3, 
                 pair_kernel_size=3, 
                 pair_kernel_weight=3, 
                 monotone_constraints=None,
                 num_pairs=0,
                 verbosity=0,
                 random_state=None
    ):
        super().__init__(
            n_features=n_features,
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=1,
            max_bins=max_bins,
            min_samples_per_bin=min_samples_per_bin,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            pairs_list=pairs_list,
            kernel_size=kernel_size,
            kernel_weight=kernel_weight,
            pair_kernel_size=pair_kernel_size,
            pair_kernel_weight=pair_kernel_weight,
            monotone_constraints=monotone_constraints,
            num_pairs=num_pairs,
            verbosity=verbosity,
            random_state=random_state
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None, on_partialed_set=False):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, labels in train_loader:

            X_main, labels = X_main.to(self.device), labels.to(self.device)
            
            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main).squeeze(-1)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main, on_partialed_set=on_partialed_set).squeeze(-1)
            y_pred += y_pred_init
            
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        # Reset active feats
        model.active_feats = active_feats
            
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader, partialed_indices=None, on_partialed_set=False):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, labels in test_loader:

                X_main, labels = X_main.to(self.device), labels.to(self.device)

                y_pred = model(mains=X_main, on_partialed_set=on_partialed_set).squeeze(-1)
        
                preds.append(y_pred.detach())
            
                loss = F.binary_cross_entropy_with_logits(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=True)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def train_epoch_pairs(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, X_pairs, labels in train_loader:

            X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                main_preds = model(mains=X_main).squeeze(-1)
            
            pair_preds = model(mains=None, pairs=X_pairs).squeeze(-1)
            
            y_pred = main_preds + pair_preds
            
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=False)

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
            for X_main, X_pairs, labels in test_loader:

                X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

                main_preds = model(mains=X_main).squeeze(-1)
                pair_preds = model(pairs=X_pairs).squeeze(-1)
                
                y_pred = main_preds + pair_preds
        
                preds.append(y_pred.detach())
            
                loss = F.binary_cross_entropy_with_logits(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=False)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def fit(self, X, y, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for training. 
            Missing values should be encoded as np.nan.
            Categorical features will automatically be detected as all columns with dtype "object" or "category".

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The labels, should be label encoded as 0 and 1.

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        return super().fit(X, y, partialed_feats=partialed_feats)
    
    def get_feature_importances(self, missing_bin="include"):
        """
        Get the feature importance scores for all features in the model.
        
        Parameters
        ----------
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the feature importance scores for each feature.
        """
        
        return super().get_feature_importances(missing_bin=missing_bin)
    
    def get_pair_shape_function(self, feat1_name, feat2_name):
        """
        Get the shape function data for an interaction affect.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
        
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the shape function data for the interaction effect.
        """
        
        return super().get_pair_shape_function(feat1_name, feat2_name)
    
    def get_shape_function(self, feature_name):
        """
        Get the shape function data, i.e. the bin scores, for a given feature.
        
        Parameters
        ----------
        feature_name : str
            The name of the feature.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the bin scores for the feature.
        """
        
        return super().get_shape_function(feature_name)
    
    def plot_feature_importances(self, n_features=10, missing_bin="include"):
        """
        Plot a bar plot with the importance score for the top k features.
        
        Parameters
        ----------
        n_features : int, default=10
            Number of features to plot.
            
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
        """
        
        return super().plot_feature_importances(n_features, missing_bin=missing_bin)
    
    def plot_pair_shape_function(self, feat1_name, feat2_name):
        """
        Plot a heatmap for an interaction shape function.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
            
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        """
        
        return super().plot_pair_shape_function(feat1_name, feat2_name)
    
    def plot_shape_function(self, feature_names, plot_missing_bin=False, axes=None):
        """
        Plot the shape function for given feature(s).
        
        Parameters
        ----------
        feature_names : str or list of str
            The name of the feature(s) to plot.
            
        plot_missing_bin : bool, default=False
            Whether to plot the missing bin.
            Only applicable for continuous features.
        """
        
        return super().plot_shape_function(feature_names, plot_missing_bin, yaxis_label=r"Contribution to $\log(\frac{p}{1-p})$", axes=axes)
    
    def predict(self, X_test):
        """
        Predict labels using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        
        return super().predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        
        preds = self.predict(X_test)
        return 1 / (1 + np.exp(-preds))
    
    def select_features(
        self, 
        X, 
        y, 
        reg_param,
        select_pairs=False, 
        partialed_feats=None,
        gamma=None,
        pair_gamma=None, 
        pair_reg_param=0, 
        entropy_param=0, 
    ):
        """
        Perform feature selection. Selected features and pairs will be stored in model.selected_feats
        and model.selected_pairs, respectively. Should be called before fit if feature selection is desired.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for the model.   
        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable. 
        reg_param : float
            Regularization parameter for feature-level regularization.
        select_pairs : bool, default=False
            Whether to select feature pairs in addition to individual features.
        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        gamma : float, default=1
            Regularization or scaling parameter; purpose depends on specific use in the model.
        pair_gamma : float or None, default=None
            Regularization or scaling parameter for feature pairs.
            If None, defaults to gamma.
        pair_reg_param : float, default=0
            Regularization parameter for feature pairs.
        entropy_param : float, default=0
            Entropy parameter to control the diversity or uncertainty.
        """
        
        return super().select_features(
            X, 
            y, 
            select_pairs=select_pairs, 
            partialed_feats=partialed_feats,
            gamma=gamma,
            pair_gamma=pair_gamma,
            reg_param=reg_param,
            pair_reg_param=pair_reg_param,
            entropy_param=entropy_param
        )
    
    def _score(self, X, y, y_train=None, model=None):
        from sklearn.metrics import roc_auc_score
        preds = self._predict(X, models=[model])
        preds = 1 / (1 + np.exp(-preds))
        
        return {"AUC": roc_auc_score(y, preds)}
    

class DNAMiteMulticlassClassifier(BaseDNAMiteModel):
    """
    DNAMiteClassifier is a model for multiclass classification using the DNAMite architecture.

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_classes : int
        The number of classes.
    n_embed : int, optional (default=32)
        The size of the embedding layer.
    n_hidden : int, optional (default=32)
        The number of hidden units in the hidden layers.
    max_bins : int, optional (default=32)
        The maximum number of bins for discretizing continuous features.
    min_samples_per_bin : int, default=None
        Minimum number of samples required in each bin.
        Default is None which sets it to min(n_train_samples / 100, 50).
    validation_size : float, optional (default=0.2)
        The proportion of the dataset to include in the validation split.
    n_val_splits : int, optional (default=5)
        The number of validation splits for cross-validation.
    learning_rate : float, optional (default=5e-4)
        The learning rate for the optimizer.
    max_epochs : int, optional (default=100)
        The maximum number of epochs for training.
    batch_size : int, optional (default=128)
        The batch size for training.
    device : str, optional (default="cpu")
        The device to run the model on ("cpu" or "cuda").
    pairs_list : list of tuple[int, int], optional
        List of pairs to use in the model.
        Each entry should be a pair of indices corresponding to the features.
        Should only be set if manual pair selection is desired.
        Set num_pairs instead for automatic pair selection.
    kernel_size : int, optional (default=5)
        The size of the kernel in convolutional layers for single features.
    kernel_weight : float, optional (default=3)
        The weight of the kernel for single feature convolutional layers.
    pair_kernel_size : int, optional (default=3)
        The size of the kernel for pairwise convolutional layers.
    pair_kernel_weight : float, optional (default=3)
        The weight of the kernel for pairwise convolutional layers.
    monotone_constraints : list or None, optional (default=None)
        The monotonic constraints for the features.
        0 indicates no constraint, 1 indicates increasing, -1 indicates decreasing.
        None means no constraints.
    num_pairs : int, default=0
        Number of pairwise interactions to use in the model.
    verbosity : int, default=0
        Level of verbosity for logging.
        0: Warning, 1: Info, 2: Debug
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self,
                 n_features,
                 n_classes, 
                 n_embed=32, 
                 n_hidden=32, 
                 max_bins=32, 
                 min_samples_per_bin=None,
                 validation_size=0.2, 
                 n_val_splits=5, 
                 learning_rate=5e-4, 
                 max_epochs=100, 
                 batch_size=128, 
                 device="cpu",  
                 pairs_list=None,
                 kernel_size=5, 
                 kernel_weight=3, 
                 pair_kernel_size=3, 
                 pair_kernel_weight=3, 
                 monotone_constraints=None,
                 num_pairs=0,
                 verbosity=0,
                 random_state=None
    ):
        super().__init__(
            n_features=n_features,
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=n_classes,
            max_bins=max_bins,
            min_samples_per_bin=min_samples_per_bin,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            pairs_list=pairs_list,
            kernel_size=kernel_size,
            kernel_weight=kernel_weight,
            pair_kernel_size=pair_kernel_size,
            pair_kernel_weight=pair_kernel_weight,
            monotone_constraints=monotone_constraints,
            num_pairs=num_pairs,
            verbosity=verbosity,
            random_state=random_state
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None, on_partialed_set=False):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, labels in train_loader:

            X_main, labels = X_main.to(self.device), labels.to(self.device)
            
            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main, on_partialed_set=on_partialed_set).squeeze(-1)
            y_pred += y_pred_init
            
            loss = F.cross_entropy(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        # Reset active feats
        model.active_feats = active_feats    
        
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader, on_partialed_set=False):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, labels in test_loader:

                X_main, labels = X_main.to(self.device), labels.to(self.device)

                y_pred = model(mains=X_main, on_partialed_set=on_partialed_set).squeeze(-1)
        
                preds.append(y_pred.detach())
            
                loss = F.cross_entropy(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=True)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def train_epoch_pairs(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, X_pairs, labels in train_loader:

            X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                main_preds = model(mains=X_main).squeeze(-1)
            
            pair_preds = model(mains=None, pairs=X_pairs).squeeze(-1)
            
            y_pred = main_preds + pair_preds
            
            loss = F.cross_entropy(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=False)

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
            for X_main, X_pairs, labels in test_loader:

                X_main, X_pairs, labels = X_main.to(self.device), X_pairs.to(self.device), labels.to(self.device)

                main_preds = model(mains=X_main).squeeze(-1)
                pair_preds = model(pairs=X_pairs).squeeze(-1)
                
                y_pred = main_preds + pair_preds
        
                preds.append(y_pred.detach())
            
                loss = F.cross_entropy(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=False)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    
    def get_feature_importances(self, ignore_missing_bin=False):
        
        importances = []
        for i, model in enumerate(self.models):
            
            # model.compute_intercept(model.bin_counts, ignore_missing_bin=ignore_missing_bin)
            
            importances = []
            for i, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[i].sum(axis=-1)
                bin_counts = model.bin_counts[i]
                
                if ignore_missing_bin:
                    feat_bin_scores = feat_bin_scores[1:]
                    bin_counts = bin_counts[1:]
                
                feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(bin_counts)) / np.sum(bin_counts)
                    
                importances.append([i, col, feat_importance, len(feat_bin_scores)])
            
            if self.fit_pairs:
                pairs_list = model.pairs_list.cpu().numpy()
                # model.compute_pairs_intercept(model.pair_bin_counts)
                    
                for i, pair in enumerate(pairs_list):
                    pair_bin_scores = model.pair_bin_scores[i].sum(axis=-1)
                    
                    pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[i])) / np.sum(model.pair_bin_counts[i])
                        
                    importances.append([i, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])
            
        self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    
    def get_shape_function(self, feature_name, label_id):
        
        is_cat_col = self.feature_dtypes[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
        
        dfs = []
        for i, model in enumerate(self.models):
        
            col_index = model.feature_names_in_.get_loc(feature_name)
            feat_bin_scores = model.bin_scores[col_index][:, label_id]
            
            
            if not is_cat_col:
                
                feat_bin_values = np.concatenate([
                    [np.nan],
                    [model.feature_bins[col_index].min() - 0.01],
                    model.feature_bins[col_index]
                ])
                
            else:
                
                # replace feature bins with ordinal encoded values
                # number of feature bins is number of unique categories + 1, and 
                # then one additional for the missing bin (represented by -1)
                
                feat_bin_values = np.arange(-1, len(model.feature_bins[col_index])+1)
                
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "bin": feat_bin_values,
                    "score": feat_bin_scores,
                    "count": model.bin_counts[col_index],
                    "split": i
                })
            )
            
        return pd.concat(dfs)
    
    def fit(self, X, y, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for training. 
            Missing values should be encoded as np.nan.
            Categorical features will automatically be detected as all columns with dtype "object" or "category".

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The labels, should be label encoded as 0, 1, ..., n_classes-1.

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        return super().fit(X, y, partialed_feats=partialed_feats)
    
    def get_pair_shape_function(self, feat1_name, feat2_name):
        """
        Get the shape function data for an interaction affect.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
        
        feat2_name : str
            The name of the second feature in the pair/interaction.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the shape function data for the interaction effect.
        """
        
        return super().get_pair_shape_function(feat1_name, feat2_name)
    
    def plot_feature_importances(self, n_features=10, missing_bin="include"):
        """
        Plot a bar plot with the importance score for the top k features.
        
        Parameters
        ----------
        n_features : int, default=10
            Number of features to plot.
            
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
        """
        
        return super().plot_feature_importances(n_features, missing_bin=missing_bin)
    
    def plot_pair_shape_function(self, feat1_name, feat2_name):
        """
        Plot a heatmap for an interaction shape function.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
            
        feat2_name : str
            The name of the second feature in the pair/interaction.
            
        is_feat1_cat_col : bool, default=False
            Whether the first feature is a categorical feature.
            
        is_feat2_cat_col : bool, default=False
            Whether the second feature is a categorical feature.
        
        """
        
        return super().plot_pair_shape_function(feat1_name, feat2_name)
    
    def plot_shape_function(self, feature_names, plot_missing_bin=False):
        """
        Plot the shape function for given feature(s).
        
        Parameters
        ----------
        feature_names : str or list of str
            The name of the feature(s) to plot.
            
        plot_missing_bin : bool, default=False
            Whether to plot the missing bin.
            Only applicable for continuous features.
        """
        
        return super().plot_shape_function(feature_names, plot_missing_bin, yaxis_label=r"Contribution to class logit")
    
    def predict(self, X_test):
        """
        Predict labels using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        
        return super().predict(X_test)
    
    def select_features(
        self, 
        X, 
        y, 
        reg_param,
        select_pairs=False, 
        partialed_feats=None,
        gamma=None,
        pair_gamma=None, 
        pair_reg_param=0, 
        entropy_param=0, 
    ):
        """
        Perform feature selection. Selected features and pairs will be stored in model.selected_feats
        and model.selected_pairs, respectively. Should be called before fit if feature selection is desired.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for the model.   
        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable.
        reg_param : float
            Regularization parameter for feature-level regularization. 
        select_pairs : bool, default=False
            Whether to select feature pairs in addition to individual features.
        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        gamma : float, default=1
            Regularization or scaling parameter; purpose depends on specific use in the model.
        pair_gamma : float or None, default=None
            Regularization or scaling parameter for feature pairs.
            If None, defaults to gamma.
        pair_reg_param : float, default=0
            Regularization parameter for feature pairs.
        entropy_param : float, default=0
            Entropy parameter to control the diversity or uncertainty.
        """
        
        return super().select_features(
            X, 
            y, 
            select_pairs=select_pairs, 
            partialed_feats=partialed_feats,
            gamma=gamma,
            pair_gamma=pair_gamma,
            reg_param=reg_param,
            pair_reg_param=pair_reg_param,
            entropy_param=entropy_param
        )
    
    def _score(self, X, y, y_train=None, model=None):
        from sklearn.metrics import accuracy_score
        preds = self._predict(X, models=[model])
        hard_preds = np.argmax(preds, axis=1)
        return {"ACCURACY": accuracy_score(y, hard_preds)}


class DNAMiteSurvival(BaseDNAMiteModel):
    """
    DNAMiteSurvival is a model for survival analysis using the DNAMite architecture.

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_eval_times : int, optional (default=100)
        The number of evaluation times for survival analysis.
    n_embed : int, optional (default=32)
        The size of the embedding layer.
    n_hidden : int, optional (default=32)
        The number of hidden units in the hidden layers.
    n_layers : int, optional (default=2)
        The number of layers in the model.
    max_bins : int, optional (default=32)
        The maximum number of bins for discretizing continuous features.
    validation_size : float, optional (default=0.2)
        The proportion of the dataset to include in the validation split.
    n_val_splits : int, optional (default=5)
        The number of validation splits for cross-validation.
    learning_rate : float, optional (default=5e-4)
        The learning rate for the optimizer.
    max_epochs : int, optional (default=100)
        The maximum number of epochs for training.
    min_samples_per_bin : int, default=None
        Minimum number of samples required in each bin.
        Default is None which sets it to min(n_train_samples / 100, 50).
    batch_size : int, optional (default=128)
        The batch size for training.
    device : str, optional (default="cpu")
        The device to run the model on ("cpu" or "cuda").
    pairs_list : list of tuple[int, int], optional
        List of pairs to use in the model.
        Each entry should be a pair of indices corresponding to the features.
        Should only be set if manual pair selection is desired.
        Set num_pairs instead for automatic pair selection.
    kernel_size : int, optional (default=5)
        The size of the kernel in convolutional layers.
    kernel_weight : float, optional (default=3)
        The weight of the kernel in convolutional layers.
    pair_kernel_size : int, optional (default=3)
        The size of the kernel for pairwise convolutional layers.
    pair_kernel_weight : float, optional (default=3)
        The weight of the kernel for pairwise convolutional layers.
    monotone_constraints : list or None, optional (default=None)
        The monotonic constraints for the features.
        0 indicates no constraint, 1 indicates increasing, -1 indicates decreasing.
        None means no constraints.
    num_pairs : int, default=0
        Number of pairwise interactions to use in the model.
    verbosity : int, default=0
        Level of verbosity for logging.
        0: Warning, 1: Info, 2: Debug
    random_state : int, optional
        Random seed for reproducibility.
    censoring_estimator : str, optional (default="km")
        The estimator to use for estimating the censoring distribution.
        "km" for Kaplan-Meier, "cox" for Cox proportional hazards.
    """

    
    def __init__(
        self, 
        n_features, 
        n_eval_times=100,
        n_embed=32,
        n_hidden=32, 
        n_layers=2,
        max_bins=32, 
        min_samples_per_bin=None,
        validation_size=0.2, 
        n_val_splits=5, 
        learning_rate=5e-4, 
        max_epochs=100, 
        batch_size=128, 
        device="cpu", 
        pairs_list=None, 
        kernel_size=5, 
        kernel_weight=3, 
        pair_kernel_size=3, 
        pair_kernel_weight=3, 
        monotone_constraints=None,
        num_pairs=0,
        verbosity=0,
        random_state=None,
        censor_estimator="km"
    ):
        super().__init__(
            n_features=n_features,
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=n_eval_times,
            n_layers=n_layers,
            max_bins=max_bins,
            min_samples_per_bin=min_samples_per_bin,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            pairs_list=pairs_list,
            kernel_size=kernel_size,
            kernel_weight=kernel_weight,
            pair_kernel_size=pair_kernel_size,
            pair_kernel_weight=pair_kernel_weight,
            monotone_constraints=monotone_constraints,
            num_pairs=num_pairs, 
            verbosity=verbosity,
            random_state=random_state
        )
        
        self.censor_estimator = censor_estimator
        assert self.censor_estimator in ["km", "cox"], "censor_estimator must be 'km' or 'cox'"
        
    def _set_gamma(self, X):
        
        # Set gamma and pair_gamma if not provided
        # Set to smaller value than default for survival analysis
        if self.gamma is None:
            super()._set_gamma(X)
            self.gamma = self.gamma / 2
        if self.pair_gamma is None:
            self.pair_gamma = self.gamma / 4
            
    def _fit_censoring_distribution(self, y, X=None):
        assert self.censor_estimator == "km" or X is not None, "if censor_estimator is cox, X must be provided."
        
        if self.censor_estimator == "km":
            self.logger.debug("Fitting Kaplan-Meier estimator for censoring distribution.")
            self.cde = CensoringDistributionEstimator()
            self.cde.fit(y)
            self.logger.debug("Finished fitting Kaplan-Meier estimator for censoring distribution.")
        else:
            self.logger.debug("Fitting Cox proportional hazards model for censoring distribution.")
            self.cde = CoxPHSurvivalAnalysis(alpha=1e-3)
            
            # Definte preprocessor so that Cox model can fit successfully
            from sklearn.compose import make_column_transformer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            
            self.censor_cox_preprocessor = make_column_transformer(
                (make_pipeline(
                    StandardScaler(),
                    SimpleImputer(strategy="constant", fill_value=0)
                ), X.select_dtypes(include=["number"]).columns),
                (make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="NA"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                ), X.select_dtypes(include=["object"]).columns),
                verbose_feature_names_out=False
            ).set_output(transform="pandas")
            X_preprocessed = self.censor_cox_preprocessor.fit_transform(X)
                
            # Fit on inverse of y to fit to censoring distribution
            y_censor = y.copy()
            y_censor["event"] = ~y_censor["event"]
            self.cde.fit(X_preprocessed, y_censor)
            self.logger.debug("Finished fitting Cox proportional hazards model for censoring distribution.")
            
        return

    # Set the pcw_obs_times
    # This needs to be done before training the model
    # So that unprocessed X's can be used
    def _set_pcw_obs_times(self, y_train=None, y_val=None, y_test=None, X_train=None, X_val=None, X_test=None):
        if y_train is not None:
            self.pcw_obs_times_train = self._predict_censoring_distribution(y_train["time"], X_train) + 1e-5
        if y_val is not None:
            self.pcw_obs_times_val = self._predict_censoring_distribution(y_val["time"], X_val) + 1e-5
        if y_test is not None:
            self.pcw_obs_times_test = self._predict_censoring_distribution(y_test["time"], X_test) + 1e-5
        return

    # Predict the survival probabilities of the censoring distribution
    def _predict_censoring_distribution(self, times, X=None):
        assert self.censor_estimator == "km" or X is not None, "if censor_estimator is cox, X must be provided."
        
        if self.censor_estimator == "km":
            return self.cde.predict_proba(times)
        else:
            X_preprocessed = self.censor_cox_preprocessor.transform(X)
            surv_fns = self.cde.predict_survival_function(X_preprocessed)
            return np.array([fn(time) for fn, time in zip(surv_fns, times)])
            
        
    def get_data_loader(self, X, y, pcw_obs_times, pairs=None, shuffle=True):
        
        assert "time" in y.dtype.names and "event" in y.dtype.names, \
            "y must be a structured array with 'time' and 'event' fields."
        
        # Convert X to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if pairs is not None:
            if hasattr(pairs, 'values'):
                pairs = pairs.values

        # pcw_obs_times = self.cde.predict_proba(y["time"]) + 1e-5
        # pcw_obs_times = self._predict_censoring_distribution(y["time"], X) + 1e-5

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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None, on_partialed_set=False):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, events, times, pcw_obs_times in train_loader:

            X_main, events, times, pcw_obs_times = X_main.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main, on_partialed_set=on_partialed_set)
            y_pred += y_pred_init
            
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
                loss += model.loss_penalty(mains=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        # Reset active feats
        model.active_feats = active_feats
        
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader, on_partialed_set=False):
        model.eval()
        total_loss = 0
        preds = []
        
        active_feats = model.active_feats
        y_pred_init = 0
        
        with torch.no_grad():
            for X_main, events, times, pcw_obs_times in test_loader:

                X_main, events, times, pcw_obs_times = X_main.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

                y_pred = model(mains=X_main, on_partialed_set=on_partialed_set)
                y_pred += y_pred_init
        
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
                    loss += model.loss_penalty(mains=True)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def train_epoch_pairs(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X_main, X_pairs, events, times, pcw_obs_times in train_loader:

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
                loss += model.loss_penalty(mains=False)

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
            for X_main, X_pairs, events, times, pcw_obs_times in test_loader:

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
                    loss += model.loss_penalty(mains=False)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)   
    
    def select_features(
        self, 
        X, 
        y, 
        reg_param,
        select_pairs=False, 
        partialed_feats=None,
        gamma=None,
        pair_gamma=None, 
        pair_reg_param=0, 
        entropy_param=0, 
    ):
        """
        Perform feature selection. Selected features and pairs will be stored in model.selected_feats
        and model.selected_pairs, respectively. Should be called before fit if feature selection is desired.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for the model.   
        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable.
        reg_param : float
            Regularization parameter for feature-level regularization.
        select_pairs : bool, default=False
            Whether to select feature pairs in addition to individual features.
        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        gamma : float, default=1
            Regularization or scaling parameter; purpose depends on specific use in the model.
        pair_gamma : float or None, default=None
            Regularization or scaling parameter for feature pairs.
            If None, defaults to gamma.
        pair_reg_param : float, default=0
            Regularization parameter for feature pairs.
        entropy_param : float, default=0
            Entropy parameter to control the diversity or uncertainty.
        """
        
        # self.cde = CensoringDistributionEstimator()
        # self.cde.fit(y)
        self._fit_censoring_distribution(y, X)
    
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
        
        self.pcw_eval_times = torch.FloatTensor(self._predict_censoring_distribution(self.eval_times.cpu().numpy(), X)).to(self.device) + 1e-5
        
        # Get the observed pcw times for the first split
        self.pcw_obs_times = []
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size, random_state=self.random_state)
        self.pcw_obs_times.append({
            "train": self._predict_censoring_distribution(y_train["time"], X_train) + 1e-5,
            "val": self._predict_censoring_distribution(y_val["time"], X_val) + 1e-5
        })
        
        return super().select_features(
            X, 
            y, 
            select_pairs=select_pairs, 
            partialed_feats=partialed_feats,
            gamma=gamma,
            pair_gamma=pair_gamma,
            reg_param=reg_param,
            pair_reg_param=pair_reg_param,
            entropy_param=entropy_param
        )
    
    def _select_features(self, X, y, partialed_feats=None, model=None):
        return super()._select_features(
            X, 
            y, 
            partialed_feats, 
            model, 
            train_loader_args={"pcw_obs_times": self.pcw_obs_times[0]["train"]},
            val_loader_args={"pcw_obs_times": self.pcw_obs_times[0]["val"]}
        )
    
    def fit(self, X, y, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for training.

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable or labels.

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        
        # Check the dtype of the labels
        assert y.dtype.names is not None, "y must be a structured array with 'time' and 'event' fields."
        assert "time" in y.dtype.names and "event" in y.dtype.names, "y must be a structured array with 'time' and 'event' fields."
        
        # self.cde = CensoringDistributionEstimator()
        # self.cde.fit(y)
        if not hasattr(self, "cde"):
            self._fit_censoring_distribution(y, X)
    
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
        
        self.pcw_eval_times = torch.FloatTensor(self._predict_censoring_distribution(self.eval_times.cpu().numpy(), X)).to(self.device) + 1e-5
        
        self.pcw_obs_times = []
        for i in range(self.n_val_splits):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size, random_state=self.random_state+i)
            self.pcw_obs_times.append({
                "train": self._predict_censoring_distribution(y_train["time"], X_train) + 1e-5,
                "val": self._predict_censoring_distribution(y_val["time"], X_val) + 1e-5
            })
        
        
        super().fit(X, y, partialed_feats=partialed_feats)
        
        return
    
    def _fit_one_split(self, X_train, y_train, X_val, y_val, random_state=None, partialed_feats=None):
        return super()._fit_one_split(
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            random_state, 
            partialed_feats, 
            train_loader_args={"pcw_obs_times": self.pcw_obs_times[random_state-self.random_state]["train"]},
            val_loader_args={"pcw_obs_times": self.pcw_obs_times[random_state-self.random_state]["val"]}
        )
    
    def get_feature_importances(self, eval_time=None, missing_bin="include"):
        """
        Get the feature importance scores for all features in the model.
        
        Parameters
        ----------
        eval_time : float or None, default=None
            The evaluation time for which to compute the feature importance.
            
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the feature importance scores for each feature.
        """
        
        if eval_time is not None:
            eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        
        importances = []
        for split, model in enumerate(self.models):
            
            for i, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[i]
                bin_counts = model.bin_counts[i]
                
                if missing_bin != "include" and self.has_missing_bin[i]:
                    missing_bin_score = feat_bin_scores[0, :]
                    feat_bin_scores = feat_bin_scores[1:, :]
                    bin_counts = bin_counts[1:]
                
                if eval_time is not None:
                    feat_importance = np.sum(np.abs(feat_bin_scores[:, eval_index]) * np.array(bin_counts)) / np.sum(bin_counts)
                else:
                    feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(bin_counts).reshape(-1, 1)) / np.sum(bin_counts)
                
                if missing_bin == "stratify":
                    if self.has_missing_bin[i]:
                        if eval_time is not None:
                            missing_importance = np.abs(missing_bin_score[eval_index])
                        else:
                            missing_importance = np.sum(np.abs(missing_bin_score))
                    else:
                        missing_importance = np.nan
                    importances.append([split, col, feat_importance, "observed", len(feat_bin_scores)])
                    importances.append([split, col, missing_importance, "missing", 1])
                else:
                    importances.append([split, col, feat_importance, len(feat_bin_scores)])
                
            if self.fit_pairs:
                pairs_list = model.pairs_list.cpu().numpy()
                for i, pair in enumerate(pairs_list):
                    pair_bin_scores = model.pair_bin_scores[i]
                                    
                    if eval_time is not None:
                        pair_importance = np.sum(np.abs(pair_bin_scores[:, eval_index]) * np.array(model.pair_bin_counts[i])) / np.sum(model.pair_bin_counts[i])
                    else:
                        pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[i]).reshape(-1, 1)) / np.sum(model.pair_bin_counts[i])
                        
                    if missing_bin == "stratify":
                        importances.append([split, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, "observed", len(pair_bin_scores)])
                    else:
                        importances.append([split, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])    
                                
        if missing_bin == "stratify":
            self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "bin_type", "n_bins"])
        else:
            self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    
    def plot_feature_importances(self, n_features=10, eval_times=None, missing_bin="include"):
        """
        Plot a bar plot with the importance score for the top k features.
        
        Parameters
        ----------
        n_features : int, default=10
            Number of features to plot.
            
        eval_times : float, list or None, default=None
            The evaluation time(s) for which to compute the feature importance.
            None means to compute the importance over all evaluation times.
            
        missing_bin : str, default="include"
            How to handle missing bin when calculating feature importances:
            
            - "include" - include the missing bin.
            - "ignore" - ignore the missing bin.
            - "stratify" - calculate separate importances for missing and non-missing bins.
        """
        
        num_eval_times = len(eval_times) if isinstance(eval_times, list) else 1
        fig, axes = plt.subplots(1, num_eval_times, figsize=((8 if missing_bin == "stratify" else 6)*num_eval_times, 4))
        # plt.figure(figsize=(8 if missing_bin == "stratify" else 6, 4))
        
        if num_eval_times == 1:
            axes = [axes]
            eval_times = [eval_times]
        
        if not isinstance(eval_times, list):
            eval_times = [eval_times]
            
        for i, et in enumerate(eval_times):
            feat_imps = self.get_feature_importances(eval_time=et, missing_bin=missing_bin)
            
            if missing_bin == "stratify":
                
                
                feat_means = feat_imps.groupby(["feature", "bin_type"])["importance"].mean().reset_index()
                feat_means = feat_means.pivot(index="feature", columns="bin_type", values="importance").reset_index()
                feat_means = feat_means.sort_values(by="observed", ascending=True).tail(n_features)
                
                if self.n_val_splits == 1:
                    axes[i].barh(feat_means["feature"], feat_means["observed"], color=sns.color_palette()[0])
                    axes[i].barh(feat_means["feature"], -1*feat_means["missing"], color=sns.color_palette()[1])
                else:
                    feat_sems = feat_imps.groupby(["feature", "bin_type"])["importance"].agg(["sem"]).reset_index()
                    feat_sems = feat_sems.pivot(index="feature", columns="bin_type", values="sem").reset_index()
                    feat_imps = feat_means.merge(feat_sems, on="feature", suffixes=("", "_sem"))
                    axes[i].barh(
                        y=feat_imps["feature"],
                        width=feat_imps["observed"],
                        xerr=1.96 * feat_imps["observed_sem"],
                    )
                    axes[i].barh(
                        y=feat_imps["feature"],
                        width=-1*feat_imps["missing"],
                        xerr=1.96 * feat_imps["missing_sem"],
                        color=sns.color_palette()[1]
                    )
                    
                axes[i].set_ylabel("")

                # Change title of legend
                axes[i].legend(["Observed", "Missing"], title="Importance Type")
                axes[i].set_title(f"Feature Importances (stratified) at t={et}")
                
                xticks = axes[i].get_xticks()
                xticklabels = [round(abs(x), 2) for x in xticks]
                axes[i].set_xticks(xticks)
                axes[i].set_xticklabels(xticklabels)
                
            else:
            
                top_k_features = feat_imps.groupby("feature")["importance"].agg(["mean", "sem"])
                top_k_features = top_k_features.sort_values("mean", ascending=True).tail(n_features)

                if self.n_val_splits == 1:
                    axes[i].barh(top_k_features.index, top_k_features["mean"])
                else:
                    axes[i].barh(top_k_features.index, top_k_features["mean"], xerr=1.96*top_k_features["sem"])
                    
                axes[i].set_title(f"Feature Importances ({missing_bin}) at t={et}")

        plt.tight_layout()
    
    def get_shape_function(self, feature_name, eval_time):
        """
        Get the shape function data, i.e. the bin scores, for a given feature.
        
        Parameters
        ----------
        feature_name : str
            The name of the feature.
            
        eval_time : float
            The evaluation time for which to compute the shape function.
            
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the bin scores for the feature.
        """
        is_cat_col = self.feature_dtypes[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
        
        eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        
        dfs = []
        for i, model in enumerate(self.models):
        
            col_index = model.feature_names_in_.get_loc(feature_name)       
            
            feat_bin_scores = model.bin_scores[col_index][:, eval_index]
            
            
            if not is_cat_col:
                
                feat_bin_values = np.concatenate([
                    [np.nan],
                    [model.feature_bins[col_index].min() - 0.01],
                    model.feature_bins[col_index]
                ])
                
            else:
                
                feat_bin_values = model.feature_bins[col_index]
                
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "bin": feat_bin_values,
                    "score": feat_bin_scores,
                    "count": model.bin_counts[col_index],
                    "split": i
                })
            )
            
        df = pd.concat(dfs)
            
        return df.reset_index()
    
    def plot_shape_function(self, feature_names, eval_times, plot_missing_bin=False):
        """
        Plot the shape function for given feature(s).
        
        Parameters
        ----------
        feature_names : str or list of str
            The name of the feature(s) to plot.
            
        eval_times : float or list of float
            The evaluation time(s) for which to compute the shape function.
            
        plot_missing_bin : bool, default=False
            Whether to plot the missing bin.
            Only applicable for continuous features.
        """
        
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            
        if not isinstance(eval_times, list):
            eval_times = [eval_times]
            
        is_cat_cols = [self.feature_dtypes[self.feature_names_in_.get_loc(f)] != 'continuous' for f in feature_names]
        
        num_axes = sum([2 if not c and plot_missing_bin else 1 for _, c in zip(feature_names, is_cat_cols)])
        # num_axes = num_axes * len(eval_times)
        
        if plot_missing_bin:
        
            # fig = plt.figure(figsize=(4*num_axes, 4))
            # fig = plt.figure(figsize=(4*num_axes, 4*len(eval_times)))
            
            
            # gs = gridspec.GridSpec(1, 2*len(feature_names) + len(feature_names)-1, width_ratios=[10, 1, 2] * (len(feature_names)-1) + [10, 1])  # Add an extra column for the space
            # def generate_indices(n_times):
            #     indices = []
            #     for i in range(n_times):
            #         indices.extend([i*3, i*3+1])
            #     return indices

            # axes = [fig.add_subplot(gs[i]) for i in generate_indices(len(feature_names))]
            
            fig = plt.figure(figsize=(4 * len(feature_names), 4 * len(eval_times)))

            # Create GridSpec with rows equal to len(eval_times) and appropriate columns
            gs = gridspec.GridSpec(len(eval_times), 2 * len(feature_names) + len(feature_names) - 1, width_ratios=[10, 1, 2] * (len(feature_names) - 1) + [10, 1])

            def generate_indices(n_times, row):
                """Generate indices for a specific row in the grid."""
                indices = []
                for i in range(n_times):
                    indices.extend([row * (2 * n_times + n_times - 1) + i * 3, row * (2 * n_times + n_times - 1) + i * 3 + 1])
                return indices

            # Create axes for all rows and columns
            axes = [fig.add_subplot(gs[i]) for row in range(len(eval_times)) for i in generate_indices(len(feature_names), row)]
            
        else:
            fig, axes = plt.subplots(len(eval_times), num_axes, figsize=(4*num_axes, 4*len(eval_times)))
            # if len(axes) == 1:
            #     axes = [axes]
            if not isinstance(axes, np.ndarray):
                axes = np.array(axes)
            if len(feature_names) == 1:
                axes = axes.reshape(-1, 1)
            if len(eval_times) == 1:
                axes = axes.reshape(1, -1)
                
        
        ax_idx = 0
        
        for eval_idx, eval_time in enumerate(eval_times):
            ax_idx = 0
            yaxis_label = f"Contribution to logit($P(T \\leq {eval_time} | X)$)"
            
            for feature_name in feature_names:
                
                is_cat_col = self.feature_dtypes[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
                
                shape_data = self.get_shape_function(feature_name, eval_time)
                
                if not is_cat_col:
                    if not plot_missing_bin:
                        line_plot_data = shape_data[shape_data["bin"].notna()].groupby("bin")["score"]
                        axes[eval_idx, ax_idx].plot(line_plot_data.mean().index, line_plot_data.mean(), drawstyle='steps-post')
                        axes[eval_idx, ax_idx].fill_between(
                            line_plot_data.mean().index,
                            line_plot_data.mean() - 1.96 * line_plot_data.sem(),
                            line_plot_data.mean() + 1.96 * line_plot_data.sem(),
                            alpha=0.3,
                            step='post'
                        )
                        axes[eval_idx, ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                        if ax_idx == 0:
                            axes[eval_idx, ax_idx].set_ylabel(yaxis_label)
                        else:
                            axes[eval_idx, ax_idx].set_ylabel("")
                        
                        ax_idx += 1
                    else:
                            
                        # Make the line plot
                        line_plot_data = shape_data[shape_data["bin"].notna()].groupby("bin")["score"]
                        axes[eval_idx, ax_idx].plot(line_plot_data.mean().index, line_plot_data.mean(), drawstyle='steps-post')
                        axes[eval_idx, ax_idx].fill_between(
                            line_plot_data.mean().index,
                            line_plot_data.mean() - 1.96 * line_plot_data.sem(),
                            line_plot_data.mean() + 1.96 * line_plot_data.sem(),
                            alpha=0.3,
                            step='post'
                        )
                        axes[eval_idx, ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                        if ax_idx == 0:
                            axes[eval_idx, ax_idx].set_ylabel(yaxis_label)
                        else:
                            axes[eval_idx, ax_idx].set_ylabel("")
                        
                        
                        ax_idx += 1

                        # Make the bar plot
                        bar_data = shape_data[shape_data["bin"].isna()]
                        axes[eval_idx, ax_idx].bar("NA", bar_data["score"].mean())
                        axes[eval_idx, ax_idx].errorbar("NA", bar_data["score"].mean(), yerr=1.96*bar_data["score"].sem(), fmt='none', color='black')
                        axes[eval_idx, ax_idx].set_ylim(axes[0].get_ylim())
                        axes[eval_idx, ax_idx].set_yticklabels([])
                        axes[eval_idx, ax_idx].set_xlim(-0.5, 0.5)
                        
                        ax_idx += 1
                        
                else:
                    shape_data = shape_data.fillna("NA")
                    sns.barplot(x="bin", y="score", data=shape_data, errorbar=('ci', 95), ax=axes[eval_idx, ax_idx])
                    axes[eval_idx, ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                    axes[eval_idx, ax_idx].tick_params(axis='x', rotation=45)
                    for label in axes[eval_idx, ax_idx].get_xticklabels():
                        label.set_ha('right')
                    if ax_idx == 0:
                        axes[eval_idx, ax_idx].set_ylabel(yaxis_label)
                    else:
                        axes[eval_idx, ax_idx].set_ylabel("")
                        
                    ax_idx += 1
                
        plt.tight_layout()
    
    def get_pair_shape_function(self, feat1_name, feat2_name, eval_time):
        """
        Get the shape function data for an interaction affect.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
        
        feat2_name : str
            The name of the second feature in the pair/interaction.
            
        eval_time : float
            The evaluation time to compute the interaction shape function at.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the shape function data for the interaction effect.
        """
        
        eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        
        bin_scores = []
        bin_values = []
        for i, model in enumerate(self.models):
        
            col1_index = model.feature_names_in_.get_loc(feat1_name)
            col2_index = model.feature_names_in_.get_loc(feat2_name)
            
            pair_index = [
                i for i, p in enumerate(model.pairs_list) if (p[0] == col1_index and p[1] == col2_index) or (p[0] == col2_index and p[1] == col1_index)
            ]
            
            if len(pair_index) == 0:
                raise ValueError("Pair not found")
            
            pair_index = pair_index[0]
            
            pair_bin_scores = model.pair_bin_scores[pair_index][:, eval_index]
            
            # feat1_nbins = len(model.feature_bins[col1_index])
            # feat2_nbins = len(model.feature_bins[col2_index])
            feat1_nbins = model.feature_sizes.cpu().numpy()[col1_index]
            feat2_nbins = model.feature_sizes.cpu().numpy()[col2_index]
            
            # grid = self._make_grid(np.arange(0, feat1_nbins+2), np.arange(0, feat2_nbins+2))
            grid = self._make_grid(np.arange(0, feat1_nbins), np.arange(0, feat2_nbins))
            
            if self.has_missing_bin[col1_index] and not self.has_missing_bin[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 0] > 0]
            elif not self.has_missing_bin[col1_index] and self.has_missing_bin[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 1] > 0]
            else:
                pair_bin_scores = pair_bin_scores[grid.prod(axis=1) > 0]
            
            # Get bin values
            if self.feature_dtypes[col1_index] != 'continuous':
                # feat1_bin_values = np.arange(-1, len(model.feature_bins[col1_index])+1)
                feat1_bin_values = model.feature_bins[col1_index]
            else:
                feat1_bin_values = np.concatenate([
                    [model.feature_bins[col1_index].min() - 0.01],
                    model.feature_bins[col1_index]
                ])
                
            if self.feature_dtypes[col2_index] != 'continuous':
                # feat2_bin_values = np.arange(-1, len(model.feature_bins[col2_index])+1)
                feat2_bin_values = model.feature_bins[col2_index]
            else:
                feat2_bin_values = np.concatenate([
                    [model.feature_bins[col2_index].min() - 0.01],
                    model.feature_bins[col2_index]
                ])

            pair_bin_values = self._make_grid(feat1_bin_values, feat2_bin_values)
            
            bin_scores.append(pair_bin_scores)
            bin_values.append(pair_bin_values)
                
        # return bin_scores, bin_values
        
        pair_scores = np.mean(np.stack(bin_scores), axis=0)
        pair_values = bin_values[0]

        pair_data_dnamite = pd.DataFrame({
            feat1_name: pair_values[:, 0],
            feat2_name: pair_values[:, 1],
            "score": pair_scores
        })

        pair_data_dnamite = pair_data_dnamite.pivot(
            index=feat1_name, 
            columns=feat2_name, 
            values="score"
        )

        # Round the values in the columns and index
        # Round the index and columns to 3 decimal places and reassign 
        if pair_data_dnamite.index.dtype == 'float64':
            pair_data_dnamite.index = pair_data_dnamite.index.round(3)
        if pair_data_dnamite.columns.dtype == 'float64':
            pair_data_dnamite.columns = pair_data_dnamite.columns.round(3)
        
        return pair_data_dnamite
    
    def plot_pair_shape_function(self, feat1_name, feat2_name, eval_times):
        """
        Plot a heatmap for an interaction shape function.
        
        Parameters
        ----------
        feat1_name : str
            The name of the first feature in the pair/interaction.
            
        feat2_name : str
            The name of the second feature in the pair/interaction.
            
        eval_times : float or list of float
            The evaluation time(s) to plot the interaction shape function at.
        """
        
        # plt.figure(figsize=(4, 4))
        if not isinstance(eval_times, list):
            eval_times = [eval_times]
        
        fig, axes = plt.subplots(1, len(eval_times), figsize=(4*len(eval_times), 4))
        if len(eval_times) == 1:
            axes = [axes]
        
        for ax, eval_time in zip(axes, eval_times):
            pair_data_dnamite = self.get_pair_shape_function(feat1_name, feat2_name, eval_time)
            sns.heatmap(pair_data_dnamite, ax=ax)
            ax.set_title(f"Interaction at t={eval_time}")
        
        plt.tight_layout()
        
    def _predict(self, X_test_discrete, models, pcw_obs_times_test):
        
        if hasattr(self, 'selected_feats'):
            if self.fit_pairs:
                pairs_list = [
                    [X_test_discrete.columns.get_loc(feat1), X_test_discrete.columns.get_loc(feat2)] for feat1, feat2 in self.selected_pairs
                ]    
        elif self.fit_pairs:
            pairs_list = self.pairs_list
        
        # Make placeholder y_test
        y_test = np.zeros(X_test_discrete.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        if not self.fit_pairs:
            test_loader = self.get_data_loader(X_test_discrete, y_test, shuffle=False, pcw_obs_times=pcw_obs_times_test)
            test_preds = np.zeros((len(models), X_test_discrete.shape[0], self.n_output))
            for i, model in enumerate(models):
                _, model_preds = self.test_epoch_mains(model, test_loader)
                test_preds[i, ...] = model_preds.cpu().numpy()
                
        else:
            X_test_interactions = X_test_discrete.values[:, pairs_list]
            test_loader = self.get_data_loader(X_test_discrete, y_test, pairs=X_test_interactions, shuffle=False, pcw_obs_times=pcw_obs_times_test)
            test_preds = np.zeros((len(models), X_test_discrete.shape[0], self.n_output))
            for i, model in enumerate(models):
                _, model_preds = self.test_epoch_pairs(model, test_loader)
                test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def predict(self, X_test):
        """
        Predict labels using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        # self._set_pcw_obs_times(y_test=np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")]), X_test=X_test)
        pcw_obs_times_test = self._predict_censoring_distribution(np.zeros(X_test.shape[0]), X_test) + 1e-5
        X_test_discrete = self._discretize_data(X_test)
        
        if hasattr(self, 'selected_feats'):
            self.logger.debug("Found selected features. Using only those features.")
            X_test_discrete = X_test_discrete[self.selected_feats]
        
        return self._predict(X_test_discrete, self.models, pcw_obs_times_test)
    
    def predict_survival(self, X_test, test_times=None):
        """
        Predict the survival probability for a set of evaluation times.
        
        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray, shape (n_samples, n_features)
            The input features for prediction.
            
        test_times : array-like, optional
            The evaluation times to predict the survival probability at.
            If None, the evaluation times used for training will be used.
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_eval_times)
            The predicted survival probabilities for each sample at each evaluation time.
        
        """
        
        cdf_preds = 1 / (1 + np.exp(-1 * self.predict(X_test)))
        surv_preds = 1 - cdf_preds
        
        if test_times is not None:
            surv_preds = surv_preds[
                :,
                np.searchsorted(self.eval_times.cpu().numpy(), test_times, side='right') - 1
            ]
            
        return surv_preds
    
    def _score(self, X, y, y_train, model=None):
        
        from sksurv.metrics import cumulative_dynamic_auc
        
        test_times = np.linspace(
            max(y_train["time"].min(), y[y["event"] > 0]["time"].min()) + 1e-4,
            min(y_train["time"].max(), y[y["event"] > 0]["time"].max()) - 1e-4,
            1000
        )
        
        pcw_obs_times_test = self._predict_censoring_distribution(np.zeros(X.shape[0]), X) + 1e-5
        
        preds = self._predict(X, models=[model], pcw_obs_times_test=pcw_obs_times_test)
        cdf_preds = 1 / (1 + np.exp(-1 * preds))
        
        cdf_preds = cdf_preds[
            :,
            np.clip(
                np.searchsorted(self.eval_times.cpu().numpy(), test_times),
                0, cdf_preds.shape[1]-1
            )
        ]
        
        return {"Mean_AUC": cumulative_dynamic_auc(y_train, y, cdf_preds, test_times)[1]}
    
    def get_calibration_data(self, X, y, eval_time, n_bins=10, binning_method="quantile"):
        """
        Get calibration data to assess the calibration of the model at a given evaluation time.
        
        Parameters
        ----------
        X : array-like
            Input data used to generate prediction. Should usually be a held-out test set.
            
        y : structured np.array of shape (n_samples,) with dtype [("event", bool), ("time", float)]
            Survival labels for corresponding to X.
            
        eval_time : float
            Evaluation time to assess calibration at.
            
        n_bins : int, optional (default=10)
            Number of bins to use for binned Kaplan-Meier estimate.
            
        binning_method : str, optional (default="quantile")
            Method for binning predictions. Options are "quantile" or "uniform".
        """
        
        # First get cdf preds
        cdf_preds = 1 / (1 + np.exp(-1 * self.predict(X)))
        eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        cdf_preds = cdf_preds[:, eval_index]
        
        
        if binning_method == "quantile":
            # Get bins from quantiles of predictions
            quantiles = np.quantile(
                cdf_preds,
                np.linspace(0, 1, n_bins+1)
            )
        elif binning_method == "uniform":
            # Get bins from uniform spacing
            quantiles = np.linspace(cdf_preds.min(), cdf_preds.max(), n_bins+1)
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
            predicted.append(np.mean(bin_data))
            observed.append(
                1 - surv_prob[np.clip(
                    np.searchsorted(times, eval_time), 0, len(times)-1
                )]
            )
            
            
        return np.array(predicted), np.array(observed), quantiles
    
    def make_calibration_plot(self, X, y, eval_times, n_bins=10, binning_method="quantile"):
        """
        Make a calibration plot to assess the calibration of the model at a given evaluation time.
        
        Parameters
        ----------
        X : array-like
            Input data used to generate prediction. Should usually be a held-out test set.
            
        y : structured np.array of shape (n_samples,) with dtype [("event", bool), ("time", float)]
            Survival labels for corresponding to X.
            
        eval_times : float or list of float
            Evaluation time(s) to assess calibration at.
            
        n_bins : int, optional (default=10)
            Number of bins to use for binned Kaplan-Meier estimate.
            
        binning_method : str, optional (default="quantile")
            Method for binning predictions. Options are "quantile" or "uniform".
        """
        
        if not isinstance(eval_times, list):
            eval_times = [eval_times]
        
        fig, axes = plt.subplots(1, len(eval_times), figsize=(4*len(eval_times), 4))
        if len(eval_times) == 1:
            axes = [axes]
        
        for eval_time, ax in zip(eval_times, axes):
        
            predicted, observed, quantiles = self.get_calibration_data(X, y, eval_time, n_bins=n_bins, binning_method=binning_method)
            
            ax.plot(predicted, observed, marker='o')
            ax.plot([predicted[0],  predicted[-1]], [predicted[0], predicted[-1]], linestyle='--', color='black')
            ax.set_xlabel(r"$\hat{P}(T \leq $" + str(eval_time) + r"$ \mid X)$")
            ax.set_ylabel("Binned KM Estimate")
            ax.set_title(f"Calibration at t={eval_time}")
            
        plt.tight_layout()