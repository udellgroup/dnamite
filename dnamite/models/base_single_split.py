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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm
from functools import partial
from dnamite.utils import discretize, get_bin_counts, get_pair_bin_counts
from dnamite.loss_fns import ipcw_rps_loss
import textwrap
from contextlib import nullcontext
import copy
from dnamite.utils import LoggingMixin
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

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
        self.feature_sizes_ = torch.tensor(feature_sizes).to(device)
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
            self.cat_feat_mask_ = cat_feat_mask.to(device)
        else:
            self.cat_feat_mask_ = torch.zeros(n_features).astype(bool).to(device)
            
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
                    (mains <= self.embedding_offsets[self.active_feats].reshape(1, -1, 1)) | (mains >= self.embedding_offsets[self.active_feats].reshape(1, -1, 1) + self.feature_sizes_[self.active_feats].reshape(1, -1, 1)),
                    torch.zeros_like(weights),
                    weights
                )
                
                # Zero out weights for categorical feats
                weights = torch.where(
                    torch.logical_and(
                        self.cat_feat_mask_[self.active_feats].reshape(1, -1, 1),
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
            pairs_cat_feat_mask = self.cat_feat_mask_[self.pairs_list[self.active_pairs]]
            
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
                
                pair_sizes = self.feature_sizes_[self.pairs_list[self.active_pairs]].to(self.device)
                
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
        
            feat_inputs = torch.arange(0, self.feature_sizes_[feat_index]).to(self.device)
            
            # Add offsets to features to get the correct indices
            # offsets = torch.tensor(self.embedding_offsets).unsqueeze(0).expand(inputs.size(0), -1).to(self.device)
            offset = self.embedding_offsets[feat_index]
            inputs = feat_inputs + offset
            
            if self.kernel_size > 0 and self.kernel_weight > 0 and not self.cat_feat_mask_[feat_index]:
                
                # Apply kernel smoothing to the embeddings
                inputs = inputs.unsqueeze(-1) + torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)
                weights = torch.exp(-torch.square(torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)) / (2 * self.kernel_weight))
                
                weights = torch.where(
                    (inputs <= offset) | (inputs >= offset + self.feature_sizes_[feat_index]),
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
            size1 = self.feature_sizes_[pair[0]]
            size2 = self.feature_sizes_[pair[1]]
            
            # Get all pairs of indices for the pair
            # pairs is of shape (batch_size, 2)
            pairs = torch.stack(torch.meshgrid(
                torch.arange(size1), torch.arange(size2), indexing="ij"
            ), dim=-1).reshape(-1, 2).to(self.device)
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (2)
            offsets = self.embedding_offsets[self.pairs_list[pair_index]].to(self.device)
            pairs = pairs + offsets
            
            if self.pair_kernel_size > 0 and self.pair_kernel_weight > 0 and not (self.cat_feat_mask_[pair[0]] and self.cat_feat_mask_[pair[1]]):
            
                # Pairs will now be shape (batch_size, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2)
                pairs = self._get_pairs_neighbors(pairs)
                
                # Get the kernel weights using gaussian kernel
                # Will be shape (2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = self._get_pair_weights()
                
                pair_sizes = self.feature_sizes_[self.pairs_list[pair_index]].to(self.device)
                
                # weights is now (batch_size, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = torch.where(
                    (pairs[:, :, :, 0] <= offsets[0]) | (pairs[:, :, :, 1] <= offsets[1]) | \
                        (pairs[:, :, :, 0] >= offsets[0] + pair_sizes[0]) | (pairs[:, :, :, 1] >= offsets[1] + pair_sizes[1]),
                    torch.zeros_like(weights), 
                    weights
                )
                
                if self.cat_feat_mask_[pair[0]] and not self.cat_feat_mask_[pair[1]]:
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
                elif not self.cat_feat_mask_[pair[0]] and self.cat_feat_mask_[pair[1]]:
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