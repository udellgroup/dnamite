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
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from tqdm import tqdm
import os
from functools import partial
from dnamite.utils import discretize, get_bin_counts, get_pair_bin_counts
from dnamite.loss_fns import ipcw_rps_loss

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
class BaseSingleSplitDNAMiteModel(nn.Module):
    def __init__(self, 
                 n_features, 
                 n_output,
                 feature_sizes, 
                 n_embed=32,
                 n_hidden=32, 
                 n_layers=2,
                 gamma=1, 
                 reg_param=0, 
                 pair_reg_param=0, 
                 entropy_param=0, 
                 fit_pairs=True, 
                 device="cpu", 
                 pairs_list=None, 
                 kernel_size=5,
                 kernel_weight=1,
                 pair_kernel_size=3,
                 pair_kernel_weight=1,
                 save_dir=None
                 ):
        super().__init__()
        self.n_features = n_features
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.feature_sizes = torch.tensor(feature_sizes).to(device)
        self.n_layers = n_layers
        self.gamma = gamma
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
            self.selected_pair_indices = pairs_list
        
        if save_dir is not None:
            self.save_dir = save_dir            
        else:
            self.save_dir = "model_saves/"
            
        # check if save_dir exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        
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
        
        

    def forward(self, mains=None, pairs=None):
        # mains of shape (batch_size, n_features)
        # pairs of shape (batch_size, |n_features choose 2|, 2)
        
        output_main = 0
        output_pairs = 0
        
        if mains is not None:
            
            mains = mains[:, self.active_feats]
            
            # Add offsets to features to get the correct indices
            offsets = self.embedding_offsets.unsqueeze(0).expand(mains.size(0), -1).to(self.device)
            mains = mains + offsets[:, self.active_feats]
            
            if self.kernel_size > 0:
            
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

            # Get smoothed z
            z_main = self.get_smooth_z()[self.active_feats]

            output_main = torch.einsum('ijk,j->ik', mains, z_main)
        
        if pairs is not None:
            
            pairs = pairs[:, self.active_pairs, :]
            
            # Get cat_feat_mask
            pairs_cat_feat_mask = self.cat_feat_mask[self.pairs_list[self.active_pairs]]
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (n_pairs, 2)
            offsets = self.embedding_offsets[self.pairs_list[self.active_pairs]].to(self.device)
            pairs = pairs + offsets
            
            if self.pair_kernel_size > 0:
            
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
                
            z_pairs = self.get_smooth_z_pairs()[self.active_pairs]
            
            output_pairs = torch.einsum('ijk,j->ik', pairs, z_pairs)
        
        
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
    def get_bin_scores(self, feat_index, center=True):
        
        with torch.no_grad():
        
            inputs = torch.arange(0, self.feature_sizes[feat_index]).to(self.device)
            
            # Add offsets to features to get the correct indices
            # offsets = torch.tensor(self.embedding_offsets).unsqueeze(0).expand(inputs.size(0), -1).to(self.device)
            offset = self.embedding_offsets[feat_index]
            inputs = inputs + offset
            
            if self.kernel_size > 0 and not self.cat_feat_mask[feat_index]:
                
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

            # Get prediction using the smoothed z
            z_main = self.get_smooth_z()
            # prediction = output_main.squeeze() * z_main[feat_index]
            prediction = mains * z_main[feat_index]
            
            if hasattr(self, 'feat_offsets') and center:
                return prediction - self.feat_offsets[feat_index, :]
            elif center:
                print("Intercept not computed. Shape function will not be centered.")
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
            
            if self.pair_kernel_size > 0 and not (self.cat_feat_mask[pair[0]] and self.cat_feat_mask[pair[1]]):
            
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
                print("Intercept not computed. Shape function will not be centered.")
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
            
    def compute_intercept(self, bin_counts, ignore_missing_bin=False):
        
        # bin counts is a list of bin counts for each feature
        
        self.bin_counts = bin_counts
        self.feat_offsets = torch.zeros(self.n_features, self.n_output).to(self.device)
        
        for feat_idx in self.active_feats:
            bin_preds = self.get_bin_scores(feat_idx, center=False)
            feat_bin_counts = torch.tensor(bin_counts[feat_idx]).to(self.device).unsqueeze(-1)
            
            if ignore_missing_bin:
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
        mains=True
    ):
        

        early_stopping_counter = 0
        best_test_loss = float('inf')

        for epoch in range(n_epochs):
            train_epoch_fn(self, train_loader, optimizer)
            test_loss, test_preds = test_epoch_fn(self, test_loader)
            
            # print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f}")
            
            # Do feature pruning
            if mains:
                self.prune_parameters(mains=True, pairs=False)
            else:
                self.prune_parameters(mains=False, pairs=True)

            # Check if the test loss has improved
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stopping_counter = 0
                
                # Save the model at the best test loss
                # torch.save(self.state_dict(), "../model_saves/tmp_best_model.pt")
                torch.save(self.state_dict(), os.path.join(self.save_dir, "tmp_best_model.pt"))
                
            else:
                early_stopping_counter += 1

            # If test loss has not improved for 5 consecutive epochs, terminate training
            if early_stopping_counter >= 5:
                print(f"Early stopping at {epoch+1} epochs: Test loss has not improved for 5 consecutive epochs.")
                break
            
        # Load the model from the best test loss
        self.load_state_dict(torch.load(os.path.join(self.save_dir, "tmp_best_model.pt")))

        return

# Parent DNAMite model
# n_features: number of features in the data
# n_embed: embedding dimension
# n_hidden: hidden layer dimension
# n_output: output dimension
# max_bins: maximum number of bins to discretize each feature
# validation_size: proportion of data to use for validation
# n_val_splits: number of validation splits to average over
# learning_rate: learning rate for the optimizer
# max_epochs: maximum number of epochs to train for
# batch_size: batch size for training
# device: device to run the model on
# **kwargs: other arguments to pass to BaseSingleSplitDNAMiteModel
class BaseDNAMiteModel(nn.Module):
    
    def __init__(
        self, 
        n_features, 
        n_output, 
        n_embed=32,
        n_hidden=32, 
        max_bins=32,
        validation_size=0.2,
        n_val_splits=5,
        learning_rate=1e-4,
        max_epochs=100,
        batch_size=128,
        device="cpu",
        fit_pairs=True,
        reg_param=0,
        entropy_param=0,
        **kwargs
    ):
        super().__init__()
        self.n_features = n_features
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.max_bins = max_bins
        self.validation_size = validation_size
        self.n_val_splits = n_val_splits
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.fit_pairs = fit_pairs
        self.reg_param = reg_param
        self.entropy_param = entropy_param
        
        self.model_args = kwargs
    
    def infer_data_types(self, X):
        
        self.feature_dtypes = []
        for i in range(X.shape[1]):
            if X.iloc[:, i].dtype.name == 'category' or \
                X.iloc[:, i].dtype.name == 'object' or \
                X.iloc[:, i].nunique() <= 2:

                self.feature_dtypes.append('categorical')
            else:
                self.feature_dtypes.append('continuous')
                
        self.cat_feat_mask = np.array(self.feature_dtypes) == 'categorical'
    
    def discretize_data(self, X):
        X_discrete = X.copy()
        
        if not hasattr(self, 'feature_dtypes'):
            self.infer_data_types(X_discrete)
        
        # If X is pandas, store column names
        if hasattr(X, 'columns'):
            col_names = X.columns
            X_discrete = X_discrete.values
        
        # If feature_bins is already set, use existing bins
        if hasattr(self, 'feature_bins'):
            for i in range(self.n_features):
                if self.feature_dtypes[i] == 'continuous':
                    X_discrete[:, i], _ = discretize(X_discrete[:, i], max_bins=self.max_bins, bins=self.feature_bins[i])
                else:
                    ordinal_map = {val: float(i) for i, val in enumerate(self.feature_bins[i])}
                    X_discrete[:, i] = np.vectorize(ordinal_map.get)(X_discrete[:, i])
                
        else:
            self.feature_bins = []
            self.feature_sizes = []
            for i in range(self.n_features):
                if self.feature_dtypes[i] == 'continuous':
                    X_discrete[:, i], bins = discretize(np.ascontiguousarray(X_discrete[:, i]), max_bins=self.max_bins)
                else:
                    
                    # Ordinal encoder and force missing value to be 0
                    ordinal_encoder = OrdinalEncoder(dtype=float, handle_unknown="use_encoded_value", unknown_value=-1)
                    X_discrete[:, i] = ordinal_encoder.fit_transform(X_discrete[:, i].reshape(-1, 1)).flatten() + 1.0
                    bins = ["NA"] + list(ordinal_encoder.categories_[0])
                
                self.feature_bins.append(bins)
                
                # Number of bins is (maximum bin index) + 1 (accounting for missing bin)
                self.feature_sizes.append(int(X_discrete[:, i].max() + 1))
            
        if hasattr(X, 'columns'):
            X_discrete = pd.DataFrame(X_discrete, columns=col_names, dtype=float)
            
        return X_discrete
    
    def compute_bin_scores(self, ignore_missing_bin_in_intercept=True):
        
        for model in self.models:
        
            model.compute_intercept(model.bin_counts, ignore_missing_bin=ignore_missing_bin_in_intercept)
            
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
    
    def fit_one_split(self, X_train, y_train, X_val, y_val, partialed_indices=None):
        
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            print("Found selected features. Using only those features.")
            X_train = X_train[self.selected_feats]
            X_val = X_val[self.selected_feats]
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False)
        
        model = BaseSingleSplitDNAMiteModel(
            n_features=X_train.shape[1], 
            n_embed=self.n_embed,
            n_hidden=self.n_hidden, 
            n_output=self.n_output,
            feature_sizes=self.feature_sizes, 
            device=self.device,
            fit_pairs=self.fit_pairs,
            reg_param=0, # only use reg/entropy during feature selection
            entropy_param=0,
            **self.model_args
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        if partialed_indices is not None:
            print("TRAINING PARTIALED FEATS")
            model.active_feats = torch.tensor(partialed_indices).to(self.device)
            model.train_(
                self.train_epoch_mains, 
                self.test_epoch_mains, 
                train_loader, 
                val_loader, 
                optimizer, 
                self.max_epochs,
                mains=True
            )
            model.active_feats = torch.arange(self.n_features).to(self.device)
        
        print("TRAINING MAINS")
        model.train_(
            train_epoch_fn=partial(self.train_epoch_mains, partialed_indices=partialed_indices),
            test_epoch_fn=self.test_epoch_mains,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            n_epochs=self.max_epochs,
            mains=True
        )
        
        if not self.fit_pairs:
            return model

        if not hasattr(self, 'selected_pair_indices'):
            self.selected_pair_indices = list(combinations(range(X_train.shape[1]), 2))
        
        model.freeze_main_effects()
        model.pairs_list = torch.LongTensor(self.selected_pair_indices).to(self.device)
        model.n_pairs = len(self.selected_pair_indices)
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
        
        print("TRAINING PAIRS")
        model.train_(
            self.train_epoch_pairs, 
            self.test_epoch_pairs, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs,
            mains=False
        )
        
        return model
    
    def select_features(self, X, y, partialed_feats=None):
            
        if partialed_feats is not None:
            partialed_indices = [X.columns.get_loc(feat) for feat in partialed_feats]
        else:
            partialed_indices = None
        
        self.feature_names_in_ = X.columns
        
        if hasattr(X, 'values'):
            X = X.values
        
        # First discretize the data
        X_discrete = self.discretize_data(X)
        
        # Make one train/val split for feature selection
        X_train, X_val, y_train, y_val = train_test_split(X_discrete, y, test_size=self.validation_size)
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False)
        
        model = BaseSingleSplitDNAMiteModel(
            n_features=self.n_features, 
            n_embed=self.n_embed,
            n_hidden=self.n_hidden, 
            n_output=self.n_output,
            feature_sizes=self.feature_sizes,
            device=self.device,
            fit_pairs=self.fit_pairs,
            reg_param=self.reg_param,
            entropy_param=self.entropy_param,
            **self.model_args
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        if partialed_indices is not None:
            model.active_feats = torch.tensor(partialed_indices).to(self.device)
            reg_param = model.reg_param
            entropy_param = model.entropy_param
            model.reg_param = 0
            model.entropy_param = 0
            
            model.train_(
                self.train_epoch_mains, 
                self.test_epoch_mains, 
                train_loader, 
                val_loader, 
                optimizer, 
                self.max_epochs,
                mains=True
            )
            
            model.active_feats = torch.arange(self.n_features).to(self.device)
            model.reg_param = reg_param
            model.entropy_param = entropy_param
        
        model.train_(
            partial(self.train_epoch_mains, partialed_indices=partialed_indices), 
            self.test_epoch_mains, 
            train_loader, 
            val_loader, 
            optimizer, 
            self.max_epochs,
            mains=True
        )
        
        print("Number of main features selected: ", len(model.active_feats))
        
        if not self.fit_pairs:
            return

        active_feats = model.active_feats.cpu().numpy()
        active_pairs = list(combinations(active_feats, 2))

        X_train_interactions = X_train[:, active_pairs]
        X_val_interactions = X_val[:, active_pairs]

        model.freeze_main_effects()
        model.pairs_list = torch.LongTensor(active_pairs).to(self.device)
        model.n_pairs = len(active_pairs)
        model.init_pairs_params(model.n_pairs)
        model.active_pairs = torch.arange(model.n_pairs).to(self.device)
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
            self.max_epochs,
            mains=False
        )
        
        self.selected_feats = self.feature_names_in_[model.active_feats.cpu().numpy()].tolist()
        self.selected_pairs = [
            [self.feature_names_in_[pair[0]], self.feature_names_in_[pair[1]]]
            for pair in model.pairs_list[model.active_pairs].cpu().numpy()
        ]
        self.selected_pair_indices = [
            [self.selected_feats.index(feat) for feat in pair] for pair in self.selected_pairs
        ]
        
        print("Number of interaction features selected: ", len(self.selected_pairs))
        
        return
    
    def fit(self, X, y, partialed_feats=None):
        
        if partialed_feats is not None:
            partialed_indices = [X.columns.get_loc(feat) for feat in partialed_feats]
        else:
            partialed_indices = None
        
        self.feature_names_in_ = X.columns
        
        # First discretize the data
        X_discrete = self.discretize_data(X)
        
        # Fit several models, one for each validation split
        self.models = []
        for i in range(self.n_val_splits):
            print("SPlIT", i)
        
            # Split the data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X_discrete, y, test_size=self.validation_size)
            
            # Fit to this split
            model = self.fit_one_split(X_train, y_train, X_val, y_val, partialed_indices=partialed_indices)
            model.feature_names_in_ = X_train.columns
            model.feature_bins = self.feature_bins
            
            # Compute the bin counts
            model.bin_counts = [
                get_bin_counts(X_train[col], nb) for col, nb in zip(X_train.columns, self.feature_sizes)
            ]
            
            if self.fit_pairs:
                pairs_list = model.pairs_list.cpu().numpy()
                model.pair_bin_counts = [
                    get_pair_bin_counts(X_train.iloc[:, col1], X_train.iloc[:, col2]) 
                    for col1, col2 in pairs_list
                ]
            
            self.models.append(model)
            
        self.compute_bin_scores()
            
        return
    
    def predict(self, X_test):
        
        X_test_discrete = self.discretize_data(X_test)
        
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            X_test_discrete = X_test_discrete[self.selected_feats]
            
        # Make placeholder y_test
        y_test = np.zeros(X_test_discrete.shape[0])
            
        if not self.fit_pairs:
            test_loader = self.get_data_loader(X_test_discrete, y_test, shuffle=False)
            test_preds = np.zeros((self.n_val_splits, X_test.shape[0]))
            for i, model in enumerate(self.models):
                _, model_preds = self.test_epoch_mains(model, test_loader)
                test_preds[i, ...] = model_preds.cpu().numpy()
        
        else:
            X_test_interactions = X_test_discrete.values[:, self.selected_pair_indices]
            test_loader = self.get_data_loader(X_test_discrete, y_test, pairs=X_test_interactions, shuffle=False)
            
            test_preds = np.zeros((self.n_val_splits, X_test.shape[0]))
            for i, model in enumerate(self.models):
                _, model_preds = self.test_epoch_pairs(model, test_loader)
                test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def get_feature_importances(self, ignore_missing_bin=False):
        
        importances = []
        for split, model in enumerate(self.models):
            
            model.compute_intercept(model.bin_counts, ignore_missing_bin=ignore_missing_bin)
            
            pairs_list = model.pairs_list.cpu().numpy()
            model.compute_pairs_intercept(model.pair_bin_counts)
            
            for i, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[i].squeeze()
                
                feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(model.bin_counts[i])) / np.sum(model.bin_counts[i])
                    
                importances.append([split, col, feat_importance, len(feat_bin_scores)])
                
            for i, pair in enumerate(pairs_list):
                pair_bin_scores = model.pair_bin_scores[i].squeeze()
                
                pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[i])) / np.sum(model.pair_bin_counts[i])
                    
                importances.append([split, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])
            
        self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    
    def plot_feature_importances(self, k=10, ignore_missing_bin=False):
        plt.figure(figsize=(6, 4))

        feat_imps = self.get_feature_importances(ignore_missing_bin=ignore_missing_bin)
        
        top_k_features = feat_imps.groupby("feature")["importance"].agg(["mean", "sem"])
        top_k_features = top_k_features.sort_values("mean", ascending=True).tail(k)

        plt.barh(top_k_features.index, top_k_features["mean"], yerr=1.96*top_k_features["sem"])

        plt.tight_layout()
    
    def get_shape_function(self, feature_name, is_cat_col=False):
        
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
                
                # replace feature bins with ordinal encoded values
                # number of feature bins is number of unique categories + 1, and 
                # then one additional for the missing bin (represented by -1)
                
                # feat_bin_values = np.arange(-1, len(model.feature_bins[col_index])+1)
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
        
        # if is_cat_col:
        #     # Replace bin values with category names
        #     # ord_encoder = self.preprocessor.transformers_[0][1]
        #     # feat_index = np.where(ord_encoder.feature_names_in_ == feature_name)[0][0]
        #     # cat_feat_map = {-1: "NA"}
        #     # for i, cat in enumerate(ord_encoder.categories_[feat_index]):
        #     #     cat_feat_map[i] = cat
            
        #     cat_feat_map = {i: cat for i, cat in enumerate(model.feature_bins[col_index])}
                
        #     df["bin"] = df["bin"].map(cat_feat_map)
            
        return df
    
    def plot_shape_function(self, feature_names, is_cat_cols=None, plot_missing_bin=False):
        
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            
        if is_cat_cols is None:
            is_cat_cols = [False] * len(feature_names)    
        elif isinstance(is_cat_cols, bool):
            is_cat_cols = [is_cat_cols]
        
        num_axes = sum([2 if not c and plot_missing_bin else 1 for _, c in zip(feature_names, is_cat_cols)])
        
        if plot_missing_bin:
        
            fig = plt.figure(figsize=(4*num_axes, 4))
            
            gs = gridspec.GridSpec(1, 2*len(feature_names) + len(feature_names)-1, width_ratios=[10, 1, 2] * (len(feature_names)-1) + [10, 1])  # Add an extra column for the space
            def generate_indices(n_times):
                indices = []
                for i in range(n_times):
                    indices.extend([i*3, i*3+1])
                return indices

            axes = [fig.add_subplot(gs[i]) for i in generate_indices(len(feature_names))]
            
        else:
            fig, axes = plt.subplots(1, num_axes, figsize=(4*num_axes, 4))
        
        ax_idx = 0
        
        for feature_name, is_cat_col in zip(feature_names, is_cat_cols):
            shape_data = self.get_shape_function(feature_name, is_cat_col=is_cat_col)
            
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
                    axes[ax_idx].set_xlabel(feature_name)
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
                    axes[ax_idx].set_xlabel(feature_name)
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
                sns.barplot(x="bin", y="score", data=shape_data, ci=95, ax=axes[ax_idx])
                ax_idx += 1
                
        plt.tight_layout()
        
    def get_pair_shape_function(self, feat1_name, feat2_name, is_feat1_cat_col=False, is_feat2_cat_col=False):
        
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
            
            feat1_nbins = len(model.feature_bins[col1_index])
            feat2_nbins = len(model.feature_bins[col2_index])
            
            grid = self.make_grid(np.arange(0, feat1_nbins+2), np.arange(0, feat2_nbins+2))
            
            if is_feat1_cat_col and not is_feat2_cat_col:
                pair_bin_scores = pair_bin_scores[grid[:, 0] > 0]
            elif not is_feat1_cat_col and is_feat2_cat_col:
                pair_bin_scores = pair_bin_scores[grid[:, 1] > 0]
            else:
                pair_bin_scores = pair_bin_scores[grid.prod(axis=1) > 0]
            
            # Get bin values
            if is_feat1_cat_col:
                feat1_bin_values = np.arange(-1, len(model.feature_bins[col1_index])+1)
            else:
                feat1_bin_values = np.concatenate([
                    [model.feature_bins[col1_index].min() - 0.01],
                    model.feature_bins[col1_index]
                ])
                
            if is_feat2_cat_col:
                feat2_bin_values = np.arange(-1, len(model.feature_bins[col2_index])+1)
            else:
                feat2_bin_values = np.concatenate([
                    [model.feature_bins[col2_index].min() - 0.01],
                    model.feature_bins[col2_index]
                ])

            pair_bin_values = self.make_grid(feat1_bin_values, feat2_bin_values)
            
            bin_scores.append(pair_bin_scores)
            bin_values.append(pair_bin_values)
                
        return bin_scores, bin_values
    
    def make_grid(self, x, y):
        grid = np.meshgrid(x, y, indexing="ij")
        grid = np.stack(grid, axis=-1)
        return grid.reshape(-1, 2)
    
    def plot_pair_shape_function(self, feat1_name, feat2_name, is_feat1_cat_col=False, is_feat2_cat_col=False):
        plt.figure(figsize=(4, 4))
        pair_scores, pair_values = self.get_pair_shape_function(feat1_name, feat2_name, is_feat1_cat_col=is_feat1_cat_col, is_feat2_cat_col=is_feat2_cat_col)
        
        pair_scores = np.mean(np.stack(pair_scores), axis=0)
        pair_values = pair_values[0]

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

        sns.heatmap(pair_data_dnamite)
        
        plt.tight_layout()
      
class DNAMiteRegressor(BaseDNAMiteModel):
    """
    DNAMite model for regression
    n_features: number of features in the data
    n_embed: embedding dimension
    n_hidden: hidden layer dimension
    max_bins: maximum number of bins to discretize each feature
    validation_size: proportion of data to use for validation
    n_val_splits: number of validation splits to average over
    learning_rate: learning rate for the optimizer
    max_epochs: maximum number of epochs to train for
    batch_size: batch size for training
    device: device to run the model on
    **kwargs: other arguments to pass to BaseSingleSplitDNAMiteModel
    """
    
    def __init__(
        self, 
        n_features, 
        n_embed=32,
        n_hidden=32, 
        max_bins=32,
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
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=1,
            max_bins=max_bins,
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, labels in tqdm(train_loader, leave=False):

            X_main, labels = X_main.to(self.device), labels.to(self.device)

            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main).squeeze(-1)
            y_pred += y_pred_init
            
            loss = F.mse_loss(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=True)

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
                    loss += model.loss_penalty(mains=True)
                    
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
            for X_main, X_pairs, labels in tqdm(test_loader, leave=False):

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

class DNAMiteBinaryClassifier(BaseDNAMiteModel):
    """ 
    # DNAMite Model for binary classification
    # n_features: number of features in the data
    # n_embed: embedding dimension
    # n_hidden: hidden layer dimension
    # max_bins: maximum number of bins to discretize each feature
    # validation_size: proportion of data to use for validation
    # n_val_splits: number of validation splits to average over
    # learning_rate: learning rate for the optimizer
    # max_epochs: maximum number of epochs to train for
    # batch_size: batch size for training
    # device: device to run the model on
    # **kwargs: other arguments to pass to BaseSingleSplitDNAMiteModel
    """
    
    
    def __init__(
        self, 
        n_features, 
        n_embed=32,
        n_hidden=32, 
        max_bins=32,
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
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=1,
            max_bins=max_bins,
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, labels in tqdm(train_loader, leave=False):

            X_main, labels = X_main.to(self.device), labels.to(self.device)
            
            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main).squeeze(-1)
            y_pred += y_pred_init
            
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader, partialed_indices=None):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X_main, labels in tqdm(test_loader, leave=False):

                X_main, labels = X_main.to(self.device), labels.to(self.device)

                y_pred = model(mains=X_main).squeeze(-1)
        
                preds.append(y_pred.detach())
            
                loss = F.binary_cross_entropy_with_logits(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=True)
                    
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
            for X_main, X_pairs, labels in tqdm(test_loader, leave=False):

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
    

class DNAMiteMulticlassClassifier(BaseDNAMiteModel):
    """  
    # DNAMite Model for multiclass classification
    # n_features: number of features in the data
    # n_embed: embedding dimension
    # n_hidden: hidden layer dimension
    # n_classes: number of classes
    # max_bins: maximum number of bins to discretize each feature
    # validation_size: proportion of data to use for validation
    # n_val_splits: number of validation splits to average over
    # learning_rate: learning rate for the optimizer
    # max_epochs: maximum number of epochs to train for
    # batch_size: batch size for training
    # device: device to run the model on
    # **kwargs: other arguments to pass to BaseSingleSplitDNAMiteModel
    """
    
    def __init__(
        self, 
        n_features, 
        n_classes,
        n_embed=32,
        n_hidden=32, 
        max_bins=32,
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
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=n_classes,
            max_bins=max_bins,
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, labels in tqdm(train_loader, leave=False):

            X_main, labels = X_main.to(self.device), labels.to(self.device)
            
            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main).squeeze(-1)
            y_pred += y_pred_init
            
            loss = F.cross_entropy(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty(mains=True)

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
            
                loss = F.cross_entropy(y_pred, labels)
                
                if model.penalized:
                    loss += model.loss_penalty(mains=True)
                    
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
            for X_main, X_pairs, labels in tqdm(test_loader, leave=False):

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
        # Default for multiclass is to compute importances summed across all classes
        
        importances = []
        for i, model in enumerate(self.models):
            
            model.compute_intercept(model.bin_counts, ignore_missing_bin=ignore_missing_bin)
            
            importances = []
            for i, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[i].sum(axis=-1)
                
                feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(model.bin_counts[i])) / np.sum(model.bin_counts[i])
                    
                importances.append([i, col, feat_importance, len(feat_bin_scores)])
            
            if self.fit_pairs:
                pairs_list = model.pairs_list.cpu().numpy()
                model.compute_pairs_intercept(model.pair_bin_counts)
                    
                for i, pair in enumerate(pairs_list):
                    pair_bin_scores = model.pair_bin_scores[i].sum(axis=-1)
                    
                    pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[i])) / np.sum(model.pair_bin_counts[i])
                        
                    importances.append([i, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])
            
        self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    
    def get_shape_function(self, feature_name, label_id, is_cat_col=False):
        
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


class DNAMiteSurvival(BaseDNAMiteModel):
    """
    # DNAMite model for survival analysis
    # Outputs predictions P(T <= t | X) for a set number of eval times t
    # Uses the IPCW / Brier Loss for classification prediction
    # n_features: number of features in the data
    # n_embed: embedding dimension
    # n_hidden: hidden layer dimension
    # n_output: number of outputs, i.e. the number of eval times
    # max_bins: maximum number of bins to discretize each feature
    # validation_size: proportion of data to use for validation
    # n_val_splits: number of validation splits to average over
    # learning_rate: learning rate for the optimizer
    # max_epochs: maximum number of epochs to train for
    # batch_size: batch size for training
    # device: device to run the model on
    # **kwargs: other arguments to pass to BaseSingleSplitDNAMiteModel
    """
    
    def __init__(
        self, 
        n_features, 
        n_eval_times=100,
        n_embed=32,
        n_hidden=32, 
        max_bins=32,
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
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_output=n_eval_times,
            max_bins=max_bins,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            **kwargs    
        )
            
        
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
    
    def train_epoch_mains(self, model, train_loader, optimizer, partialed_indices=None):
        model.train()
        total_loss = 0
        
        active_feats = model.active_feats
        y_pred_init = 0

        for X_main, events, times, pcw_obs_times in tqdm(train_loader, leave=False):

            X_main, events, times, pcw_obs_times = X_main.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

            if partialed_indices is not None:
                with torch.no_grad():
                    model.active_feats = torch.tensor(partialed_indices).to(self.device)
                    y_pred_init = model(mains=X_main)
                    
                    # Reset active_feats
                    model.active_feats = torch.tensor([i for i in active_feats if i not in partialed_indices]).to(self.device)

            y_pred = model(mains=X_main)
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
            
        model.active_feats = active_feats
        return total_loss / len(train_loader)
    
    def test_epoch_mains(self, model, test_loader):
        model.eval()
        total_loss = 0
        preds = []
        
        active_feats = model.active_feats
        y_pred_init = 0
        
        with torch.no_grad():
            for X_main, events, times, pcw_obs_times in tqdm(test_loader, leave=False):

                X_main, events, times, pcw_obs_times = X_main.to(self.device), events.to(self.device), times.to(self.device), pcw_obs_times.to(self.device)

                y_pred = model(mains=X_main)
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
                    loss += model.loss_penalty(mains=False)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)   
    
    def select_features(self, X, y, partialed_feats=None):
        self.cde = CensoringDistributionEstimator()
        self.cde.fit(y)
    
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
        
        self.pcw_eval_times = torch.FloatTensor(self.cde.predict_proba(self.eval_times.cpu().numpy())).to(self.device) + 1e-5
        
        return super().select_features(X, y, partialed_feats=partialed_feats)
    
    def fit(self, X, y, partialed_feats=None):
        
        self.cde = CensoringDistributionEstimator()
        self.cde.fit(y)
    
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
        
        self.pcw_eval_times = torch.FloatTensor(self.cde.predict_proba(self.eval_times.cpu().numpy())).to(self.device) + 1e-5
        
        super().fit(X, y, partialed_feats=partialed_feats)
        
        return
    
    def get_feature_importances(self, eval_time=None, ignore_missing_bin=False):
        
        if eval_time is not None:
            eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        
        importances = []
        for split, model in enumerate(self.models):
            
            model.compute_intercept(model.bin_counts, ignore_missing_bin=ignore_missing_bin)
            
            pairs_list = model.pairs_list.cpu().numpy()
            model.compute_pairs_intercept(model.pair_bin_counts)
            
            for i, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[i]
                
                if eval_time is not None:
                    feat_importance = np.sum(np.abs(feat_bin_scores[:, eval_index]) * np.array(model.bin_counts[i])) / np.sum(model.bin_counts[i])
                else:
                    feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(model.bin_counts[i]).reshape(-1, 1)) / np.sum(model.bin_counts[i])
                    
                importances.append([split, col, feat_importance, len(feat_bin_scores)])
                
            for i, pair in enumerate(pairs_list):
                pair_bin_scores = model.pair_bin_scores[i]
                                
                if eval_time is not None:
                    pair_importance = np.sum(np.abs(pair_bin_scores[:, eval_index]) * np.array(model.pair_bin_counts[i])) / np.sum(model.pair_bin_counts[i])
                else:
                    pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[i]).reshape(-1, 1)) / np.sum(model.pair_bin_counts[i])
                    
                importances.append([split, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])
            
        self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    
    def plot_feature_importances(self, k=10, eval_time=None, ignore_missing_bin=False):
        plt.figure(figsize=(6, 4))

        feat_imps = self.get_feature_importances(eval_time=eval_time, ignore_missing_bin=ignore_missing_bin)
        
        top_k_features = feat_imps.groupby("feature")["importance"].agg(["mean", "sem"])
        top_k_features = top_k_features.sort_values("mean", ascending=True).tail(k)

        plt.barh(top_k_features.index, top_k_features["mean"], yerr=1.96*top_k_features["sem"])

        plt.tight_layout()
    
    def get_shape_function(self, feature_name, eval_time, is_cat_col=False):
        
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
                
                # replace feature bins with ordinal encoded values
                # number of feature bins is number of unique categories + 1, and 
                # then one additional for the missing bin (represented by -1)
                
                # feat_bin_values = np.arange(-1, len(model.feature_bins[col_index])+1)
                feat_bin_values = model.feature_bins[col_index]
                print("FEAT BIN VALUE FOR MODEL", i, ":", feat_bin_values)
                # print("FEAT BIN VALUE FOR MODEL", i, ":", feat_bin_values)
                
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
        
        # if is_cat_col:
        #     # Replace bin values with category names
        #     # ord_encoder = self.preprocessor.transformers_[0][1]
        #     # feat_index = np.where(ord_encoder.feature_names_in_ == feature_name)[0][0]
        #     # cat_feat_map = {-1: "NA"}
        #     # for i, cat in enumerate(ord_encoder.categories_[feat_index]):
        #     #     cat_feat_map[i] = cat
            
        #     cat_feat_map = {i: cat for i, cat in enumerate(model.feature_bins[col_index])}
                
        #     df["bin"] = df["bin"].map(cat_feat_map)
            
        return df
    
    def plot_shape_function(self, feature_names, eval_time, is_cat_cols=None, plot_missing_bin=False):
        
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            
        if is_cat_cols is None:
            is_cat_cols = [False] * len(feature_names)    
        elif isinstance(is_cat_cols, bool):
            is_cat_cols = [is_cat_cols]
        
        num_axes = sum([2 if not c and plot_missing_bin else 1 for _, c in zip(feature_names, is_cat_cols)])
        
        if plot_missing_bin:
        
            fig = plt.figure(figsize=(4*num_axes, 4))
            
            gs = gridspec.GridSpec(1, 2*len(feature_names) + len(feature_names)-1, width_ratios=[10, 1, 2] * (len(feature_names)-1) + [10, 1])  # Add an extra column for the space
            def generate_indices(n_times):
                indices = []
                for i in range(n_times):
                    indices.extend([i*3, i*3+1])
                return indices

            axes = [fig.add_subplot(gs[i]) for i in generate_indices(len(feature_names))]
            
        else:
            fig, axes = plt.subplots(1, num_axes, figsize=(4*num_axes, 4))
        
        ax_idx = 0
        
        for feature_name, is_cat_col in zip(feature_names, is_cat_cols):
            shape_data = self.get_shape_function(feature_name, eval_time, is_cat_col=is_cat_col)
            
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
                    axes[ax_idx].set_xlabel(feature_name)
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
                    axes[ax_idx].set_xlabel(feature_name)
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
                print(shape_data["bin"].value_counts())
                sns.barplot(x="bin", y="score", data=shape_data, errorbar=('ci', 95), ax=axes[ax_idx])
                # make the x-axis labels tilted
                axes[ax_idx].tick_params(axis='x', rotation=45)
                
                ax_idx += 1
                
        plt.tight_layout()
    
    def get_pair_shape_function(self, feat1_name, feat2_name, eval_time, is_feat1_cat_col=False, is_feat2_cat_col=False):
        
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
            
            feat1_nbins = len(model.feature_bins[col1_index])
            feat2_nbins = len(model.feature_bins[col2_index])
            
            grid = self.make_grid(np.arange(0, feat1_nbins+2), np.arange(0, feat2_nbins+2))
            
            if is_feat1_cat_col and not is_feat2_cat_col:
                pair_bin_scores = pair_bin_scores[grid[:, 0] > 0]
            elif not is_feat1_cat_col and is_feat2_cat_col:
                pair_bin_scores = pair_bin_scores[grid[:, 1] > 0]
            else:
                pair_bin_scores = pair_bin_scores[grid.prod(axis=1) > 0]
            
            # Get bin values
            if is_feat1_cat_col:
                feat1_bin_values = np.arange(-1, len(model.feature_bins[col1_index])+1)
            else:
                feat1_bin_values = np.concatenate([
                    [model.feature_bins[col1_index].min() - 0.01],
                    model.feature_bins[col1_index]
                ])
                
            if is_feat2_cat_col:
                feat2_bin_values = np.arange(-1, len(model.feature_bins[col2_index])+1)
            else:
                feat2_bin_values = np.concatenate([
                    [model.feature_bins[col2_index].min() - 0.01],
                    model.feature_bins[col2_index]
                ])

            pair_bin_values = self.make_grid(feat1_bin_values, feat2_bin_values)
            
            bin_scores.append(pair_bin_scores)
            bin_values.append(pair_bin_values)
                
        return bin_scores, bin_values
    
    def predict(self, X_test, y_test=None):
        
        # self.selected_pair_indices = list(combinations(range(self.n_features), 2))
        
        X_test_discrete = self.discretize_data(X_test)
        
        # Make placeholder y_test
        if y_test is None:
            y_test = np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        X_test_interactions = X_test_discrete.values[:, self.selected_pair_indices]
        
        test_loader = self.get_data_loader(X_test_discrete, y_test, pairs=X_test_interactions, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0], self.n_output))
        for i, model in enumerate(self.models):
            test_loss, model_preds = self.test_epoch_pairs(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
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
    
    def get_feature_importances(self, eval_time, ignore_missing_bin=False):
        
        eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        
        importances = []
        for i, model in enumerate(self.models):
            
            model.compute_intercept(model.bin_counts, ignore_missing_bin=ignore_missing_bin)
            
            pairs_list = model.pairs_list.cpu().numpy()
            model.compute_pairs_intercept(model.pair_bin_counts)
            
            for j, col in enumerate(model.feature_names_in_):
                feat_bin_scores = model.bin_scores[j][:, eval_index]
                
                feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(model.bin_counts[j])) / np.sum(model.bin_counts[j])
                    
                importances.append([i, col, feat_importance, len(feat_bin_scores)])
                
            for j, pair in enumerate(pairs_list):
                pair_bin_scores = model.pair_bin_scores[j][:, eval_index]
                
                pair_importance = np.sum(np.abs(pair_bin_scores) * np.array(model.pair_bin_counts[j])) / np.sum(model.pair_bin_counts[j])
                    
                importances.append([i, f"{model.feature_names_in_[pair[0]]} || {model.feature_names_in_[pair[1]]}", pair_importance, len(pair_bin_scores)])
            
        self.importances = pd.DataFrame(importances, columns=["split", "feature", "importance", "n_bins"])
        return self.importances
    