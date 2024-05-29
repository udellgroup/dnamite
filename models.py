import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from sa_transformer import (
    Encoder, 
    PositionwiseFeedForward, 
    EncoderLayer, 
    MultiHeadedAttention,
    TranDataset,
    SrcEmbed,
    TranFinalLayer
)

# Class for DNAMite model
# n_features: number of input features
# n_embed: embedding dimension for each feature
# n_hidden: hidden layer dimension for each feature
# n_output: output dimension for each feature
# feature_sizes: list containing the number of unique values/bins for each feature
# categorical_feats: list of indices of categorical features, None means no cat features
# n_layers: number of hidden layers for each feature MLP
# gamma: gamma parameter for smooth-step function
# reg_param: regularization parameter for feature sparsity
# main_pair_strength: regularization parameter for pair sparsity
# entropy_param: regularization parameter for entropy
# has_pairs: whether the model is trained with pairs
# n_pairs: number of pairs, None means all pairs
# device: device to run the model on
# strong_hierarchy: whether to enforce strong hierarchy on pairs
# pairs_list: list of pairs, None means all pairs
# kernel_size: kernel size for kernel smoothing, 0 means no smoothing
# kernel_weight: weight for kernel smoothing
# pair_kernel_size: kernel size for pair kernel smoothing, 0 means no smoothing
# pair_kernel_weight: weight for pair kernel smoothing
class DNAMite(nn.Module):
    def __init__(self, 
                 n_features, 
                 n_embed,
                 n_hidden, 
                 n_output,
                 feature_sizes, 
                 categorical_feats=None,
                 n_layers=2,
                 gamma=1, 
                 reg_param=0.1, 
                 main_pair_strength=1.0, 
                 entropy_param=1e-3, 
                 has_pairs=True, 
                 n_pairs=None, 
                 device="cpu", 
                 strong_hierarchy=False, 
                 pairs_list=None, 
                 kernel_size=0,
                 kernel_weight=1,
                 pair_kernel_size=None,
                 pair_kernel_weight=None
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
        self.main_pair_strength = main_pair_strength
        self.entropy_param = entropy_param 
        self.has_pairs = has_pairs
        self.main_effects_frozen = False
        self.device=device
        self.strong_hierarchy=strong_hierarchy
        self.kernel_size = kernel_size
        self.kernel_weight = kernel_weight
        
        if pair_kernel_size is None:
            self.pair_kernel_size = self.kernel_size
        else:
            self.pair_kernel_size = pair_kernel_size
            
        if pair_kernel_weight is None:
            self.pair_kernel_weight = self.kernel_weight
        else:
            self.pair_kernel_weight = pair_kernel_weight
        
        # Create mask that has True only for categorical features
        self.cat_feat_mask = torch.zeros(n_features, dtype=torch.bool).to(device)
        if categorical_feats is not None:
            self.cat_feat_mask[torch.IntTensor(categorical_feats).to(device)] = True
        
        
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
        
        if has_pairs:
            if n_pairs is None:
                n_pairs = int((n_features * (n_features - 1)) / 2)
                self.pairs_list = torch.IntTensor(list(combinations(range(n_features), 2))).to(device)
            else:
                if self.strong_hierarchy:
                    assert pairs_list is not None, \
                        "Pairs list is None, but n_pairs is provided, meaning not all pairs are given"
                
                self.pairs_list = pairs_list.to(device)
                
            self.init_pairs_params(n_pairs)
            
            self.active_pairs = torch.arange(n_pairs).to(device)
            
        self.reset_parameters()
        
        # import itertools
        # self.feature_offsets = [0] + list(itertools.accumulate(feature_sizes))[:-1]
        # self.feature_offsets = torch.tensor(self.feature_offsets).to(device)
        self.feature_offsets = torch.cat([
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
    
    def _get_pairs_neighbors(self, pairs):
        
        # Add neighboring indices to pairs in new dimension
        # where neighbors come from square around the pair
        # pair kernel size comes from relationship (2x+1)^2 = 2k + 1
        pair_kernel_size = (np.sqrt(2 * self.pair_kernel_size + 1) - 1) // 2
        
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
        pair_kernel_size = (np.sqrt(2 * self.pair_kernel_size + 1) - 1) // 2
        
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
            -torch.square(kernel_offset).sum(dim=2).to(self.device)
        ) / (2 * self.pair_kernel_weight)
        
        return weights
        
        

    def forward(self, mains=None, pairs=None, mains_raw=None):
        # mains of shape (batch_size, n_features)
        # pairs of shape (batch_size, |n_features choose 2|, 2)
        
        output_main = 0
        output_pairs = 0
        
        if mains is not None:
            
            mains = mains[:, self.active_feats]
            
            # Add offsets to features to get the correct indices
            offsets = self.feature_offsets.unsqueeze(0).expand(mains.size(0), -1).to(self.device)
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
                    (mains <= self.feature_offsets[self.active_feats].reshape(1, -1, 1)) | (mains >= self.feature_offsets[self.active_feats].reshape(1, -1, 1) + self.feature_sizes[self.active_feats].reshape(1, -1, 1)),
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
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (n_pairs, 2)
            offsets = self.feature_offsets[self.pairs_list[self.active_pairs]].to(self.device)
            pairs = pairs + offsets
            
            if self.pair_kernel_size > 0:
            
                # Pairs will now be shape (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2)
                pairs = self._get_pairs_neighbors(pairs)
                
                # Get the kernel weights using gaussian kernel
                # Will be shape (2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = self._get_pair_weights()
                
                pair_sizes = self.feature_sizes[self.pairs_list[self.active_pairs]].to(self.device)
                
                # Add a <= instead of < on left side so eliminate weight on missing bin
                weights = torch.where(
                    (pairs[:, :, :, :, 0] <= offsets[:, 0].reshape(1, -1, 1, 1)) | (pairs[:, :, :, :, 1] <= offsets[:, 1].reshape(1, -1, 1, 1)) | \
                        (pairs[:, :, :, :, 0] >= offsets[:, 0].reshape(1, -1, 1, 1) + pair_sizes[:, 0].reshape(1, -1, 1, 1)) | (pairs[:, :, :, :, 1] >= offsets[:, 1].reshape(1, -1, 1, 1) + pair_sizes[:, 1].reshape(1, -1, 1, 1)),
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
        
        
            # pairs = torch.einsum(
            #     'ijk,jkl->ijl', 
            #     pairs, 
            #     self.first_weights_pairs[self.active_pairs, :, :]
            # ) + self.first_biases_pairs[self.active_pairs, :]
            # pairs = F.relu(pairs)
            
            # pairs = torch.einsum(
            #     'ijk,jkl->ijl', 
            #     pairs, 
            #     self.hidden_weights_pairs[self.active_pairs, :, :]
            # ) + self.hidden_biases_pairs[self.active_pairs, :]
            # pairs = F.relu(pairs)
            
            # pairs = torch.einsum(
            #     'ijk,jkl->ijl', 
            #     pairs, 
            #     self.output_weights_pairs[self.active_pairs, :, :]
            # ) + self.output_biases_pairs[self.active_pairs, :]

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
            # offsets = torch.tensor(self.feature_offsets).unsqueeze(0).expand(inputs.size(0), -1).to(self.device)
            offset = self.feature_offsets[feat_index]
            inputs = inputs + offset
            
            if self.kernel_size > 0:
                
                # Apply kernel smoothing to the embeddings
                
                inputs = inputs.unsqueeze(-1) + torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)
                weights = torch.exp(-torch.square(torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)) / (2 * self.kernel_weight))
                
                weights = torch.where(
                    (inputs < offset) | (inputs >= offset + self.feature_sizes[feat_index]),
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
                torch.arange(size1), torch.arange(size2)
            ), dim=-1).reshape(-1, 2).to(self.device)
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (2)
            offsets = self.feature_offsets[self.pairs_list[pair_index]].to(self.device)
            pairs = pairs + offsets
            
            if self.pair_kernel_size > 0:
            
                # Pairs will now be shape (batch_size, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2)
                pairs = self._get_pairs_neighbors(pairs)
                
                # Get the kernel weights using gaussian kernel
                # Will be shape (2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
                weights = self._get_pair_weights()
                
                pair_sizes = self.feature_sizes[self.pairs_list[pair_index]].to(self.device)
                
                weights = torch.where(
                    (pairs[:, :, :, 0] < offsets[0]) | (pairs[:, :, :, 1] < offsets[1]) | \
                        (pairs[:, :, :, 0] >= offsets[0] + pair_sizes[0]) | (pairs[:, :, :, 1] >= offsets[1] + pair_sizes[1]),
                    torch.zeros_like(weights), 
                    weights
                )
                
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
            assert self.has_pairs, "Tried to prune pairs parameters without model having pairs"
            
            z_pairs = self.get_smooth_z_pairs()
            self.active_pairs = torch.where(z_pairs > 0)[0] 
            
    def compute_intercept(self, bin_counts):
        
        # bin counts is a list of bin counts for each feature
        
        self.bin_counts = bin_counts
        self.feat_offsets = torch.zeros(self.n_features, self.n_output).to(self.device)
        
        for feat_idx in self.active_feats:
            bin_preds = self.get_bin_scores(feat_idx, center=False)
            feat_bin_counts = torch.tensor(bin_counts[feat_idx]).to(self.device).unsqueeze(-1)
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

# NAM model, which is similar to DNAMite but without binning / kernel smoothing
class NAM(nn.Module):
    def __init__(
        self, 
        n_features, 
        n_hidden, 
        n_output, 
        n_layers=2, 
        exu=False,
        gamma=1, 
        reg_param=0.1, 
        main_pair_strength=1.0, 
        entropy_param=1e-3, 
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
        pairs_x = grid_feat1.reshape(-1)
        pairs_y = grid_feat2.reshape(-1)
        
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

# Transformer model for survival analysis
# Based on this paper: https://proceedings.mlr.press/v146/hu21a/hu21a.pdf
class SATransformer(nn.Module):
    def __init__(self, *, n_feats, d_model, n_eval_times, n_layers, num_heads, d_ff, drop_prob, device="cpu"):
        super().__init__()
        
        self.n_eval_times = n_eval_times
        
        self.embed = SrcEmbed(n_feats, d_model)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=drop_prob,
                batch_first=True
            ),
            n_layers
        )
        self.final_layer = TranFinalLayer(d_model)
        
        # Create the positional encoding
        self.pe = torch.zeros(n_eval_times, d_model)
        position = torch.arange(0, n_eval_times).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).to(device)
        
    def forward(self, x):
        
        # Embed features
        embeds = self.embed(x)
        
        # Stack embeddings and add positional encoding
        embeds = embeds.unsqueeze(1).repeat(1, self.n_eval_times, 1)
        embeds += self.pe
        
        # Pass through transformer
        output = self.transformer(embeds)
        
        # Return final layer output
        return self.final_layer(output)