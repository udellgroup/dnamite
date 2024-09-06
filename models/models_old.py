import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations
            
# Neural Additive Model with discretized features. Supports
# 1) Kernel smoothing to smooth shape functions.
# 2) Pairwise interactions between features.
# 3) Feature sparsity via learnable binary gates.
# Params
# n_features: Number of features
# n_embed: Embedding size for each feature
# n_hidden: Number of hidden units in each feature's network
# n_output: Number of output units
# feature_sizes: List of number of unique values for each feature
# gamma: parameter controlling steepness of smooth-step function (see https://openreview.net/pdf?id=F5DYsAc7Rt)
# reg_param: Regularization parameter controlling strength of feature sparsity
# main_pair_strength: Regularization parameter controlling relative strength of pairs vs main effects
# entropy_param: Regularization parameter controlling how fast binary gates convergence to 0-1
# has_pairs: Whether to include pairwise interactions
# n_pairs: Number of pairs fitted. Only set if pairs_list is also set.
# device: Device to run model on
# pairs_list: List of pairs to fit. If None, all pairs are fitted.
# kernel_size: Size of kernel for kernel smoothing. Set to 0 for no kernel smoothing.
# kernel_weight: Weight of kernel for kernel smoothing.
class DiscreteNAM(nn.Module):
    def __init__(self, 
                 n_features, 
                 n_embed,
                 n_hidden, 
                 n_output,
                 feature_sizes, 
                 gamma=1, 
                 reg_param=0.0, 
                 main_pair_strength=1.0, 
                 entropy_param=0.0, 
                 has_pairs=False, 
                 n_pairs=None, 
                 device="cpu", 
                 pairs_list=None, 
                 kernel_size=0,
                 kernel_weight=1
                 ):
        super().__init__()
        self.n_features = n_features
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.feature_sizes = torch.tensor(feature_sizes).to(device)
        self.gamma = gamma
        self.reg_param = reg_param
        self.main_pair_strength = main_pair_strength
        self.entropy_param = entropy_param 
        self.has_pairs = has_pairs
        self.main_effects_frozen = False
        self.device=device
        self.kernel_size = kernel_size
        self.kernel_weight = kernel_weight
        
        if self.reg_param == 0:
            self.penalized = False
        else:
            self.penalized = True
        
        self.first_weights_main = nn.Parameter(torch.empty(n_features, n_embed, n_hidden))
        self.first_biases_main = nn.Parameter(torch.empty(n_features, n_hidden))
        
        self.hidden_weights_main = nn.Parameter(torch.empty(n_features, n_hidden, n_hidden))
        self.hidden_biases_main = nn.Parameter(torch.empty(n_features, n_hidden))
        
        self.output_weights_main = nn.Parameter(torch.empty(n_features, n_hidden, n_output))
        self.output_biases_main = nn.Parameter(torch.empty(n_features, n_output))
        
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
        
        # Normally, NAMs use a separate embedding for each feature
        # However, this is very slow because it requires a manual for loop
        # Alternative here is to stack all the feature embedding into one "super" embedding module
        self.feature_offsets = torch.cat([
            torch.tensor([0]), 
            torch.cumsum(torch.tensor(feature_sizes), dim=0)[:-1]
        ]).to(device)
        self.total_feature_size = sum(feature_sizes)
        self.embedding = nn.Embedding(self.total_feature_size, n_embed)
            
            
        
    def init_pairs_params(self, n_pairs):
        
        # Set number of pairs to be |n_features choose 2|
        self.n_pairs = n_pairs

        self.first_weights_pairs = nn.Parameter(torch.empty(self.n_pairs, (self.n_embed) * 2, self.n_hidden))
        self.first_biases_pairs = nn.Parameter(torch.empty(self.n_pairs, self.n_hidden))

        self.hidden_weights_pairs = nn.Parameter(torch.empty(self.n_pairs, self.n_hidden, self.n_hidden))
        self.hidden_biases_pairs = nn.Parameter(torch.empty(self.n_pairs, self.n_hidden))

        self.output_weights_pairs = nn.Parameter(torch.empty(self.n_pairs, self.n_hidden, self.n_output))
        self.output_biases_pairs = nn.Parameter(torch.empty(self.n_pairs, self.n_output))

        self.z_pairs = nn.Parameter(torch.empty(self.n_pairs))
        
        self.reset_pairs_parameters()
        
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.first_weights_main, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.hidden_weights_main, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.output_weights_main, a=np.sqrt(5))
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.first_weights_main)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.first_biases_main, -bound, bound)
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.hidden_weights_main)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.hidden_biases_main, -bound, bound)
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_weights_main)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.output_biases_main, -bound, bound)
        
        nn.init.uniform_(self.z_main, -self.gamma/100, self.gamma/100)
        
        if self.has_pairs:
            self.reset_pairs_parameters()
            
            
    def reset_pairs_parameters(self):
        
        nn.init.kaiming_uniform_(self.first_weights_pairs, a=np.sqrt(3))
        nn.init.kaiming_uniform_(self.hidden_weights_pairs, a=np.sqrt(3))
        nn.init.kaiming_uniform_(self.output_weights_pairs, a=np.sqrt(3))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.first_weights_pairs)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.first_biases_pairs, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.hidden_weights_pairs)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.hidden_biases_pairs, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_weights_pairs)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.output_biases_pairs, -bound, bound)

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
        pair_kernel_size = (np.sqrt(2 * self.kernel_size + 1) - 1) // 2
        
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
        pair_kernel_size = (np.sqrt(2 * self.kernel_size + 1) - 1) // 2
        
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
        ) / 2
        
        return weights
        

    def forward(self, mains=None, pairs=None):
        # mains of shape (batch_size, n_features)
        # pairs of shape (batch_size, |n_features choose 2|, 2)
        
        output_main = 0
        output_pairs = 0
        
        if mains is not None:
            
            mains = mains[:, self.active_feats]
            
            # Add offsets to features to get the correct indices
            offsets = self.feature_offsets.unsqueeze(0).expand(mains.size(0), -1).to(self.device)
            mains = mains + offsets
            
            # Add neighboring indices to mains in new dimension
            # mains is now of shape (batch_size, n_features, 2 * kernel_size + 1)
            mains = mains.unsqueeze(-1) + torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)
            
            # Add the embeddings for each feature using weight function
            weights = torch.exp(-torch.square(torch.arange(-self.kernel_size, self.kernel_size+1).to(self.device)) / (2 * self.kernel_weight))
            
            weights = torch.where(
                (mains < self.feature_offsets.reshape(1, -1, 1)) | (mains >= self.feature_offsets.reshape(1, -1, 1) + self.feature_sizes.reshape(1, -1, 1)),
                torch.zeros_like(weights),
                weights
            )
            
            mains = torch.sum(
                self.embedding(
                    torch.clamp(mains.long(), 0, self.total_feature_size-1)
                ) * weights.unsqueeze(-1), 
                dim=2
            )
            
            mains = torch.einsum(
                'ijk,jkl->ijl', 
                mains, 
                self.first_weights_main[self.active_feats, :, :]
            ) + self.first_biases_main[self.active_feats, :]
            mains = F.relu(mains)

            mains = torch.einsum(
                'ijk,jkl->ijl', 
                mains, 
                self.hidden_weights_main[self.active_feats, :, :]
            ) + self.hidden_biases_main[self.active_feats, :]
            mains = F.relu(mains)

            mains = torch.einsum(
                'ijk,jkl->ijl', 
                mains, 
                self.output_weights_main[self.active_feats, :, :]
            ) + self.output_biases_main[self.active_feats, :]

            # Get smoothed z
            z_main = self.get_smooth_z()[self.active_feats]

            output_main = torch.einsum('ijk,j->ik', mains, z_main)
        
        if pairs is not None:
            
            pairs = pairs[:, self.active_pairs, :]
            
            # Add offsets to features to get the correct indices
            # offsets is of shape (n_pairs, 2)
            offsets = self.feature_offsets[self.pairs_list[self.active_pairs]].to(self.device)
            pairs = pairs + offsets
            
            # Pairs will now be shape (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2)
            pairs = self._get_pairs_neighbors(pairs)
            
            # Get the kernel weights using gaussian kernel
            # Will be shape (2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1)
            weights = self._get_pair_weights()
            
            pair_sizes = self.feature_sizes[self.pairs_list[self.active_pairs]].to(self.device)
            
            weights = torch.where(
                (pairs[:, :, :, :, 0] < offsets[:, 0].reshape(1, -1, 1, 1)) | (pairs[:, :, :, :, 1] < offsets[:, 1].reshape(1, -1, 1, 1)) | \
                    (pairs[:, :, :, :, 0] >= offsets[:, 0].reshape(1, -1, 1, 1) + pair_sizes[:, 0].reshape(1, -1, 1, 1)) | (pairs[:, :, :, :, 1] >= offsets[:, 1].reshape(1, -1, 1, 1) + pair_sizes[:, 1].reshape(1, -1, 1, 1)),
                torch.zeros_like(weights), 
                weights
            )
            
            # pairs has shape (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2, n_embed)
            pairs = self.embedding(
                torch.clamp(pairs.long(), 0, self.total_feature_size-1)
            )
            
            # concat last two dimensions of pairs
            # pairs now has shape (batch_size, n_pairs, 2 * pair_kernel_size + 1, 2 * pair_kernel_size + 1, 2 * n_embed)
            # pairs = pairs.reshape(pairs.size(0), pairs.size(1), pairs.size(2), pairs.size(3), 2*self.n_embed)
            
            
            # pairs now has shape (batch_size, n_pairs, 2, n_embed)
            pairs = torch.sum(
                pairs * weights.unsqueeze(-1).unsqueeze(-1), 
                dim=[2, 3]
            )
            
            # Concat embeddings of both features in pair
            # pairs now has shape (batch_size, n_pairs, 2 * n_embed)
            pairs = pairs.reshape(pairs.size(0), pairs.size(1), 2*self.n_embed)
            
        
            pairs = torch.einsum(
                'ijk,jkl->ijl', 
                pairs, 
                self.first_weights_pairs[self.active_pairs, :, :]
            ) + self.first_biases_pairs[self.active_pairs, :]
            pairs = F.relu(pairs)
            
            pairs = torch.einsum(
                'ijk,jkl->ijl', 
                pairs, 
                self.hidden_weights_pairs[self.active_pairs, :, :]
            ) + self.hidden_biases_pairs[self.active_pairs, :]
            pairs = F.relu(pairs)
            
            pairs = torch.einsum(
                'ijk,jkl->ijl', 
                pairs, 
                self.output_weights_pairs[self.active_pairs, :, :]
            ) + self.output_biases_pairs[self.active_pairs, :]
            
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
                return self.entropy_param * omegas_pairs.sum()
            
            else:
                return self.entropy_param * (omegas_main.sum() + omegas_pairs.sum())
        
        else:
        
            return self.entropy_param * omegas_main.sum()
        
    def freeze_main_effects(self):
        self.main_effects_frozen = True
        self.first_weights_main.requires_grad = False
        self.first_biases_main.requires_grad = False
        
        self.hidden_weights_main.requires_grad = False
        self.hidden_biases_main.requires_grad = False
        
        self.output_weights_main.requires_grad = False
        self.output_biases_main.requires_grad = False
        
        self.z_main.requires_grad = False
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
                
                inputs = torch.sum(
                    self.embedding(
                        torch.clamp(inputs.long(), 0, self.total_feature_size-1)
                    ) * weights.unsqueeze(-1),
                    dim=1
                )
                
            else:
                # inputs = self.embeddings[feat_index](inputs.long())
                inputs = self.embedding(inputs.long())
                
                
            hidden_main = torch.einsum(
                'ik,kl->il', 
                inputs, 
                self.first_weights_main[feat_index, :, :]
            ) + self.first_biases_main[feat_index, :]
            hidden_main = F.relu(hidden_main)

            hidden_main = torch.einsum(
                'ik,kl->il', 
                hidden_main, 
                self.hidden_weights_main[feat_index, :, :]
            ) + self.hidden_biases_main[feat_index, :]
            hidden_main = F.relu(hidden_main)

            output_main = torch.einsum(
                'ik,kl->il', 
                hidden_main, 
                self.output_weights_main[feat_index, :, :]
            ) + self.output_biases_main[feat_index, :]

            # Get prediction using the smoothed z
            z_main = self.get_smooth_z()
            prediction = output_main * z_main[feat_index]
            
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
            
            if self.kernel_size > 0:
                
                inputs = self._get_pairs_neighbors(inputs) # shape (N, 2 * kernel_size + 1, 2 * kernel_size + 1, 2)
                weights = self._get_pair_weights()
                
                # Get sizes of both features in pair
                size1 = self.feature_sizes[self.pairs_list[pair_index][0]]
                size2 = self.feature_sizes[self.pairs_list[pair_index][1]]
                
                # if pairs[:, pair_index, :, :, :] is < 0 or > feature_size, set weight to 0
                weights = torch.where(
                    (inputs[:, :, :, 0] < 0) | (inputs[:, :, :, 1] < 0) | (inputs[:, :, :, 0] >= size1) | (inputs[:, :, :, 1] >= size2),
                    torch.zeros_like(weights), 
                    weights
                )
                
                # Concat embeddings of both features in pair across kernel
                inputs = torch.cat([
                    self.embeddings[pair[0]](
                        torch.clamp(inputs[:, :, :, 0].long(), 0, size1-1)
                    ),
                    self.embeddings[pair[1]](
                        torch.clamp(inputs[:, :, :, 1].long(), 0, size2-1)
                    )
                ], dim=-1)
                
                # Apply kernel
                inputs = torch.sum(
                    inputs * weights.unsqueeze(-1), 
                    dim=[1, 2]
                )
                
            else:
                
                inputs = torch.cat([
                    self.embeddings[pair[0]](inputs[:, 0].long()).unsqueeze(1),
                    self.embeddings[pair[1]](inputs[:, 1].long()).unsqueeze(1)
                ], dim=1)
            
            pairs = torch.einsum(
                'ik,kl->il', 
                inputs, 
                self.first_weights_pairs[pair_index, :, :]
            ) + self.first_biases_pairs[pair_index, :]
            pairs = F.relu(pairs)
            
            pairs = torch.einsum(
                'ik,kl->il', 
                pairs, 
                self.hidden_weights_pairs[pair_index, :, :]
            ) + self.hidden_biases_pairs[pair_index, :]
            pairs = F.relu(pairs)
            
            output_pairs = torch.einsum(
                'ik,kl->il', 
                pairs, 
                self.output_weights_pairs[pair_index, :, :]
            ) + self.output_biases_pairs[pair_index, :]
            
            z_pairs = self.get_smooth_z_pairs()
            prediction = output_pairs.squeeze() * z_pairs[pair_index]
            
            if hasattr(self, 'pair_offsets') and center:
                return prediction - self.pair_offsets[pair_index, :]
            elif center:
                print("Intercept not computed. Shape function will not be centered.")
                return prediction
            else:
                return prediction
            
                
    
    def get_interaction_function(
        self, pair_index, feat1_min, feat1_max, feat2_min, feat2_max
    ):
        
        inputs_feat1 = torch.linspace(feat1_min, feat1_max, 100)
        inputs_feat2 = torch.linspace(feat2_min, feat2_max, 100)
        
        # Create 2D grids for x and y
        grid_feat1, grid_feat2 = torch.meshgrid(inputs_feat1, inputs_feat2, indexing='ij')

        # Reshape the grids to 1D tensors to get pairwise combinations
        pairs_x = grid_feat1.reshape(-1)
        pairs_y = grid_feat2.reshape(-1)
        
        inputs = torch.stack([grid_feat1.reshape(-1), grid_feat2.reshape(-1)], dim=1).to(self.device)
        
        hidden_pairs = torch.einsum(
            'ik,kl->il', 
            inputs, 
            self.first_weights_pairs[pair_index, :, :]
        ) + self.first_biases_pairs[pair_index, :]
        hidden_pairs = F.relu(hidden_pairs)

        hidden_pairs = torch.einsum(
            'ik,kl->il', 
            hidden_pairs, 
            self.hidden_weights_pairs[pair_index, :, :]
        ) + self.hidden_biases_pairs[pair_index, :]
        hidden_pairs = F.relu(hidden_pairs)

        output_pairs = torch.einsum(
            'ik,kl->il', 
            hidden_pairs, 
            self.output_weights_pairs[pair_index, :, :]
        ) + self.output_biases_pairs[pair_index, :]

        # Get smoothed z
        z_pairs = self.get_smooth_z_pairs()

        # output_pairs = torch.einsum('ijk,j->ik', output_pairs, z_pairs)
        
        return output_pairs.squeeze() * z_pairs[pair_index], inputs
        
        
        
    
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
        
        for feat_idx in range(self.n_features):
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
        
    
    
class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)
        
        
        