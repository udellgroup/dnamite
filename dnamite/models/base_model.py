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
from tqdm import tqdm
from functools import partial
from dnamite.utils import discretize, get_bin_counts, get_pair_bin_counts
import textwrap
import copy
from dnamite.utils import LoggingMixin
from sklearn.base import BaseEstimator
from .base_single_split import _BaseSingleSplitDNAMiteModel


class BaseDNAMiteModel(BaseEstimator, LoggingMixin):
    """
    BaseDNAMiteModel is the parent class for DNAMite models.

    This class provides the foundational architecture for DNAMite models, including embedding layers, hidden layers, and output layers. It also manages training and validation processes.

    Parameters
    ----------
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
                 kernel_size=5, 
                 kernel_weight=3, 
                 pair_kernel_size=3, 
                 pair_kernel_weight=3, 
                 monotone_constraints=None,
                 num_pairs=0,
                 verbosity=0,
                 random_state=None,
    ):
        LoggingMixin.__init__(self, verbosity=verbosity)
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
        self.kernel_size = kernel_size
        self.kernel_weight = kernel_weight
        self.pair_kernel_size = pair_kernel_size
        self.pair_kernel_weight = pair_kernel_weight
        self.monotone_constraints = monotone_constraints
        self.num_pairs = num_pairs
        self.verbosity = verbosity
        self.random_state = random_state if random_state is not None else np.random.randint(0, 1000)
        self.fit_pairs = self.num_pairs > 0
        
        
    def _set_gamma(self, X):
        if self.gamma is None:
            self.gamma = min(((X.shape[0] / self.batch_size) / 250) / (self.n_hidden / 16), 1.0)
        if self.pair_gamma is None:
            self.pair_gamma = self.gamma / 4
    
    def _infer_data_types(self, X):
        
        self.feature_dtypes_ = []
        for i in range(X.shape[1]):
            if set(X.iloc[:, i].unique()).issubset({0, 1}):
                self.feature_dtypes_.append('binary')
            elif X.iloc[:, i].dtype.name == 'category' or \
                X.iloc[:, i].dtype.name == 'object':
                self.feature_dtypes_.append('categorical')
            else:
                self.feature_dtypes_.append('continuous')
                
        self.cat_feat_mask_ = np.array(self.feature_dtypes_) != 'continuous'
        self.cat_feat_mask_ = torch.tensor(self.cat_feat_mask_).to(self.device)
    
    def _discretize_data(self, X):
        X_discrete = X.copy()
        
        if not hasattr(self, 'feature_dtypes_'):
            self._infer_data_types(X_discrete)
        
        if hasattr(X, 'columns'):
            col_names = X.columns
            X_discrete = X_discrete.values
        
        # If feature_bins is already set, use existing bins
        if hasattr(self, 'feature_bins_'):
            for i in range(self.n_features):
                if self.feature_dtypes_[i] == 'continuous':
                    X_discrete[:, i], _ = discretize(np.ascontiguousarray(X_discrete[:, i]), max_bins=self.max_bins, bins=self.feature_bins_[i])
                elif self.feature_dtypes_[i] == 'binary':
                    ordinal_map = {val: float(j) for j, val in enumerate(self.feature_bins_[i])}
                    X_discrete[:, i] = np.vectorize(ordinal_map.get)(X_discrete[:, i].astype(float))
                else:
                    ordinal_map = {val: float(j) for j, val in enumerate(self.feature_bins_[i])}
                    
                    # Replace the NAs because they are handled incorrectly in dict
                    X_discrete[:, i] = np.where(X_discrete[:, i] == X_discrete[:, i], X_discrete[:, i], "NA")
                    ordinal_map["NA"] = 0.0
                    
                    for val in pd.Series(X_discrete[:, i]).unique():
                        if val not in ordinal_map:
                            # This means that the category was either not seen in training
                            # Or was an infrequent category
                            if self.feature_bins_[i][-1] == "Other":
                                ordinal_map[val] = self.feature_sizes_[i] - 1.0
                            else:
                                ordinal_map[val] = 0.0
                    X_discrete[:, i] = np.vectorize(ordinal_map.get)(X_discrete[:, i])
                
        else:
            self.feature_bins_ = []
            self.feature_sizes_ = []
            self.has_missing_bin_ = [] # different than cat_feat_mask because cat feats can have missing values
            self.has_missing_values_ = [] # different than has_missing_bin because continuous features always have missing bin (to make kernel work correctly)
            self.logger.debug("Discretizing features...")
            for i in range(self.n_features):
                if self.feature_dtypes_[i] == 'continuous':
                    if pd.Series(X_discrete[:, i]).isna().sum() > 0:
                        self.has_missing_values_.append(True)
                    else:
                        self.has_missing_values_.append(False)
                    X_discrete[:, i], bins = discretize(
                        np.ascontiguousarray(X_discrete[:, i]), 
                        max_bins=self.max_bins, 
                        min_samples_per_bin=self.min_samples_per_bin
                    )
                    self.has_missing_bin_.append(True)
                elif self.feature_dtypes_[i] == 'binary':
                    bins = np.array([0, 1])
                    self.has_missing_bin_.append(False)
                    self.has_missing_values_.append(False)
                else:
                    # Ordinal encoder and force missing/unknown value to be 0
                    ordinal_encoder = OrdinalEncoder(
                        dtype=float, 
                        min_frequency=min(X.shape[0] // 100, 50) if X.shape[0] > 100 else 1,
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
                        self.has_missing_bin_.append(True)
                        self.has_missing_values_.append(True)
                    else:
                        bins = [c for c in ordinal_encoder.categories_[0] if c is not None and c == c and c not in infrequent_categories]
                        self.has_missing_bin_.append(False)
                        self.has_missing_values_.append(False)
                        
                    if len(infrequent_categories) > 0:
                        bins.append("Other")
                        
                    bins = np.array(bins)
                
                self.feature_bins_.append(bins)
                
                # Number of bins is (maximum bin index) + 1 (accounting for missing bin)
                self.feature_sizes_.append(int(X_discrete[:, i].max() + 1))
                # self.feature_sizes_.append(len(bins))
            
        if hasattr(X, 'columns'):
            X_discrete = pd.DataFrame(X_discrete, columns=col_names, dtype=float)
            
        return X_discrete
    
    def _compute_bin_scores(self, ignore_missing_bin_in_intercept=True):
        
        # has_missing_bins = [bins[0] != bins[0] if len(bins) > 0 else True for bins in self.feature_bins_]
        
        for model in self.models:
        
            model.compute_intercept(
                model.bin_counts, 
                ignore_missing_bin=ignore_missing_bin_in_intercept,
                has_missing_bin=self.has_missing_bin_
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
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(random_state)
        
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats_'):
            self.logger.debug("Found selected features. Using only those features.")
            X_train = X_train[self.selected_feats_]
            X_val = X_val[self.selected_feats_]
            
            if self.fit_pairs:
                pairs_list = [
                    [X_train.columns.get_loc(feat1), X_train.columns.get_loc(feat2)] for feat1, feat2 in self.selected_pairs_
                ]
            else:
                pairs_list = None
                
            feature_sizes = [self.feature_sizes_[self.feature_names_in_.get_loc(feat)] for feat in self.selected_feats_]
            cat_feat_mask = self.cat_feat_mask_[self.feature_names_in_.get_indexer(self.selected_feats_)]
            monotone_constraints = None if self.monotone_constraints is None else [self.monotone_constraints[self.feature_names_in_.get_loc(feat)] for feat in self.selected_feats_]

        else:
            pairs_list = self.pairs_list
            feature_sizes = self.feature_sizes_
            cat_feat_mask = self.cat_feat_mask_
            monotone_constraints = self.monotone_constraints
            
        if partialed_feats is not None:
            partialed_indices = [X_train.columns.get_loc(feat) for feat in partialed_feats]
        else:
            partialed_indices = None
        
        # Get data loaders
        self.logger.debug("GETTING DATA LOADERS")
        train_loader = self.get_data_loader(X_train, y_train, **train_loader_args)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False, **val_loader_args)
        
        model = _BaseSingleSplitDNAMiteModel(
            n_features=X_train.shape[1], # don't use n_features in case of selected_feats 
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
            cat_feat_mask=cat_feat_mask,
            monotone_constraints=monotone_constraints,
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
        
        if hasattr(self, 'selected_feats_'):
            model.feature_bins = [self.feature_bins_[self.feature_names_in_.get_loc(feat)] for feat in self.selected_feats_]
            model.bin_counts = [
                get_bin_counts(X_train[col], nb) for col, nb in zip(X_train.columns, feature_sizes)
            ]
        else:
            model.feature_bins = self.feature_bins_
                
            # Compute the bin counts
            model.bin_counts = [
                get_bin_counts(X_train[col], nb) for col, nb in zip(X_train.columns, self.feature_sizes_)
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
        Perform feature selection. Selected features and pairs will be stored in ``self.selected_feats_``
        and ``self.selected_pairs_``, respectively. Should be called before fit if feature selection is desired.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
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
        
        
        if isinstance(X, np.ndarray):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        if hasattr(self, "feature_names_in_"):
            if not all(X.columns == self.feature_names_in_):
                raise ValueError("Input data columns do not match saved feature names.")
        
        self.gamma = gamma
        self.pair_gamma = pair_gamma if pair_gamma is not None else gamma
        self.reg_param = reg_param
        self.pair_reg_param = pair_reg_param
        self.entropy_param = entropy_param
        self.n_features = X.shape[1]
        
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
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
                feature_sizes=self.feature_sizes_, 
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
                cat_feat_mask=self.cat_feat_mask_,
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
            self.selected_feats_ = self.feature_names_in_[model.active_feats.cpu().numpy()].tolist()
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
        
        self.selected_feats_ = self.feature_names_in_[model.active_feats.cpu().numpy()].tolist()
        self.selected_pairs_ = [
            [self.feature_names_in_[pair[0]], self.feature_names_in_[pair[1]]]
            for pair in model.pairs_list[model.active_pairs].cpu().numpy()
        ]
        self.pairs_list = [selected_pair_indices[i] for i in model.active_pairs.cpu().numpy()]
        
        self.logger.info(f"Number of interaction features selected: {len(self.selected_pairs_)}")
        
        return best_val_loss, val_score, model
    
    def get_regularization_path(self, X, y, init_reg_param, partialed_feats=None):
        
        original_fit_pairs = self.fit_pairs
        
        self.reg_param = init_reg_param
        self.fit_pairs = False
        self.gamma = None # will be set in _select_features
        self.pair_gamma = None
        self.pair_reg_param = 0
        self.entropy_param = 0
        self.n_features = X.shape[1]
        
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
            n_feat_selected = len(self.selected_feats_)
            num_feats.append(n_feat_selected)
            losses.append(best_val_loss)
            feats[n_feat_selected] = self.selected_feats_
            scores.append(list(val_score.values())[0])
            
            self.logger.debug(f"Number of features selected: {n_feat_selected}")
            self.logger.debug(f"Selected features: {self.selected_feats_}")
            self.logger.debug(f"Validation loss: {best_val_loss}")
            self.logger.debug(f"Validation score: {val_score}")
            
            # model.reg_param = model.reg_param * 2
            self.reg_param = self.reg_param * 2
            del self.selected_feats_
            
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
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
        
        self.selected_pairs_ = [
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
    
    def fit(self, X, y, pairs_list=None, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input features for training.

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable or labels.
            
        pairs_list : list of tuple[str, str] or None, optional
            List of feature interactions to include; if None, no specific pairs are used.

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        self.logger.debug("STARTING FIT...")
        
        # Check that X and y are the same length
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        if hasattr(self, "feature_names_in_"):
            if not all(X.columns == self.feature_names_in_):
                raise ValueError("Input data columns do not match saved feature names.")
        
        if hasattr(self, 'selected_feats_'):
            if hasattr(self, 'selected_pairs_') and self.fit_pairs:
                self.logger.warning("Found selected features and pairs. Using only those features and pairs.")
            else:
                self.logger.warning("Found selected features. Using only those features.")
                
        if pairs_list is not None and self.num_pairs > 0:
            raise ValueError("Cannot specify pairs_list and num_pairs at the same time.")
        
        self.feature_names_in_ = X.columns
        self.n_features = X.shape[1]
        self.pairs_list = pairs_list
        if pairs_list is not None:
            self.fit_pairs = True
            self.pairs_list = [
                (X.columns.get_loc(feat1), X.columns.get_loc(feat2)) for feat1, feat2 in pairs_list
            ]
        
        # First discretize the data
        X_discrete = self._discretize_data(X)
        
        self.models = []
        
        # Fit several models, one for each validation split
        for i in range(self.n_val_splits):
            self.logger.info(f"SPlIT: {i}")

            # Split the data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X_discrete, y, test_size=self.validation_size, random_state=self.random_state+i)
            
            # Fit to this split
            model = self._fit_one_split(X_train, y_train, X_val, y_val, random_state=self.random_state+i, partialed_feats=partialed_feats)
            
            self.models.append(model)
            
        self._compute_bin_scores()
            
        return self
    
    def _predict(self, X_test_discrete, models):
        self.logger.debug("STARTING _PREDICT...")
        
        # # If selected_feats is set, only use those features
        # if hasattr(self, 'selected_feats'):
        #     X_test_discrete = X_test_discrete[self.selected_feats_]
        
        if hasattr(self, 'selected_feats_'):
            if self.fit_pairs:
                pairs_list = [
                    [X_test_discrete.columns.get_loc(feat1), X_test_discrete.columns.get_loc(feat2)] for feat1, feat2 in self.selected_pairs_
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
        X_test : pandas.DataFrame, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        if isinstance(X_test, np.ndarray):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        if hasattr(self, "feature_names_in_"):
            if not all(X_test.columns == self.feature_names_in_):
                raise ValueError("Input data columns do not match saved feature names.")
            
        if not hasattr(self, 'models'):
            raise ValueError("Model has not been fitted yet. Please fit the model before predicting.")
        
        X_test_discrete = self._discretize_data(X_test)
                
        if hasattr(self, 'selected_feats_'):
            self.logger.debug("Found selected features. Using only those features.")
            X_test_discrete = X_test_discrete[self.selected_feats_]
        
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
                feat_idx = self.feature_names_in_.get_loc(col)
                missing_bin_count = bin_counts[0] if self.has_missing_bin_[feat_idx] else 0
                
                if missing_bin != "include" and self.has_missing_bin_[feat_idx]:
                    missing_bin_score = feat_bin_scores[0]
                    feat_bin_scores = feat_bin_scores[1:]
                    bin_counts = bin_counts[1:]
                if missing_bin == "ignore" and not self.has_missing_values_[feat_idx]:
                    feat_bin_scores = feat_bin_scores[1:]
                    bin_counts = bin_counts[1:]
                
                feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(bin_counts)) / np.sum(bin_counts)
                
                if missing_bin == "stratify":
                    importances.append([split, col, feat_importance, "observed", len(feat_bin_scores)])
                    if self.has_missing_values_[feat_idx]:
                        if missing_bin_count == 0:
                            raise ValueError(f"Tried to compute a missing bin feature score for a feature without a missing bin.")
                        missing_importance = np.abs(missing_bin_score)
                        importances.append([split, col, missing_importance, "missing", 1])
                else:
                    importances.append([split, col, feat_importance, len(feat_bin_scores)])
            
            if model.fit_pairs:
                
                if missing_bin == "include":
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
        
        if missing_bin != "include" and self.fit_pairs:
            self.logger.warning(f"missing_bin was set to {missing_bin}, which does not support interactions. Thus, interactions will not be scored.")
        
        plt.figure(figsize=(8 if missing_bin == "stratify" else 6, 4))
        
        feat_imps = self.get_feature_importances(missing_bin=missing_bin)

        if missing_bin == "stratify":
            
            if self.n_val_splits == 1:
                feat_imps = feat_imps.groupby(["feature", "bin_type"])["importance"] \
                        .agg(["mean"]) \
                        .reset_index() \
                        .pivot(index="feature", columns="bin_type") \
                        .sort_values(("mean", "observed"), ascending=True) \
                        .tail(n_features)
                feat_imps = feat_imps.fillna(0)
                plt.barh(
                    y=feat_imps.index,
                    width=feat_imps[("mean", "observed")],
                    color=sns.color_palette()[0]
                )
                if ("mean", "missing") in feat_imps.columns:
                    plt.barh(
                        y=feat_imps.index,
                        width=-1 * feat_imps[("mean", "missing")],
                        color=sns.color_palette()[1]
                    )   
            else:
                feat_imps = feat_imps.groupby(["feature", "bin_type"])["importance"] \
                        .agg(["mean", "sem"]) \
                        .reset_index() \
                        .pivot(index="feature", columns="bin_type") \
                        .sort_values(("mean", "observed"), ascending=True) \
                        .tail(n_features)
                feat_imps = feat_imps.fillna(0)
                plt.barh(
                    y=feat_imps.index,
                    width=feat_imps[("mean", "observed")],
                    xerr=1.96 * feat_imps[("sem", "observed")],
                    color=sns.color_palette()[0]
                )
                if ("mean", "missing") in feat_imps.columns:
                    plt.barh(
                        y=feat_imps.index,
                        width=-1 * feat_imps[("mean", "missing")],
                        xerr=1.96 * feat_imps[("sem", "missing")],
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
        
        is_cat_col = self.feature_dtypes_[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
        has_missing_values = self.has_missing_values_[self.feature_names_in_.get_loc(feature_name)]
        
        dfs = []
        for i, model in enumerate(self.models):
        
            col_index = model.feature_names_in_.get_loc(feature_name)
            feat_bin_scores = model.bin_scores[col_index].squeeze()
            
            
            if not is_cat_col:
                
                if has_missing_values:
                    feat_bin_values = np.concatenate([
                        [np.nan],
                        [model.feature_bins[col_index].min() - 0.01],
                        model.feature_bins[col_index]
                    ])
                    bin_counts = model.bin_counts[col_index]
                else:
                    feat_bin_scores = feat_bin_scores[1:]
                    feat_bin_values = np.concatenate([
                        [model.feature_bins[col_index].min() - 0.01],
                        model.feature_bins[col_index]
                    ])
                    bin_counts = model.bin_counts[col_index][1:]
                
            else:
                
                feat_bin_values = model.feature_bins[col_index]
                bin_counts = model.bin_counts[col_index]
                
            dfs.append(
                pd.DataFrame({
                    "feature": feature_name,
                    "bin": feat_bin_values,
                    "score": feat_bin_scores,
                    "count": bin_counts,
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
        
        # def set_font_sizes(axes):
        #     LABEL_FONT_SIZE = 14 * (1.5 if plot_missing_bin else 1)
        #     TICK_FONT_SIZE = 12 * (1.5 if plot_missing_bin else 1)
        #     for ax in axes:
        #         ax.xaxis.label.set_fontsize(LABEL_FONT_SIZE)
        #         ax.yaxis.label.set_fontsize(LABEL_FONT_SIZE)
        #         ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            
        is_cat_cols = [self.feature_dtypes_[self.feature_names_in_.get_loc(f)] != 'continuous' for f in feature_names]
        has_missing_values = [self.has_missing_values_[self.feature_names_in_.get_loc(f)] for f in feature_names]

        num_main_plots = len(feature_names)
        num_missing_bin_plots = sum([1 if not c and m and plot_missing_bin else 0 for c, m in zip(is_cat_cols, has_missing_values)])
        num_axes = num_main_plots + num_missing_bin_plots

        if axes is None:
            if plot_missing_bin:
            
                fig = plt.figure(figsize=(4*num_main_plots + num_missing_bin_plots, 4))
                
                width_ratios = []
                for i in range(num_main_plots):
                    if not is_cat_cols[i] and has_missing_values[i]:
                        width_ratios.extend([10, 1])
                    else:
                        width_ratios.append(10)
                gs = gridspec.GridSpec(1, num_axes, width_ratios=width_ratios)
                axes = [fig.add_subplot(gs[i]) for i in range(num_axes)]
                
            else:
                fig, axes = plt.subplots(1, num_axes, figsize=(4*num_axes, 4))
                if num_axes == 1:
                    axes = [axes]
                    
            # set_font_sizes(axes)
        ax_idx = 0

        for feature_name in feature_names:
            shape_data = self.get_shape_function(feature_name)
            is_cat_col = self.feature_dtypes_[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
            has_missing_values = self.has_missing_values_[self.feature_names_in_.get_loc(feature_name)]
            
            if not is_cat_col:
                if not plot_missing_bin or not has_missing_values:
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
                    # axes[ax_idx].set_ylim(axes[0].get_ylim())
                    # axes[ax_idx].set_yticklabels([])
                    axes[ax_idx].set_xlim(-0.5, 0.5)
                    
                    ax_idx += 1
                    
            else:
                shape_data = shape_data.fillna("Missing")
                sns.barplot(x="bin", y="score", data=shape_data, errorbar=('ci', 95), ax=axes[ax_idx], color=sns.color_palette()[0])
                axes[ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                axes[ax_idx].tick_params(axis='x', rotation=45)
                for label in axes[ax_idx].get_xticklabels():
                    label.set_ha('right')
                if ax_idx == 0:
                    axes[ax_idx].set_ylabel("\n".join(textwrap.wrap(yaxis_label, width=25)))
                else:
                    axes[ax_idx].set_ylabel("")
                ax_idx += 1
        
        # if not plot_missing_bin:
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
                np.arange(0, self.feature_sizes_[col1_index]), 
                np.arange(0, self.feature_sizes_[col2_index])
            )
            if self.has_missing_bin_[col1_index] and not self.has_missing_bin_[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 0] > 0]
            elif not self.has_missing_bin_[col1_index] and self.has_missing_bin_[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 1] > 0]
            else:
                pair_bin_scores = pair_bin_scores[grid.prod(axis=1) > 0]
            
            # Get bin values
            if self.feature_dtypes_[col1_index] != 'continuous':
                if self.has_missing_bin_[col1_index]:
                    feat1_bin_values = model.feature_bins[col1_index][1:]
                else:
                    feat1_bin_values = model.feature_bins[col1_index]
            else:
                feat1_bin_values = np.concatenate([
                    [model.feature_bins[col1_index].min() - 0.01],
                    model.feature_bins[col1_index]
                ])
                
            if self.feature_dtypes_[col2_index] != 'continuous':
                if self.has_missing_bin_[col2_index]:
                    feat2_bin_values = model.feature_bins[col2_index][1:]
                else:
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
        if self.feature_dtypes_[col1_index] == 'continuous':
            pair_data_dnamite.index = pair_data_dnamite.index.to_series().astype(float).round(3)
        if self.feature_dtypes_[col2_index] == 'continuous':
            pair_data_dnamite.columns = pair_data_dnamite.columns.to_series().astype(float).round(3)
        
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
        
        if not self.fit_pairs:
            raise ValueError("Model was not fit with pairs.")
        
        plt.figure(figsize=(4, 4))
        
        pair_data_dnamite = self.get_pair_shape_function(feat1_name, feat2_name)

        sns.heatmap(pair_data_dnamite)
        
        plt.tight_layout()