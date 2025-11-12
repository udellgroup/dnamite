import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin
from .base_model import BaseDNAMiteModel

class DNAMiteBinaryClassifier(ClassifierMixin, BaseDNAMiteModel):
    """
    DNAMiteClassifier is a model for binary classification using the DNAMite architecture.

    Parameters
    ----------
    n_embed : int, optional (default=32)
        The size of the embedding layer.
    n_hidden : int, optional (default=32)
        The number of hidden units in the hidden layers.
    n_layers : int, default=2
        Number of hidden layers in the model.
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
            n_embed=n_embed,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_output=1,
            max_bins=max_bins,
            min_samples_per_bin=min_samples_per_bin,
            validation_size=validation_size,
            n_val_splits=n_val_splits,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
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
    
    def fit(self, X, y, pairs_list=None, partialed_feats=None):
        """
        Train model.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input features for training. 
            Missing values should be encoded as np.nan.
            Categorical features will automatically be detected as all columns with dtype "object" or "category".

        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The labels. Should have two unique values.
            
        pairs_list : list of tuple[str, str] or None, optional
            List of feature interactions to include; if None, no specific pairs are used.

        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        """
        
        if len(np.unique(y)) != 2:
            raise ValueError("y should have exactly two unique values")
        
        if not hasattr(self, "label_encoder"):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = self.label_encoder.transform(y)
        
        return super().fit(X, y, pairs_list=pairs_list, partialed_feats=partialed_feats)
    
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
        X_test : pandas.DataFrame, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        
        return super().predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities using the trained model.
        
        Parameters
        ----------
        X_test : pandas.DataFrame, shape (n_samples, n_features)
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
        
        if len(np.unique(y)) != 2:
            raise ValueError("y should have exactly two unique values")
        
        if not hasattr(self, "label_encoder"):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = self.label_encoder.transform(y)
        
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
        
        y = self.label_encoder.transform(y)
        
        preds = self._predict(X, models=[model])
        preds = 1 / (1 + np.exp(-preds))
        
        return {"AUC": roc_auc_score(y, preds)}
    
    def get_regularization_path(self, X, y, init_reg_param, partialed_feats=None):
        """
        Get the regularization path for the model.
        
        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input features for the model.
        y : pandas.Series or numpy.ndarray, shape (n_samples,)
            The target variable.
        init_reg_param : float
            Initial regularization parameter.
        partialed_feats : list or None, optional
            A list of features that should be fit completely before fitting all other features.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the regularization path.
        """
        
        if not hasattr(self, "label_encoder"):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = self.label_encoder.transform(y)
        
        return super().get_regularization_path(X, y, init_reg_param, partialed_feats=partialed_feats)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            The AUC score.
        """
        return roc_auc_score(y, self.predict_proba(X))