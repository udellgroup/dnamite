import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from dnamite.loss_fns import ipcw_rps_loss
import textwrap
from .base_model import BaseDNAMiteModel

class DNAMiteSurvival(BaseDNAMiteModel):
    """
    DNAMiteSurvival is a model for survival analysis using the DNAMite architecture.

    Parameters
    ----------
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
    censor_estimator : str, optional (default="km")
        The estimator to use for estimating the censoring distribution.
        "km" for Kaplan-Meier, "cox" for Cox proportional hazards.
    """

    
    def __init__(
        self, 
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
            kernel_size=kernel_size,
            kernel_weight=kernel_weight,
            pair_kernel_size=pair_kernel_size,
            pair_kernel_weight=pair_kernel_weight,
            monotone_constraints=monotone_constraints,
            num_pairs=num_pairs, 
            verbosity=verbosity,
            random_state=random_state
        )
        
        self.n_eval_times = n_eval_times
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
        
        if isinstance(X, np.ndarray):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        if hasattr(self, "feature_names_in_"):
            if not all(X.columns == self.feature_names_in_):
                raise ValueError("Input data columns do not match saved feature names.")
        
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
        
        
        super().fit(X, y, pairs_list=pairs_list, partialed_feats=partialed_feats)
        
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
                feat_idx = self.feature_names_in_.get_loc(col)
                missing_bin_count = bin_counts[0] if self.has_missing_bin_[feat_idx] else 0
                
                if missing_bin != "include" and self.has_missing_bin_[feat_idx]:
                    missing_bin_score = feat_bin_scores[0]
                    feat_bin_scores = feat_bin_scores[1:]
                    bin_counts = bin_counts[1:]
                if missing_bin == "ignore" and not self.has_missing_values_[feat_idx]:
                    feat_bin_scores = feat_bin_scores[1:]
                    bin_counts = bin_counts[1:]
                
                if eval_time is not None:
                    feat_importance = np.sum(np.abs(feat_bin_scores[:, eval_index]) * np.array(bin_counts)) / np.sum(bin_counts)
                else:
                    feat_importance = np.sum(np.abs(feat_bin_scores) * np.array(bin_counts).reshape(-1, 1)) / np.sum(bin_counts)

                    
                if missing_bin == "stratify":
                    importances.append([split, col, feat_importance, "observed", len(feat_bin_scores)])
                    if self.has_missing_values_[feat_idx]:
                        if missing_bin_count == 0:
                            raise ValueError(f"Tried to compute a missing bin feature score for a feature without a missing bin.")
                        
                        if eval_time is not None:
                            missing_importance = np.abs(missing_bin_score[eval_index])
                        else:
                            missing_importance = np.sum(np.abs(missing_bin_score))
                            
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
        is_cat_col = self.feature_dtypes_[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
        has_missing_values = self.has_missing_values_[self.feature_names_in_.get_loc(feature_name)]
        
        eval_index = np.searchsorted(self.eval_times.cpu().numpy(), eval_time)
        
        dfs = []
        for i, model in enumerate(self.models):
        
            col_index = model.feature_names_in_.get_loc(feature_name)       
            
            feat_bin_scores = model.bin_scores[col_index][:, eval_index]
            
            
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
            
        is_cat_cols = [self.feature_dtypes_[self.feature_names_in_.get_loc(f)] != 'continuous' for f in feature_names]
        has_missing_values = [self.has_missing_values_[self.feature_names_in_.get_loc(f)] for f in feature_names]
        
        num_main_plots = len(feature_names)
        num_missing_bin_plots = sum([1 if not c and m and plot_missing_bin else 0 for c, m in zip(is_cat_cols, has_missing_values)])
        num_axes = num_main_plots + num_missing_bin_plots

        if plot_missing_bin:
        
            fig = plt.figure(figsize=(4*num_main_plots + num_missing_bin_plots, 4*len(eval_times)))
            
            width_ratios = []
            for i in range(num_main_plots):
                if not is_cat_cols[i] and has_missing_values[i]:
                    width_ratios.extend([10, 1])
                else:
                    width_ratios.append(10)
                   
            gs = gridspec.GridSpec(len(eval_times), num_axes, width_ratios=width_ratios)
            axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(num_axes)] for i in range(len(eval_times))])
            
        else:
            fig, axes = plt.subplots(len(eval_times), num_axes, figsize=(4*num_axes, 4*len(eval_times)))
            if not isinstance(axes, np.ndarray):
                axes = np.array(axes)
            if len(feature_names) == 1:
                axes = axes.reshape(-1, 1)
            if len(eval_times) == 1:
                axes = axes.reshape(1, -1)
        
        # num_axes = sum([2 if not c and plot_missing_bin else 1 for _, c in zip(feature_names, is_cat_cols)])
        # num_axes = num_axes * len(eval_times)
        
        # if plot_missing_bin:
        
            # fig = plt.figure(figsize=(4*num_axes, 4))
            # fig = plt.figure(figsize=(4*num_axes, 4*len(eval_times)))
            
            
            # gs = gridspec.GridSpec(1, 2*len(feature_names) + len(feature_names)-1, width_ratios=[10, 1, 2] * (len(feature_names)-1) + [10, 1])  # Add an extra column for the space
            # def generate_indices(n_times):
            #     indices = []
            #     for i in range(n_times):
            #         indices.extend([i*3, i*3+1])
            #     return indices

            # axes = [fig.add_subplot(gs[i]) for i in generate_indices(len(feature_names))]
            
            # fig = plt.figure(figsize=(4 * len(feature_names), 4 * len(eval_times)))

            # # Create GridSpec with rows equal to len(eval_times) and appropriate columns
            # gs = gridspec.GridSpec(len(eval_times), 2 * len(feature_names) + len(feature_names) - 1, width_ratios=[10, 1, 2] * (len(feature_names) - 1) + [10, 1])

            # def generate_indices(n_times, row):
            #     """Generate indices for a specific row in the grid."""
            #     indices = []
            #     for i in range(n_times):
            #         indices.extend([row * (2 * n_times + n_times - 1) + i * 3, row * (2 * n_times + n_times - 1) + i * 3 + 1])
            #     return indices

            # # Create axes for all rows and columns
            # axes = [fig.add_subplot(gs[i]) for row in range(len(eval_times)) for i in generate_indices(len(feature_names), row)]
            
        # else:
        #     fig, axes = plt.subplots(len(eval_times), num_axes, figsize=(4*num_axes, 4*len(eval_times)))
        #     # if len(axes) == 1:
        #     #     axes = [axes]
        #     if not isinstance(axes, np.ndarray):
        #         axes = np.array(axes)
        #     if len(feature_names) == 1:
        #         axes = axes.reshape(-1, 1)
        #     if len(eval_times) == 1:
        #         axes = axes.reshape(1, -1)
        
        ax_idx = 0
        
        for eval_idx, eval_time in enumerate(eval_times):
            ax_idx = 0
            yaxis_label = f"Contribution to logit($P(T \\leq {eval_time} | X)$)"
            
            for feature_name in feature_names:
                
                is_cat_col = self.feature_dtypes_[self.feature_names_in_.get_loc(feature_name)] != 'continuous'
                has_missing_values = self.has_missing_values_[self.feature_names_in_.get_loc(feature_name)]
                
                shape_data = self.get_shape_function(feature_name, eval_time)
                
                if not is_cat_col:
                    if not plot_missing_bin or not has_missing_values:
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
                        # axes[eval_idx, ax_idx].set_ylim(axes[eval_idx, 0].get_ylim())
                        # axes[eval_idx, ax_idx].set_yticklabels([])
                        axes[eval_idx, ax_idx].set_xlim(-0.5, 0.5)
                        
                        ax_idx += 1
                        
                else:
                    
                    shape_data = shape_data.fillna("Missing")
                    sns.barplot(x="bin", y="score", data=shape_data, errorbar=('ci', 95), ax=axes[eval_idx, ax_idx], color=sns.color_palette()[0])
                    axes[eval_idx, ax_idx].set_xlabel("\n".join(textwrap.wrap(feature_name, width=25)))
                    axes[eval_idx, ax_idx].tick_params(axis='x', rotation=45)
                    for label in axes[eval_idx, ax_idx].get_xticklabels():
                        label.set_ha('right')
                    if ax_idx == 0:
                        axes[eval_idx, ax_idx].set_ylabel("\n".join(textwrap.wrap(yaxis_label, width=25)))
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
            
            if self.has_missing_bin_[col1_index] and not self.has_missing_bin_[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 0] > 0]
            elif not self.has_missing_bin_[col1_index] and self.has_missing_bin_[col2_index]:
                pair_bin_scores = pair_bin_scores[grid[:, 1] > 0]
            else:
                pair_bin_scores = pair_bin_scores[grid.prod(axis=1) > 0]
            
            # Get bin values
            if self.feature_dtypes_[col1_index] != 'continuous':
                # feat1_bin_values = np.arange(-1, len(model.feature_bins[col1_index])+1)
                feat1_bin_values = model.feature_bins[col1_index]
            else:
                feat1_bin_values = np.concatenate([
                    [model.feature_bins[col1_index].min() - 0.01],
                    model.feature_bins[col1_index]
                ])
                
            if self.feature_dtypes_[col2_index] != 'continuous':
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
            pair_data_dnamite.index = pair_data_dnamite.index.to_series().round(3)
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
        
        if not self.fit_pairs:
            raise ValueError("Model was not fit with pairs.")
        
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
        
        if hasattr(self, 'selected_feats_'):
            if self.fit_pairs:
                pairs_list = [
                    [X_test_discrete.columns.get_loc(feat1), X_test_discrete.columns.get_loc(feat2)] for feat1, feat2 in self.selected_pairs_
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
        X_test : pandas.DataFrame, shape (n_samples, n_features)
            The input features for prediction.
        
        """
        
        if isinstance(X_test, np.ndarray):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        if hasattr(self, "feature_names_in_"):
            if not all(X_test.columns == self.feature_names_in_):
                raise ValueError("Input data columns do not match saved feature names.")
        
        pcw_obs_times_test = self._predict_censoring_distribution(np.zeros(X_test.shape[0]), X_test) + 1e-5
        X_test_discrete = self._discretize_data(X_test)
        
        if hasattr(self, 'selected_feats_'):
            self.logger.debug("Found selected features. Using only those features.")
            X_test_discrete = X_test_discrete[self.selected_feats_]
        
        return self._predict(X_test_discrete, self.models, pcw_obs_times_test)
    
    def predict_survival(self, X_test, test_times=None):
        """
        Predict the survival probability for a set of evaluation times.
        
        Parameters
        ----------
        X_test : pandas.DataFrame, shape (n_samples, n_features)
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
        
        return super().get_regularization_path(X, y, init_reg_param, partialed_feats=partialed_feats)