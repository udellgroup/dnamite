import numpy as np 
import pandas as pd 
from itertools import combinations
from sklearn.model_selection import train_test_split
import torch
from dnamite.models import DNAMiteSurvival
from sksurv.datasets import load_flchain

def get_data():
    X, y = load_flchain()

    # Change y dtypes and names
    y.dtype = np.dtype([('event', bool), ('time', float)])

    # # Impute chapter with new category
    X["chapter"] = X["chapter"].astype("category")
    X["chapter"] = X["chapter"].cat.add_categories("Unknown")
    X["chapter"] = X["chapter"].fillna("Unknown")

    # Clip Training data to avoid overflow errors
    X["creatinine"] = X["creatinine"].clip(-5, 5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test

def test_init():
    X_train, X_test, y_train, y_test = get_data()
    
    # Minimal initialization
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
    )
    assert model.reg_param == 0
    assert model.pair_reg_param == 0
    assert model.pair_gamma == model.gamma
    assert model.kernel_size > 0
    assert model.kernel_weight > 0
    assert model.pair_kernel_weight > 0
    assert model.pair_kernel_size > 0
    assert model.pairs_list == list(combinations(range(model.n_features), 2))
    
def test_train():
    X_train, X_test, y_train, y_test = get_data()
    
    # First fit without pairs
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        max_epochs=2,
        n_val_splits=2,
        fit_pairs=False
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds is not None
    
    # Then fit with pairs
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        max_epochs=2,
        n_val_splits=2
    )
    model.fit(X_train, y_train)
    for m in model.models:
        assert m.selected_pair_indices == list(combinations(range(model.n_features), 2))
        assert torch.equal(m.pairs_list, torch.tensor(list(combinations(range(model.n_features), 2))))
        assert m.n_pairs == model.n_features * (model.n_features - 1) // 2
        
    preds = model.predict(X_test)
    assert preds is not None


def test_train_with_pairs_list():
    
    X_train, X_test, y_train, y_test = get_data()
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        pairs_list=[(0, 1), (1, 2), (2, 3)],
        max_epochs=2,
        n_val_splits=2
    )
    assert model.pairs_list == [(0, 1), (1, 2), (2, 3)]
    
    model.fit(X_train, y_train)
    assert model.pairs_list == [(0, 1), (1, 2), (2, 3)]
    for m in model.models:
        assert m.selected_pair_indices == [(0, 1), (1, 2), (2, 3)]
        assert torch.equal(m.pairs_list, torch.tensor([(0, 1), (1, 2), (2, 3)]))
        assert m.n_pairs == 3
        
    preds = model.predict(X_test)
    assert preds is not None
    
def test_select_features():
    X_train, X_test, y_train, y_test = get_data()
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        reg_param=0.2,
        gamma=0.04,
        max_epochs=6,
        fit_pairs=False
    )
    model.select_features(X_train, y_train)
    assert len(model.selected_feats) < model.n_features
    assert model.pairs_list is None
    model.fit(X_train, y_train)
    assert len(model.selected_feats) < model.n_features
    assert model.pairs_list is None
    preds = model.predict(X_test)
    assert preds is not None
    
    # Now try with also selecting pairs
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        reg_param=0.2,
        gamma=0.04,
        max_epochs=6,
        pair_reg_param=0.005,
        pair_gamma=0.02,
    )
    model.select_features(X_train, y_train)
    assert len(model.selected_feats) < model.n_features
    assert model.pairs_list is not None
    assert len(model.pairs_list) < model.n_features * (model.n_features - 1) // 2
    assert model.selected_pairs == [[model.feature_names_in_[pair[0]], model.feature_names_in_[pair[1]]] for pair in model.pairs_list]
    model.fit(X_train, y_train)
    assert len(model.selected_feats) < model.n_features
    assert model.pairs_list is not None
    assert len(model.pairs_list) < model.n_features * (model.n_features - 1) // 2
    assert model.selected_pairs == [[model.feature_names_in_[pair[0]], model.feature_names_in_[pair[1]]] for pair in model.pairs_list]
    preds = model.predict(X_test)
    assert preds is not None
    
def test_feature_importances():
    X_train, _, y_train, _ = get_data()
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        fit_pairs=False
    )
    model.fit(X_train, y_train)
    feat_imps = model.get_feature_importances(eval_time=365)
    assert feat_imps["importance"].min() > 0
    model.plot_feature_importances(k=5, eval_time=365)
    
    # Now try with also selecting pairs
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
    )
    model.fit(X_train, y_train)
    feat_imps = model.get_feature_importances(eval_time=365)
    assert feat_imps["importance"].min() > 0
    model.plot_feature_importances(k=5, eval_time=365)
    
def test_shape_functions():
    X_train, _, y_train, _ = get_data()
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        fit_pairs=False
    )
    model.fit(X_train, y_train)
    _ = model.get_shape_function("chapter", eval_time=365)
    _ = model.get_shape_function("creatinine", eval_time=365)
    model.plot_shape_function("chapter", eval_time=365)
    model.plot_shape_function(["chapter", "creatinine"], eval_time=365)
    
    # Now try with also selecting pairs
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
    )
    model.fit(X_train, y_train)
    
    example_pair = model.pairs_list[0]
    feat1, feat2 = model.feature_names_in_[example_pair[0]], model.feature_names_in_[example_pair[1]]
    _ = model.get_pair_shape_function(feat1, feat2, eval_time=365)
    model.plot_pair_shape_function(feat1, feat2, eval_time=365)
    
def test_survival_preds():
    X_train, X_test, y_train, y_test = get_data()
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        fit_pairs=False
    )
    model.fit(X_train, y_train)
    surv_preds = model.predict_survival(X_test)
    assert surv_preds.shape[0] == X_test.shape[0]
    assert surv_preds.shape[1] == len(model.eval_times)
    assert np.all((0 <= surv_preds) & (surv_preds <= 1))
    
    test_times = np.array([365, 730, 1095])
    surv_preds = model.predict_survival(X_test, test_times)
    assert surv_preds.shape[0] == X_test.shape[0]
    assert surv_preds.shape[1] == len(test_times)
    assert np.all((0 <= surv_preds) & (surv_preds <= 1))
    
def test_calibration():
    X_train, X_test, y_train, y_test = get_data()
    model = DNAMiteSurvival(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        fit_pairs=False
    )
    model.fit(X_train, y_train)
    predicted, observed, quantiles = model.get_calibration_data(X_test, y_test, eval_time=365, n_bins=10)
    assert len(predicted) == len(observed) == 10
    
    model.make_calibration_plot(X_test, y_test, eval_time=365, n_bins=10)
    
    