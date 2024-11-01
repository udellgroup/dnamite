import numpy as np 
import pandas as pd 
from itertools import combinations
from sklearn.model_selection import train_test_split
import torch
from dnamite.models import DNAMiteRegressor
from sklearn.datasets import fetch_california_housing

def get_data():
    data = fetch_california_housing(as_frame=True)
    X, y = data["data"], data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test

def test_init():
    X_train, X_test, y_train, y_test = get_data()
    
    # Minimal initialization
    model = DNAMiteRegressor(
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
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        max_epochs=2,
        n_val_splits=2,
        fit_pairs=False,
        batch_size=128
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds is not None
    
    # Then fit with pairs
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        max_epochs=2,
        n_val_splits=2,
        batch_size=128
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
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        pairs_list=[(0, 1), (1, 2), (2, 3)],
        max_epochs=2,
        n_val_splits=2,
        batch_size=128
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
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        n_val_splits=2,
        reg_param=0.08,
        gamma=0.1,
        max_epochs=10,
        fit_pairs=False,
        batch_size=128
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
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        n_val_splits=2,
        reg_param=0.08,
        gamma=0.1,
        max_epochs=10,
        pair_reg_param=0.005,
        pair_gamma=0.1,
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
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        fit_pairs=False,
        batch_size=128
    )
    model.fit(X_train, y_train)
    feat_imps = model.get_feature_importances()
    assert feat_imps["importance"].min() > 0
    model.plot_feature_importances(k=5)
    
    # Now try with also selecting pairs
    model = DNAMiteRegressor(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        batch_size=128
    )
    model.fit(X_train, y_train)
    feat_imps = model.get_feature_importances()
    assert feat_imps["importance"].min() > 0
    model.plot_feature_importances(k=5)
    

    
    