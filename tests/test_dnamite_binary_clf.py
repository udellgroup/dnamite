import numpy as np 
import pandas as pd 
from itertools import combinations
from sklearn.model_selection import train_test_split
import torch
from dnamite.models import DNAMiteBinaryClassifier
from sklearn.datasets import load_breast_cancer

def get_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data["data"], data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test

def test_init():
    X_train, X_test, y_train, y_test = get_data()
    
    # Minimal initialization
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
    )
    assert model.kernel_size > 0
    assert model.kernel_weight > 0
    assert model.pair_kernel_weight > 0
    assert model.pair_kernel_size > 0
    assert model.verbosity == 0
    
def test_train():
    X_train, X_test, y_train, y_test = get_data()
    
    # First fit without pairs
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        max_epochs=2,
        n_val_splits=2,
        batch_size=32
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds is not None
    
    # Then fit with pairs
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        max_epochs=2,
        n_val_splits=2,
        batch_size=32,
        num_pairs=3
    )
    model.fit(X_train, y_train)
    for m in model.models:
        assert m.n_pairs == 3
        assert m.pairs_list is not None
        assert m.selected_pair_indices is not None
        
    preds = model.predict(X_test)
    assert preds is not None


def test_train_with_pairs_list():
    
    X_train, X_test, y_train, y_test = get_data()
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        pairs_list=[(0, 1), (1, 2), (2, 3)],
        max_epochs=2,
        n_val_splits=2,
        batch_size=32
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
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=10,
        batch_size=32
    )
    model.select_features(X_train, y_train, reg_param=0.08, gamma=0.05)
    assert len(model.selected_feats) < model.n_features
    assert model.pairs_list is None
    model.fit(X_train, y_train)
    assert len(model.selected_feats) < model.n_features
    assert model.pairs_list is None
    preds = model.predict(X_test)
    assert preds is not None
    
    # Now try with also selecting pairs
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        batch_size=32,
        max_epochs=10
    )
    model.select_features(
        X_train, y_train, select_pairs=True, 
        reg_param=0.08,
        gamma=0.05,
        pair_reg_param=0.005,
        pair_gamma=0.1,
    )
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
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        batch_size=32
    )
    model.fit(X_train, y_train)
    feat_imps = model.get_feature_importances()
    assert feat_imps["importance"].min() > 0
    model.plot_feature_importances(n_features=5, missing_bin="include")
    model.plot_feature_importances(n_features=5, missing_bin="ignore")
    model.plot_feature_importances(n_features=5, missing_bin="stratify")
    
def test_prob_preds():
    X_train, X_test, y_train, y_test = get_data()
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
    )
    model.fit(X_train, y_train)
    prob_preds = model.predict_proba(X_test)
    assert len(prob_preds) == X_test.shape[0]
    assert np.all((0 <= prob_preds) & (prob_preds <= 1))
    
def test_kernel():
    X_train, X_test, y_train, y_test = get_data()
    
    # Test with no kernel size or weight
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        kernel_size=0,
        kernel_weight=0
    )
    model.fit(X_train, y_train)
    _ = model.predict(X_test)

def test_partial():
    X_train, X_test, y_train, y_test = get_data()
    
    # Test with normal
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
    )
    model.fit(X_train, y_train, partialed_feats=[X_train.columns[0]])
    _ = model.predict(X_test)
    
    # Test with pairs
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        num_pairs=3
    )
    model.fit(X_train, y_train, partialed_feats=[X_train.columns[0]])
    _ = model.predict(X_test)
    
    # Test with feature selection
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=10,
        batch_size=32
    )
    model.select_features(X_train, y_train, reg_param=0.08, gamma=0.05, partialed_feats=[X_train.columns[0]])
    model.fit(X_train, y_train, partialed_feats=[X_train.columns[0]])
    _ = model.predict(X_test)
    
def test_monotone():
    X_train, X_test, y_train, y_test = get_data()
    
    # Test with normal
    model = DNAMiteBinaryClassifier(
        n_features=X_train.shape[1],
        n_val_splits=2,
        max_epochs=2,
        monotone_constraints=[1] + [0] * (X_train.shape[1] - 1)
    )
    model.fit(X_train, y_train)
    
    shape_data = model.get_shape_function(X_train.columns[0])
    for split in shape_data["split"].unique():
        split_shape_data = shape_data[(shape_data["split"] == split) & (shape_data["bin"].notna())]
        split_shape_data = split_shape_data.sort_values("bin")
        assert split_shape_data["score"].diff().dropna().ge(0).all()
        

    
    