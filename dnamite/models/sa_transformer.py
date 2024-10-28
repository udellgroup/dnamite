"""
SAMPLE BEGINNING OF FILE DOCSTRING
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable
from torch.utils.data import Dataset
from operator import itemgetter

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sksurv.nonparametric import kaplan_meier_estimator
from dnamite.loss_fns import bce_surv_loss

class TranDataset(Dataset):
    def __init__(self, features, labels, max_time, is_train=True):
        self.is_train = is_train
        self.data = []

        temp = []
        for feature, label in zip(features, labels):
            feature = torch.from_numpy(feature).float()
            duration, is_observed = label[0], label[1]
            temp.append([duration, is_observed, feature])
        sorted_temp = sorted(temp, key=itemgetter(0))

        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, is_observed, feature in new_temp:
            if is_observed:
                mask = max_time * [1.]
                label = duration * [1.] + (max_time - duration) * [0.]
                feature = torch.stack(max_time * [feature])
                self.data.append([feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])
            else:
                # NOTE plus 1 to include day 0
                mask = (duration + 1) * [1.] + (max_time - (duration + 1)) * [0.]
                label = max_time * [1.]
                feature = torch.stack(max_time * [feature])
                self.data.append([feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a+1, len(self.data))
                
            batch = [ [self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a])) ]
        else:
            batch = self.data[index_a]
            
        print("BATCH SIZE", batch[0].shape)
        return batch

    def __len__(self):
        return len(self.data)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


# Final layer for the Transformer
class TranFinalLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        return torch.sigmoid(x.squeeze(-1))


class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout, num_features):
        super().__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model)
        
    def forward(self, x, mask=None):
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# Transformer model for survival analysis
# Based on this paper: https://proceedings.mlr.press/v146/hu21a/hu21a.pdf
class SATransformerSingleSplit(nn.Module):
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
    
    def train_(
        self, 
        train_epoch_fn, 
        test_epoch_fn,
        train_loader,
        test_loader,
        optimizer,
        n_epochs,
    ):
        
        early_stopping_counter = 0
        best_test_loss = float('inf')

        for epoch in range(n_epochs):
            train_epoch_fn(self, train_loader, optimizer)
            test_loss, test_preds = test_epoch_fn(self, test_loader)
            
            # print(f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Test loss: {test_loss:.3f} | Num Feats: {len([z for z in self.get_smooth_z() if z > 0])}")

            # Check if the test loss has improved
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stopping_counter = 0
                
                # Save the model at the best test loss
                torch.save(self.state_dict(), "../model_saves/tmp_best_model.pt")
                
            else:
                early_stopping_counter += 1

            # If test loss has not improved for 5 consecutive epochs, terminate training
            if early_stopping_counter >= 5:
                print(f"Early stopping at {epoch+1} epochs: Test loss has not improved for 5 consecutive epochs.")
                break
            
        # Load the model from the best test loss
        self.load_state_dict(torch.load("../model_saves/tmp_best_model.pt"))

        return
    
class SATransformer(nn.Module):
    """   
    SATransformer
    """
    
    def __init__(
        self, 
        d_model, 
        n_eval_times, 
        n_layers, 
        num_heads, 
        d_ff, 
        drop_prob,
        validation_size=0.2,
        n_val_splits=5,
        learning_rate=1e-4,
        max_epochs=100,
        batch_size=128,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_eval_times = n_eval_times
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.drop_prob = drop_prob
        self.validation_size = validation_size
        self.n_val_splits = n_val_splits
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        
        self.model_args = kwargs
        
    def preprocess_data(self, X):
        
        if not hasattr(self, 'preprocessor'):
            self.preprocessor = make_column_transformer(
                (
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary"),
                    X.select_dtypes(include="category").columns
                ),
                (
                    make_pipeline(
                        StandardScaler(),
                        SimpleImputer(strategy="constant", fill_value=0)
                    ),
                    X.select_dtypes(include="number").columns
                ),
                remainder="passthrough",
                verbose_feature_names_out=False
            ).set_output(transform="pandas")
            
            X = self.preprocessor.fit_transform(X)
            
        else:
            X = self.preprocessor.transform(X)
            
        self.n_feats = X.shape[1]
        return X    
        
    
    def get_data_loader(self, X, y, pairs=None, shuffle=True):
        
        assert "time" in y.dtype.names and "event" in y.dtype.names, \
            "y must be a structured array with 'time' and 'event' fields."
        
        # Convert X to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if pairs is not None:
            if hasattr(pairs, 'values'):
                pairs = pairs.values

        if pairs is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(pairs),
                torch.BoolTensor(y["event"]),
                torch.FloatTensor(y["time"].copy()),
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), 
                torch.BoolTensor(y["event"]),
                torch.FloatTensor(y["time"].copy()),
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0

        for X, events, times in tqdm(train_loader, leave=False):

            X, events, times = X.to(self.device), events.to(self.device), times.to(self.device)

            y_pred = model(X).squeeze()
            
            surv_probs = torch.cumprod(y_pred, dim=1)
            
            loss =  bce_surv_loss(surv_probs, events, times, self.eval_times)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def test_epoch(self, model, test_loader):
        model.eval()
        total_loss = 0
        preds = []
        
        with torch.no_grad():
            for X, events, times in tqdm(test_loader, leave=False):

                X, events, times = X.to(self.device), events.to(self.device), times.to(self.device)

                y_pred = model(X).squeeze()
        
                preds.append(y_pred.detach())
            
                surv_probs = torch.cumprod(y_pred, dim=1)
        
                loss =  bce_surv_loss(surv_probs, events, times, self.eval_times)
                    
                total_loss += loss.item() 
        
        return total_loss / len(test_loader), torch.cat(preds)
    
    def fit_one_split(self, X_train, y_train, X_val, y_val):
        # If selected_feats is set, only use those features
        if hasattr(self, 'selected_feats'):
            X_train = X_train[self.selected_feats]
            X_val = X_val[self.selected_feats]
        
        # Get data loaders
        train_loader = self.get_data_loader(X_train, y_train)
        val_loader = self.get_data_loader(X_val, y_val, shuffle=False)
        
        model = SATransformerSingleSplit(
            n_feats=self.n_feats, 
            d_model=self.d_model, 
            n_eval_times=self.n_eval_times, 
            n_layers=self.n_layers, 
            num_heads=self.num_heads, 
            d_ff=self.d_ff, 
            drop_prob=self.drop_prob, 
            device=self.device
        ).to(self.device)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train_(
            train_epoch_fn=self.train_epoch,
            test_epoch_fn=self.test_epoch,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            n_epochs=self.max_epochs
        )
        
        return model
    
    def fit(self, X, y):
        
        # Preprocess features
        X = self.preprocess_data(X)
        
        self.feature_names_in_ = X.columns
        self.n_feats = X.shape[1]
        
                # Get evaluation times before train/val split
        quantiles = torch.quantile(
            torch.FloatTensor(y["time"].copy()),
            torch.linspace(0, 1, self.n_eval_times+2)
        )

        self.eval_times = quantiles[1:-1].to(self.device)
        
        # Remove duplicates in eval times
        if len(self.eval_times.unique()) < len(self.eval_times):
            self.eval_times = self.eval_times.unique()
            self.n_eval_times = len(self.eval_times)
        
        
        # Fit several models, one for each validation split
        self.models = []
        for i in range(self.n_val_splits):
            print("SPlIT", i)
        
            # Split the data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_size, random_state=10+i)
            
            # Fit to this split
            model = self.fit_one_split(X_train, y_train, X_val, y_val)
            model.feature_names_in_ = X_train.columns   
            
            self.models.append(model)
            
        return
    
    def predict(self, X_test):
        
        # Preprocess features
        X_test = self.preprocess_data(X_test)
        
        # Make placeholder y_test
        y_test = np.zeros(X_test.shape[0], dtype=[("event", "?"), ("time", "f8")])
        
        test_loader = self.get_data_loader(X_test, y_test, shuffle=False)
        
        test_preds = np.zeros((self.n_val_splits, X_test.shape[0], len(self.eval_times)))
        for i, model in enumerate(self.models):
            _, model_preds = self.test_epoch(model, test_loader)
            test_preds[i, ...] = model_preds.cpu().numpy()
            
        return np.mean(test_preds, axis=0)
    
    def get_calibration_data(self, X, y, eval_time, n_bins=20, method="quantile"):
        
        # First get cdf preds
        cdf_preds = 1 - np.cumprod(self.predict(X), axis=1)
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
            predicted.append(bin_data.mean())
            observed.append(
                1 - surv_prob[np.clip(
                    np.searchsorted(times, eval_time), 0, len(times)-1
                )]
            )
            
            
        return np.array(predicted), np.array(observed), quantiles