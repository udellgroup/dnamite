import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss_fns import ipcw_rps_loss, bce_surv_loss

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train_epoch_nam_mains(model, optimizer, train_loader):
    model.train()
    total_loss = 0

    for X_main, labels in tqdm(train_loader, leave=False):

        X_main, labels = X_main.to(device), labels.to(device)

        y_pred = model(mains=X_main).squeeze(-1)
        
        loss = F.binary_cross_entropy_with_logits(y_pred, labels)
        
        if model.penalized:
            loss += model.loss_penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test_epoch_nam_mains(model, test_loader):
    model.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for X_main, labels in tqdm(test_loader, leave=False):

            X_main, labels = X_main.to(device), labels.to(device)

            y_pred = model(mains=X_main).squeeze(-1)
    
            preds.append(y_pred.detach())
        
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            
            if model.penalized:
                loss += model.loss_penalty()
                
            total_loss += loss.item() 
    
    return total_loss / len(test_loader), torch.cat(preds)

def train_epoch_nam_pairs(model_pairs, optimizer, train_loader, *, model_mains):
    model_pairs.train()
    total_loss = 0

    for X_main, X_pairs, labels in tqdm(train_loader, leave=False):

        X_main, X_pairs, labels = X_main.to(device), X_pairs.to(device), labels.to(device)

        with torch.no_grad():
            main_preds = model_mains(mains=X_main).squeeze(-1)
        
        pair_preds = model_pairs(mains=None, pairs=X_pairs).squeeze(-1)
        
        y_pred = main_preds + pair_preds
        
        loss = F.binary_cross_entropy_with_logits(y_pred, labels)
        
        if model_pairs.penalized:
            loss += model_pairs.loss_penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test_epoch_nam_pairs(model_pairs, test_loader, *, model_mains):
    model_pairs.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for X_main, X_pairs, labels in tqdm(test_loader, leave=False):

            X_main, X_pairs, labels = X_main.to(device), X_pairs.to(device), labels.to(device)

            main_preds = model_mains(mains=X_main).squeeze(-1)
            pair_preds = model_pairs(pairs=X_pairs).squeeze(-1)
            
            y_pred = main_preds + pair_preds
    
            preds.append(y_pred.detach())
        
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            
            if model_pairs.penalized:
                loss += model_pairs.loss_penalty()
                
            total_loss += loss.item() 
    
    return total_loss / len(test_loader), torch.cat(preds)


def train_epoch_nam_survival_mains(model, optimizer, train_loader, eval_times, pcw_eval_times):
    model.train()
    total_loss = 0

    for X_main, events, times, pcw_obs_times in tqdm(train_loader, leave=False):

        X_main, events, times, pcw_obs_times = X_main.to(device), events.to(device), times.to(device), pcw_obs_times.to(device)

        y_pred = model(mains=X_main)
        
        cdf_preds = torch.sigmoid(y_pred)
        
        loss = ipcw_rps_loss(
            cdf_preds, 
            pcw_eval_times,
            pcw_obs_times, 
            events,
            times,
            eval_times
        )
        
        if model.penalized:
            loss += model.loss_penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test_epoch_nam_survival_mains(model, test_loader, eval_times, pcw_eval_times):
    model.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for X_main, events, times, pcw_obs_times in tqdm(test_loader, leave=False):

            X_main, events, times, pcw_obs_times = X_main.to(device), events.to(device), times.to(device), pcw_obs_times.to(device)

            y_pred = model(mains=X_main)
    
            preds.append(y_pred.detach())
        
            cdf_preds = torch.sigmoid(y_pred)
        
            loss = ipcw_rps_loss(
                cdf_preds, 
                pcw_eval_times,
                pcw_obs_times, 
                events,
                times,
                eval_times
            )
            
            if model.penalized:
                loss += model.loss_penalty()
                
            total_loss += loss.item() 
    
    return total_loss / len(test_loader), torch.cat(preds)

def train_epoch_nam_survival_pairs(model_pairs, optimizer, train_loader, eval_times, pcw_eval_times, *, model_mains):
    model_pairs.train()
    total_loss = 0

    for X_main, X_pairs, events, times, pcw_obs_times in tqdm(train_loader, leave=False):

        X_main, X_pairs, events, times, pcw_obs_times = \
            X_main.to(device), X_pairs.to(device), events.to(device), times.to(device), pcw_obs_times.to(device)

        with torch.no_grad():
            main_preds = model_mains(mains=X_main)
        
        pair_preds = model_pairs(mains=None, pairs=X_pairs)
        
        y_pred = main_preds + pair_preds
        
        cdf_preds = torch.sigmoid(y_pred)
        
        loss = ipcw_rps_loss(
            cdf_preds, 
            pcw_eval_times,
            pcw_obs_times, 
            events,
            times,
            eval_times
        )
            
        
        if model_pairs.penalized:
            loss += model_pairs.loss_penalty()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test_epoch_nam_survival_pairs(model_pairs, test_loader, eval_times, pcw_eval_times, *, model_mains):
    model_pairs.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for X_main, X_pairs, events, times, pcw_obs_times in tqdm(test_loader, leave=False):

            X_main, X_pairs, events, times, pcw_obs_times = \
                X_main.to(device), X_pairs.to(device), events.to(device), times.to(device), pcw_obs_times.to(device)

            main_preds = model_mains(mains=X_main)
            pair_preds = model_pairs(pairs=X_pairs)
            
            y_pred = main_preds + pair_preds
    
            preds.append(y_pred.detach())
        
            cdf_preds = torch.sigmoid(y_pred)
            
            loss = ipcw_rps_loss(
                cdf_preds, 
                pcw_eval_times,
                pcw_obs_times, 
                events,
                times,
                eval_times
            )
            
            if model_pairs.penalized:
                loss += model_pairs.loss_penalty()
                
            total_loss += loss.item() 
    
    return total_loss / len(test_loader), torch.cat(preds)

def train_epoch_sa_transformer(model, optimizer, train_loader, eval_times):
    model.train()
    total_loss = 0

    for X_main, events, times in tqdm(train_loader, leave=False):

        X_main, events, times = X_main.to(device), events.to(device), times.to(device)

        y_pred = model(X_main)
        
        surv_probs = torch.cumprod(y_pred, dim=1)
        
        loss =  bce_surv_loss(surv_probs, events, times, eval_times)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test_epoch_sa_transformer(model, test_loader, eval_times):
    model.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for X_main, events, times in tqdm(test_loader, leave=False):
            X_main, events, times = X_main.to(device), events.to(device), times.to(device)

            y_pred = model(X_main)
    
            preds.append(y_pred.detach())
        
            surv_probs = torch.cumprod(y_pred, dim=1)
        
            loss = bce_surv_loss(surv_probs, events, times, eval_times)

            total_loss += loss.item() 
    
    return total_loss / len(test_loader), torch.cat(preds)