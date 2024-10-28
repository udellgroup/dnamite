import torch


def ipcw_rps_loss(
    cdf_preds, 
    pcw_eval_times, 
    pcw_obs_times,
    events, 
    times, 
    eval_times
):
    """  
    IPCW Loss
    """
    
    # cdf_preds is of shape (N, T) where N is batch size and T is len(eval_times)
    #     cdf_preds[i, j] is hat(P)(Ti <= tj | Xi)
    # pcw_eval_times is of shape (T)
    #     pcw_eval_times[j] is hat(P)(C > tj)
    # pcw_obs_times is of shape (N)
    #     pcw_obs_times[i] is hat(P)(C > Ti)
    
    # Maximize survival before event time
    loss = torch.sum(
        (cdf_preds)**2 * \
        (eval_times < times.unsqueeze(-1)).float() / \
        pcw_eval_times.repeat(cdf_preds.shape[0], 1),
        axis=1
    )
    
    # Minimize survival past event time for patients that are not censored
    loss += torch.sum(
        (1 - cdf_preds)**2 * \
        torch.logical_and(
            eval_times >= times.unsqueeze(-1),
            events.unsqueeze(-1)
        ).float() / \
        pcw_obs_times.unsqueeze(-1).repeat(1, cdf_preds.shape[1]),
        axis=1
    )
    
    # Normalize by censor distribution to get correct convergence
    # divisor = (eval_times < censor_times.unsqueeze(-1)).sum(axis=0) + 1e-5
    
    # return torch.sum(loss / divisor)
    return torch.mean(loss)

def rps_loss(surv_preds, events, times, eval_times):
    """  
    RPS Loss
    """
    
    # surv_preds is of shape (N, T)
    # where N is batch size and T is number of evaluation times
    # surv_preds(i, j) gives P(Ti > tj | Xi)
    
    uncensored_loss = torch.square(surv_preds - (eval_times < times.unsqueeze(-1)).float()).sum(axis=1)
    censored_loss = (torch.square(surv_preds - 1) * (eval_times <= times.unsqueeze(-1)).float()).sum(axis=1)
    
    return torch.mean(events.float() * uncensored_loss + (1 - events.float()) * censored_loss)

# Loss taken from Eq 1 of https://dl.acm.org/doi/pdf/10.1145/3534678.3539259
# Can't just use MSE loss because pseudo-values not strictly in [0, 1]
def pseudo_value_loss(
    surv_preds,
    pseudo_values
):
    """  
    pseudo-value loss
    """
    
    return torch.mean(
        pseudo_values * (1 - 2*surv_preds) + torch.square(surv_preds), dim=0
    ).sum()
    
def coxph_loss(y_hat, events, times):
    """
    CoxPH Loss
    """
    
    # Sort y_hat and events by times
    sort_idx = torch.argsort(times)
    y_hat = y_hat[sort_idx]
    events = events[sort_idx]
    
    e = torch.exp(y_hat)
    e_sums = torch.cumsum(e.flip(0), 0).flip(0)
    
    
    return -1 * events.to(torch.float32) @ (y_hat - torch.log(e_sums))

def bce_surv_loss(surv_preds, events, times, eval_times):
    """
    Binary Cross-Entropy Survival Loss
    """
    
    # surv_preds is of shape (N, T)
    # where N is batch size and T is number of evaluation times
    # surv_preds(i, j) gives P(Ti > tj | Xi)
    
    # Maximize survival for all samples before their times
    loss = torch.sum(
        -1 * torch.log(surv_preds + 1e-5) * \
        (eval_times < times.unsqueeze(-1)).float(),
        axis=1
    )
    
    # Minimize survival loss for all uncensored samples after their times
    loss += torch.sum(
        -1 * torch.log(1 - surv_preds + 1e-5) * \
        torch.logical_and(
            eval_times >= times.unsqueeze(-1),
            events.unsqueeze(-1)
        ).float(),
        axis=1
    )
    
    return torch.mean(loss)

def drsa_loss(hazard_preds, events, times, eval_times):
    """
    DRSA-Loss
    """
    
    # Find evaluation time for each event time
    sample_eval_times = torch.clip(
        torch.searchsorted(eval_times, times).unsqueeze(-1),
        0,
        len(eval_times)-1
    )
    
    
    # Get first part of loss
    loss = -1 * events.unsqueeze(-1).float() * (
        torch.log(hazard_preds + 1e-5).gather(1, sample_eval_times) + \
        torch.sum(torch.log(1 - hazard_preds + 1e-5) * (eval_times < times.unsqueeze(-1)).float(), dim=1)
    )
    
    # Get second part of loss
    
    surv_preds = torch.cumprod(1 - hazard_preds, dim=1)
    
    loss += -1 * (~events).float() * torch.log(surv_preds + 1e-5).gather(1, sample_eval_times)
    loss += -1 * events.float() * torch.log(1 - surv_preds + 1e-5).gather(1, sample_eval_times)
    
    return loss.mean()