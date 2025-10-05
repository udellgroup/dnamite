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
    IPCW Loss function for survival analysis.
    Alternative to the Cox loss that does not assume proportional hazards.
    Requires CDF estimates at specified evaluation times.
    For mathematical details, see Equation 6 in https://arxiv.org/pdf/2411.05923?.
    
    Parameters
    ----------
    cdf_preds : torch.Tensor
        Predicted cumulative distribution function values at eval_times.
        Shape: (N, T) where N is batch size and T is len(eval_times).
    pcw_eval_times : torch.Tensor
        Estimated probability of censoring at evaluation times.
        Shape: (T)
    pcw_obs_times : torch.Tensor
        Estimated probability of censoring at observed times.
        Shape: (N)
    events : torch.Tensor
        Event indicators, where 1 indicates event occurred and 0 indicates censored.
        Shape: (N)
    times : torch.Tensor
        Observed event times.
        Shape: (N)
    eval_times : torch.Tensor
        Evaluation times.
        Shape: (T)
        
    Returns
    -------
    torch.Tensor
        Computed IPCW loss.
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
    
def coxph_loss(y_hat, events, times):
    """
    Cox Proportional Hazards loss function for survival analysis.
    
    Parameters
    ----------
    y_hat : torch.Tensor
        Predicted log-risk scores. Shape: (N)
    events : torch.Tensor
        Event indicators, where 1 indicates event occurred and 0 indicates censored. Shape: (N)
    times : torch.Tensor
        Observed event times. Shape: (N)
        
    Returns
    -------
    torch.Tensor
        Computed CoxPH loss.
    """
    
    # Sort y_hat and events by times
    sort_idx = torch.argsort(times)
    y_hat = y_hat[sort_idx]
    events = events[sort_idx]
    
    e = torch.exp(y_hat)
    e_sums = torch.cumsum(e.flip(0), 0).flip(0)
    
    
    return -1 * events.to(torch.float32) @ (y_hat - torch.log(e_sums))
