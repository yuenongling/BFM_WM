import torch
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
    """
    Compute weighted mean squared error loss
    
    Args:
        inputs: Model predictions
        targets: Ground truth values
        weights: Optional weights tensor of same shape as inputs/targets
        
    Returns:
        Weighted MSE loss value
    """
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    """
    Compute weighted L1 loss
    
    Args:
        inputs: Model predictions
        targets: Ground truth values
        weights: Optional weights tensor of same shape as inputs/targets
        
    Returns:
        Weighted L1 loss value
    """
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    """
    Compute weighted focal MSE loss
    
    Args:
        inputs: Model predictions
        targets: Ground truth values
        weights: Optional weights tensor of same shape as inputs/targets
        activate: Activation function to use ('sigmoid' or 'tanh')
        beta: Scaling factor for activation
        gamma: Exponential factor for focal weighting
        
    Returns:
        Weighted focal MSE loss value
    """
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    """
    Compute weighted Huber loss (smooth L1)
    
    Args:
        inputs: Model predictions
        targets: Ground truth values
        weights: Optional weights tensor of same shape as inputs/targets
        beta: Threshold for switching between L2 and L1 loss
        
    Returns:
        Weighted Huber loss value
    """
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def EWC_loss(model, fisher_matrix, old_params, lambda_ewc=0.1):
    """
    Compute EWC loss for Elastic Weight Consolidation
    
    Args:
        model: The model instance
        fisher_matrix: Fisher information matrix
        old_params: Old parameters of the model
        lambda_ewc: Scaling factor for EWC loss
        
    Returns:
        EWC loss value
    """
    ewc_loss = 0
    for name, param in model.named_parameters():
        if name in fisher_matrix:
            ewc_loss += (fisher_matrix[name] * (param - old_params[name]) ** 2).sum()
    print(f"Calculated EWC loss: {ewc_loss.item()*lambda_ewc:.6f}")
    return lambda_ewc * ewc_loss
