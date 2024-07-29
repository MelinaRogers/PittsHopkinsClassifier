import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implements Focal Loss for binary classification

    Args:
    alpha (float): Balancing factor, default is 0.25
    gamma (float): Focusing parameter, default is 2

    Attributes:
    alpha (float): Balancing factor for the rare class
    gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Computes the Focal Loss

        Args:
        inputs (torch.Tensor): Model predictions (logits)
        targets (torch.Tensor): Ground truth labels

        Returns:
        torch.Tensor: Computed Focal Loss
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)