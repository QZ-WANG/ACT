import torch
import torch.nn as nn

# Adapted from https://github.com/Choubo/deviation-network-image/blob/main/modeling/layers/deviation_loss.py
class DeviationLoss(nn.Module):
    def __init__(self, ref_dim, ref_mean, ref_std, confidence_margin=5.0):
        super().__init__()
        self.ref_dim = ref_dim
        self.ref_mean = ref_mean
        self.ref_std = ref_std
        self.confidence_margin = confidence_margin

    def forward(self, y_pred, y_true, device):
        confidence_margin = self.confidence_margin
        # size=5000 is the setting of l in algorithm 1 in the paper
        ref = torch.normal(mean=0., std=torch.full([5000], 1.0)).to(device)
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)