from models.losses.deviation_loss import DeviationLoss
import torch

def build_criterion(criterion, confidence_margin=5.0):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss(5000, 0, 1.0, confidence_margin)
    elif criterion == "bce":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError