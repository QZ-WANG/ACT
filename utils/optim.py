import torch.optim as optim


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimiser(name, param, lr, weight_decay):
    if name.lower() == 'adam':
        optimiser = optim.Adam(param, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'adamw':
        optimiser = optim.AdamW(param, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'sgd':
        optimiser = optim.SGD(param, lr=lr)
    else:
        raise NotImplementedError("Optimiser function not supported!")
    return optimiser


def get_scheduler(optimiser, name, **kwargs):
    if name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=kwargs.get('step_size'), gamma=kwargs.get('gamma'))
    elif name == "cos_ann":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=5, eta_min=1.0e-04)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, 10, eta_min=0.0001)
    else:
        raise NotImplementedError

    return scheduler
