def single_pass(model, x_all, loader, single_layer, device):
    _, n_id, adjs = next(loader)

    if single_layer:
        adjs = [adjs]

    adjs = [adj.to(device) for adj in adjs]

    ebds = model(x_all[n_id].to(device), adjs)

    return ebds
