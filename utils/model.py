import torch


def load_model(model, path):
    if path is not None:
        print("Loading model from: %s" %path)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint.pop('model_state_dict'))
        return model
    else:
        raise RuntimeError("Checkpoint is None")
