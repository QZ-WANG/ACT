import torch.nn as nn
import torch.nn.functional as F
from models.gnns import build_feature_extractor


class GraphDevNet(nn.Module):
    def __init__(self, backbone_name, args, source):
        super(GraphDevNet, self).__init__()
        self.feature_extractor = build_feature_extractor(backbone_name, args, source)
        in_dim = args.source['ebd_dim'] if args.source is not None else args.ebd_dim

        if args.source is None and args.target is None:
            in_dim = args.ebd_dim
        elif source:
            in_dim = args.source['ebd_dim']
        else:
            in_dim = args.target['ebd_dim']

        self.linear = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, x, adjs):
        embeddings = self.feature_extractor(x, adjs)
        scores = self.linear(embeddings)
        return scores
