from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, layer_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.n_layers = len(layer_dims) - 1

        self.fcs = nn.Sequential()

        for i in range(self.n_layers - 1):
            self.fcs.add_module("fc_"+ str(i), nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.fcs.add_module("relu_"+ str(i), nn.ReLU())

        self.fcs.add_module("fc_"+ str(self.n_layers-1), nn.Linear(layer_dims[self.n_layers-1], layer_dims[self.n_layers]))

    def forward(self, x):
        """Forward the discriminator."""
        out = self.fcs(x)
        return out
