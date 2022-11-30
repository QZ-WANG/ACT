from models.gnns.base_gnns import GraphSAGE, GAT

# INPUT_DIM = 10000 # for debug
# INPUT_TEST = 10000
# HIDDEN_DIM = 128
# EBD_DIM = 64
# N_CLASSES = 2
# N_LAYERS = 3


def build_feature_extractor(backbone_name, args, source):
    if backbone_name == 'sage':
        print("Feature extractor using: GraphSage")

        if (args.source is not None or args.target is not None) and source is not None:
            domain_args = args.source if source else args.target

            return GraphSAGE(in_channels=domain_args['input_dim'],
                             hidden_channels=domain_args['hidden_dim'],
                             out_channels=domain_args['ebd_dim'],
                             dropout=domain_args['drop_out'],
                             num_layers=domain_args['n_layers'],
                             output_type="ebd",
                             adj_dropout=args.adj_dropout)
        else:
            return GraphSAGE(in_channels=args.input_dim, hidden_channels=args.hidden_dim, out_channels=args.ebd_dim,
                             dropout=args.dropout, num_layers=args.n_layers, output_type="ebd")
    elif backbone_name.lower() == 'gat':
        print("Feature extractor using: GAT")

        if (args.source is not None or args.target is not None) and source is not None:
            domain_args = args.source if source else args.target
            return GAT(in_channels=domain_args['input_dim'],
                   hidden_channels=domain_args['hidden_dim'],
                   out_channels=domain_args['ebd_dim'],
                   num_layers=domain_args['n_layers'],
                   heads=1)
        else:
            return GAT(in_channels=args.input_dim, hidden_channels=args.hidden_dim, out_channels=args.ebd_dim,
                       num_layers=args.n_layers, heads=1)
    else:
        raise NotImplementedError("Backbone not available")