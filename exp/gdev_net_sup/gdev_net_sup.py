# Source model training - Graph DevNet

import numpy as np
import torch
from torch_geometric.loader import NeighborSampler
from utils.data import load_mat, mat_to_pyg_data, load_yaml, train_test_split
from runners.g_dev_net_runner import GraphDevNetRunner
from models.gnns.graph_dev_net import GraphDevNet
import argparse
from datetime import datetime
from os.path import join
import random
import json


def do_exp():
    if args.timestamp is not None:
        timestamp = args.timestamp
    else:
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")

    print("timestamp: %s" % timestamp)

    data = load_mat(args.dset_dir, args.dset_fn)

    data, ad_labels = mat_to_pyg_data(data, undirected=False)

    if args.use_train_split:
        print("Use a subset (%.2f) for training..." % args.train_ratio)
        (inlier_train_idx, outlier_train_idx), _, val_idx = train_test_split(ad_labels, args.train_ratio, 0.0)
    else:
        inlier_train_idx, outlier_train_idx = np.where(ad_labels == 0)[0], np.where(ad_labels == 1)[0]

    split = {
        'train': torch.from_numpy(np.sort(np.hstack((inlier_train_idx, outlier_train_idx)))),
        'test': None,
        'val': torch.from_numpy(np.sort(val_idx)) if args.use_train_split else None
    }

    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')

    model = GraphDevNet(args.encoder_type, args, source=None)

    optim_param = {
        "lr": args.lr,
        "name": "adam",
        "weight_decay": args.weight_decay,
        "n_epochs": args.n_epochs,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,
    }

    out_batch_size = args.batch_size if outlier_train_idx.shape[0] >= args.batch_size else outlier_train_idx.shape[0]
    in_batch_size = args.batch_size if inlier_train_idx.shape[0] > args.batch_size else inlier_train_idx.shape[0]
    args.in_batch_size = in_batch_size
    print("outlier batch size: ", out_batch_size)

    dataloaders = {
        'train_inlier': NeighborSampler(data.edge_index, node_idx=torch.from_numpy(inlier_train_idx),
                                        sizes=args.sampling_sizes, batch_size=args.in_batch_size,
                                        shuffle=True, num_workers=args.n_workers, drop_last=True),
        'train_outlier': NeighborSampler(data.edge_index, node_idx=torch.from_numpy(outlier_train_idx),
                                         sizes=args.sampling_sizes, batch_size=out_batch_size,
                                         shuffle=True, num_workers=args.n_workers, drop_last=True),
        'val': NeighborSampler(data.edge_index, node_idx=None,
                               sizes=args.sampling_sizes,
                               batch_size=data.x.size(0), shuffle=False,
                               num_workers=args.n_workers),
        'test': None
    }

    out_paths = {
        'ckpt_dir': join(args.share_dir, "ckpt", args.exp_name),
        'tb_data': join(args.share_dir, "tb_data", args.exp_name),
        'timestamp': timestamp,
        'exp_name': "gdev_pretrain",
        'task_name': args.dset_name.lower(),
        'plot_dir': join(args.proj_dir, "plot", "gdev_pretrain", args.dset_name.lower()),
    }

    runner = GraphDevNetRunner(data.x.float(), dataloaders, split=split, model=model, optim_parm=optim_param,
                               device=device, labels=ad_labels, out_paths=out_paths, resume_from=None,
                               criterion="deviation", args=args, ckpt_dir=args.ckpt_dir)

    runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str, default="/media/nvme1/pycharm_mirror/GUDA_release")
    parser.add_argument("--config_fn", type=str, default="config_res")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    if args.config_fn is not None:
        config = load_yaml(join(args.proj_dir, "exp", "gdev_net_sup/config", args.config_fn + ".yaml"))
        args = argparse.Namespace(**{**vars(args), **config})
    else:
        raise RuntimeError("Config file not found!")

    print(json.dumps(vars(args), sort_keys=False, indent=4))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("use train split %s" % args.use_train_split)

    do_exp()
