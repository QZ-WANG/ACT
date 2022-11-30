import numpy as np
import torch
from utils.data import load_mat, mat_to_pyg_data, load_yaml
from runners.act_runner import ACTRunner

from models.gnns.graph_dev_net import GraphDevNet

import argparse

from os.path import join

from datetime import datetime
import random
import json
import os

from utils.model import load_model
from models.gnns import build_feature_extractor

def do_exp():
    if args.timestamp is None:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    else:
        timestamp = args.timestamp

    task_name = args.source['dset_name'] + "2" + args.target['dset_name']

    print("timestamp: %s" % timestamp)

    src_data = load_mat(join(args.dset_dir, args.source['dset_name']), args.source['dset_fn'])
    src_data, src_ad_labels = mat_to_pyg_data(src_data, undirected=False)

    tar_data = load_mat(join(args.dset_dir, args.target['dset_name']), args.target['dset_fn'])
    tar_data, tar_ad_labels = mat_to_pyg_data(tar_data, undirected=False)

    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

    print("using device %s" % device)

    source_model = GraphDevNet(args.encoder_type, args, source=True)

    # fn = ("best" if args.source['which_state'] == "best" else "last") + ".pt"
    src_ckpt_dir = args.src_ckpt_dir if args.src_ckpt_dir is not None else args.source['model_path']
    print("using source state: %s" % src_ckpt_dir)
    state_path = join(src_ckpt_dir, args.source['ckpt'])
    #
    source_model = load_model(source_model, state_path)

    target_model = build_feature_extractor(args.encoder_type, args, source=False)

    x_all = {
        'source': src_data.x,
        'target': tar_data.x,
    }

    edge_all = {
        'source': src_data.edge_index,
        'target': tar_data.edge_index,
    }

    labels = {
        'source': src_ad_labels,
        'target': tar_ad_labels,
    }

    models = {
        "source": source_model,
        "target": target_model,
        # "classifier": classifier,
    }

    optim_param = {
        "adapt": {
            "n_epochs": args.adapt['n_epochs'],
            "lr": args.adapt['domain_lr'],
            "name": args.adapt['name'],
            "weight_decay": args.adapt['weight_decay'],
        },
    }

    tb_data = join(args.share_dir, "tb_data", args.exp_name) \
        if args.batch_id is None else join(args.share_dir, "tb_data", args.exp_name, args.batch_id)

    out_paths = {
        'ckpt_dir': join(args.share_dir, "ckpt", args.exp_name),
        'tb_data': tb_data,
        'timestamp': timestamp,
        'exp_name': args.exp_name,
        'task_name': task_name,
        'plot_dir': join(args.proj_dir, "plot", args.exp_name, task_name),
        'pred_rank_dir': join(args.proj_dir, "stats", "pred_ranks", task_name, timestamp),
    }

    if args.target['adapt'] and args.target['model_path'] is None:
        runner = ACTRunner(x_all, edge_all, models, labels, optim_param, device, out_paths, 128, args,
                                 ckpt_dir=args.ckpt_dir)

        if args.graph_obj == "sage_unsup":
            runner.sage_unsup_train(args.adapt['n_epochs'])
        else:
            raise NotImplementedError("Invalid Graph Objective")
    elif not args.target['adapt'] and args.target['model_path'] is not None:
        target_model_path = join(args.target['model_path'], "unsup_warmup_49.pt")
        models['target'] = load_model(models['target'], target_model_path)
        runner = MultiViewRunner(x_all, edge_all, models, labels, optim_param, device, out_paths, 128, args)
        print("target model loaded successfully, testing base auroc...")
        print("auroc", runner.val_target("src"))
    else:
        raise NotImplementedError("Mode not implemented...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str, default="/media/nvme1/pycharm_mirror/GDA")
    parser.add_argument("--config_fn", type=str, default="config_res_hotel_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_id", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--src_ckpt_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    if args.batch_id == "null":
        args.batch_id = None

    print(os.getcwd())

    if args.config_fn is not None:
        config = load_yaml(join(args.proj_dir, "exp", "act/config", args.config_fn + ".yaml"))
        args = argparse.Namespace(**{**vars(args), **config})
    else:
        raise RuntimeError("Config file not found!")

    print(json.dumps(vars(args), sort_keys=False, indent=4))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    do_exp()