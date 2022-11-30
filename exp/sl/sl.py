import numpy as np
import torch

from utils.data import load_mat, mat_to_pyg_data, load_yaml

from models.gnns.graph_dev_net import GraphDevNet
from models.gnns import build_feature_extractor
import argparse

from datetime import datetime
import random

import json

import os
from os.path import join

from utils.ranking import pseudo_labeling_eval, sigma_selection
from utils.log import save_config, set_up_logging
from utils.basic_ops import single_pass
from utils.model import load_model

from torch_geometric.loader import NeighborSampler
from runners.sl_runner import SLRunner
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.ensemble import IsolationForest

from torch_geometric.graphgym.init import init_weights


def get_all_ebds(encoder, data, device):
    encoder.eval()

    x_all, edge_index = data.x, data.edge_index

    graph_loader = NeighborSampler(
        edge_index=edge_index,
        node_idx=None,
        sizes=args.labeling_sampling_sizes,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False
    )

    with torch.no_grad():
        graph_iter = iter(graph_loader)

        all_ebds = None

        for i in range(len(graph_loader)):
            ebds = single_pass(encoder, x_all, graph_iter, False, device)

            all_ebds = torch.vstack((all_ebds, ebds)) if all_ebds is not None else ebds

    return all_ebds


def gdev_if_pred(model, data, device):
    model.to(device)
    all_ebds = get_all_ebds(model, data, device)

    all_ebds_np = all_ebds.cpu().numpy()

    clf = IsolationForest(random_state=args.seed, contamination=0.05, n_estimators=500, n_jobs=args.n_workers).fit(
        all_ebds_np)
    pred = torch.from_numpy(abs(clf.score_samples(all_ebds_np)).flatten())
    return pred


def gdev_clf_pred(src_model, target_model, target_data, device):
    src_model.to(device)
    discriminator = src_model.linear
    pred = discriminator(get_all_ebds(target_model, target_data, device)).cpu()

    return pred


def test_apply_src(src_model, target_data, target_labels, device):
    pred = src_model.linear(get_all_ebds(src_model.feature_extractor, target_data, device))
    auroc = roc_auc_score(target_labels.cpu().numpy(), pred.cpu().detach())
    aupr = average_precision_score(target_labels.cpu().numpy(), pred.cpu().detach())
    print("auroc: %.4f, aupr: %.4f" % (auroc, aupr))


def test_one_timestamp(src_data, tar_data, src_labels, tar_labels, ckpts, optim_params, out_paths, device):
    tar_encoder = build_feature_extractor(args.encoder_type, args, source=False)
    tar_model = GraphDevNet(args.encoder_type, args, source=False)

    # src_path = join(args.source['model_dir'], ckpts['src'], args.source['state_fn'])
    tar_path = join(args.target['model_dir'], ckpts['tar'], args.target['state_fn'])
    # print("src path: ", src_path)
    print("tar path: ", tar_path)

    src_model, tar_encoder = None, load_model(tar_encoder, tar_path).to(device)

    # Display dataset information
    n_all, n_in, n_out = tar_data.x.size(0), torch.sum(tar_labels == 1), torch.sum(tar_labels == 0)
    print("Target dataset info: %d nodes, %d outliers (%.4f) and %d inliers" % (n_all, n_in, n_in/n_all, n_out))

    if args.which_label == 'iForest':
        pred = gdev_if_pred(tar_encoder, tar_data, device)
        auroc, aupr = roc_auc_score(tar_labels, pred), average_precision_score(tar_labels, pred)
        print("iForest Performance: auroc=%.3f  aupr=%.3f" % (auroc, aupr))
    else:
        raise NotImplementedError("Self-labelling method: %s not implemented" % args.which_label.upper())

    if args.select == "sigma":
        pseudo_out_idx = sigma_selection(pred, args.init_a_out, args.max_out_ratio, True)
        pseudo_in_idx = sigma_selection(pred, args.init_a_in, args.max_in_ratio, False)
    else:
        raise NotImplementedError("Invalid labelling method.")

    # pseudo_labeling_eval(pseudo_out_idx, torch.where(tar_labels == 1)[0], outlier=True)
    # pseudo_labeling_eval(pseudo_in_idx, torch.where(tar_labels == 0)[0], outlier=False)

    pseudo_idx = {"inlier": pseudo_in_idx, "outlier": pseudo_out_idx}

    init_weights(tar_model)

    runner = SLRunner(src_data, tar_data, tar_model, optim_params, device, tar_labels, out_paths,
                      src_labels, pseudo_idx, args)

    runner.train(src_model)


def do_exp():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    print("timestamp: %s" % timestamp)

    task_name = "%s_%s" % (args.target['dset_name'], args.method_name)

    # Register logging info
    log_dir = join(args.proj_dir, "log", args.exp_name, task_name, timestamp)
    logger = set_up_logging(log_dir, timestamp, "gdev_net_sup_logger")
    save_config(vars(args), log_dir)

    src_data = load_mat(join(args.dset_dir, args.source['dset_name']), args.source['dset_fn'])
    src_data, src_ad_labels = mat_to_pyg_data(src_data, undirected=False)

    tar_data = load_mat(join(args.dset_dir, args.target['dset_name']), args.target['dset_fn'])
    tar_data, tar_ad_labels = mat_to_pyg_data(tar_data, undirected=False)

    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

    print("using device %s" % device)

    src_ad_labels, tar_ad_labels = torch.from_numpy(src_ad_labels), torch.from_numpy(tar_ad_labels),

    tb_data = join(args.share_dir, "tb_data", args.exp_name) if args.batch_id is None else \
        join(args.share_dir, "tb_data", args.exp_name, args.batch_id)

    optim_param = {
        "lr": args.lr,
        "name": args.optim_name,
        "weight_decay": args.weight_decay,
        "scheduler": None,
    }

    out_paths = {
        'ckpt_dir': join(args.share_dir, "ckpt", args.exp_name),
        'tb_data': tb_data,
        'timestamp': timestamp,
        'exp_name': 'pseudo_test',
        'task_name': task_name,
        'plot_dir': join(args.proj_dir, "plot", args.exp_name, task_name),
    }

    # mode  in [pseudo_unsup_test, src_raw_test, src_raw_test]
    if args.mode == "sl":
        for i in range(len(args.target['ckpts'])):

            if isinstance(args.source['ckpt'], list):
                src_ckpt = args.source['ckpt'][i]
            else:
                src_ckpt = args.source['ckpt']

            ts_ckpts = {"src": src_ckpt, "tar": args.target["ckpts"][i]}

            test_one_timestamp(src_data, tar_data, src_ad_labels, tar_ad_labels, ts_ckpts, optim_param, out_paths,
                               device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str, default="/media/nvme1/pycharm_mirror/GUDA_release")
    parser.add_argument("--config_fn", type=str, default="nyc_to_amz")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_id", type=str, default=None)
    parser.add_argument("--ckpts", nargs='+', default=None)
    parser.add_argument("--which_label", type=str, default=None)
    parser.add_argument("--src_ckpt_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    if args.batch_id == "null":
        args.batch_id = None

    print(os.getcwd())

    if args.config_fn is not None:
        config = load_yaml(join(args.proj_dir, "exp", "sl/config", args.config_fn + ".yaml"))
        args = argparse.Namespace(**{**vars(args), **config})
    else:
        raise RuntimeError("Config file not found!")

    print(json.dumps(vars(args), sort_keys=False, indent=4))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.ckpts is not None:
        args.target['ckpts'] = args.ckpts

    if args.src_ckpt_dir is not None:
        args.source['ckpt'] = args.src_ckpt_dir
    do_exp()
