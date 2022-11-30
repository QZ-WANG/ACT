from runners.base_runner import BaseRunner
from utils.optim import get_optimiser, get_scheduler
from torch_geometric.loader import NeighborSampler
from models.losses import build_criterion

import torch

from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN_START_STR = "self-labelling training..."


class SLRunner(BaseRunner):
    def __init__(self, src_data, tar_data, model, optim_parm, device, labels, out_paths, src_labels, pseudo_idx, args=None):
        super(SLRunner, self).__init__(None, None, model, optim_parm, device, out_paths=out_paths, resume_from=None)
        self.labels = torch.tensor(labels.flatten()).squeeze().to(self.device)
        self.evaluator = None
        self.scheduler = None
        self.single_layer = self.model.feature_extractor.num_layers == 1
        self.args = args
        self.src_edge_index, self.tar_edge_index = src_data.edge_index, tar_data.edge_index
        self.src_x_all, self.tar_x_all = src_data.x, tar_data.x

        self.pseudo_in_idx, self.pseudo_out_idx = pseudo_idx['inlier'], pseudo_idx['outlier']
        self.src_labels = src_labels

        # batch_size = self.pseudo_out_idx.size(0)
        batch_size, n_out = self.args.batch_size, self.pseudo_out_idx.size(0)

        if batch_size >= n_out:
            batch_size = n_out
        else:
            if batch_size * 2 <= n_out:
                if n_out % batch_size >= int(batch_size * 0.6):
                    batch_size = n_out // (n_out // batch_size + 1)

        print("batch size: %d" % batch_size)

        self.inlier_train_loader = NeighborSampler(
            edge_index=self.tar_edge_index,
            node_idx=self.pseudo_in_idx,
            sizes=self.args.sampling_sizes,
            batch_size=self.args.in_batch_size,
            shuffle=True,
            num_workers=self.args.n_workers,
            drop_last=True
        )

        self.outlier_train_loader = NeighborSampler(
            edge_index=self.tar_edge_index,
            node_idx=self.pseudo_out_idx,
            sizes=self.args.sampling_sizes,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.n_workers,
            drop_last=True
        )

        self.test_loader = NeighborSampler(
            edge_index=self.tar_edge_index,
            node_idx=None,
            sizes=self.args.sampling_sizes,
            batch_size=self.tar_x_all.size(0),
            shuffle=False,
            num_workers=self.args.n_workers,
            drop_last=False
        )

        self.src_in_loader = NeighborSampler(
            edge_index=self.src_edge_index,
            node_idx=self.src_labels == 0,
            sizes=self.args.sampling_sizes,
            batch_size=self.args.in_batch_size,
            shuffle=True,
            num_workers=self.args.n_workers,
            drop_last=True
        )

        self.src_out_loader = NeighborSampler(
            edge_index=self.src_edge_index,
            node_idx=self.src_labels == 1,
            sizes=self.args.sampling_sizes,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.n_workers,
            drop_last=True
        )

    def train(self, src_model=None):
        self.model.to(self.device)
        self.optimiser = get_optimiser(self.optim_parm['name'], self.model.parameters(), lr=self.optim_parm['lr'],
                                       weight_decay=self.optim_parm['weight_decay'])

        if self.optim_parm['scheduler'] is not None:
            self.scheduler = get_scheduler(self.optimiser, self.optim_parm['scheduler'],
                                           step_size=self.optim_parm['step_size'], gamma=self.optim_parm['gamma'])

        self.model.train()
        self.labels = self.labels.long()

        n_batch_out, n_batch_in = len(self.outlier_train_loader), len(self.inlier_train_loader)
        src_nb_out, src_nb_in = len(self.src_out_loader), len(self.src_in_loader)

        print("%3d batches in out loader and %3d batches in in loader" % (n_batch_in, n_batch_out))

        criterion = build_criterion("deviation", self.args.dev_conf_margin)
        pseudo_in_iter, pseudo_out_iter = iter(self.inlier_train_loader), iter(self.outlier_train_loader)

        print(TRAIN_START_STR)

        print("%3d batches per epoch" % n_batch_out)

        for step in range(self.args.total_steps):
            self.optimiser.zero_grad()

            if step > 0 and step % n_batch_out == 0:
                pseudo_out_iter = iter(self.outlier_train_loader)

            if step > 0 and step % n_batch_in == 0:
                pseudo_in_iter = iter(self.inlier_train_loader)

            out_ebds, out_n_ids = self.single_pass(self.model, self.tar_x_all, pseudo_out_iter, self.single_layer)
            in_ebds, in_n_ids = self.single_pass(self.model, self.tar_x_all, pseudo_in_iter, self.single_layer)

            all_ebds = torch.vstack((out_ebds, in_ebds))
            all_labels = torch.hstack((torch.ones(out_ebds.size(0), dtype=torch.long),
                                       torch.zeros(in_ebds.size(0), dtype=torch.long))).to(self.device)
            loss = criterion(all_ebds, all_labels.unsqueeze(1).float(), self.device)
            loss.backward()

            if self.args.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            self.optimiser.step()

            if loss <= 1.0:
                print("step: %3d, loss: %.4f" % (step, loss))
                auroc, aupr = self.fs_eval()
                print("auroc: %.4f    aupr: %.4f" % (auroc, aupr))
                break

            if step % 10 == 0:
                print("step: %3d, loss: %.4f" % (step, loss))
                auroc, aupr = self.fs_eval()
                print("auroc: %.4f    aupr: %.4f" % (auroc, aupr))

    def fs_eval(self):
        self.model.eval()

        xs = []

        with torch.no_grad():
            for batch_size, n_id, adjs in self.test_loader:
                if self.single_layer == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                out = self.model(self.tar_x_all[n_id].to(self.device), adjs)
                x = out
                xs.append(x.cpu())

        x_all = torch.cat(xs, dim=0)

        # y_pred = torch.exp(x_all).permute(1, 0)[1]

        y_pred = x_all
        auroc = roc_auc_score(self.labels.cpu().numpy(), y_pred)
        aupr = average_precision_score(self.labels.cpu().numpy(), y_pred)
        self.model.train()
        return auroc, aupr

    def single_pass(self, model, x_all, loader, single_layer):
        _, n_id, adjs = next(loader)

        if single_layer:
            adjs = [adjs]

        adjs = [adj.to(self.device) for adj in adjs]

        ebds = model(x_all[n_id].to(self.device), adjs)

        return ebds, n_id[:adjs[1].size[1]]

    def get_all_embedding(self, model, loader, x_all, n_batches):
        model.to(self.device)
        model.eval()

        loader_iter = iter(loader)
        for bn in range(len(loader)):
            if bn == n_batches:
                break

            ebds, _ = self.single_pass(model, x_all, loader_iter, self.single_layer)

            if bn == 0:
                all_ebds = ebds
            else:
                all_ebds = torch.cat((all_ebds, ebds), 0)

        model.train()
        return all_ebds