import warnings
from runners.base_runner import BaseRunner
from utils.optim import get_optimiser, get_scheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.loader import NeighborSampler
from models.samplers.sage_sampler import SageUnsupSampler

from geomloss import SamplesLoss


class ACTRunner(BaseRunner):
    def __init__(self, x_all, edge_all, models, labels, optim_parm, device, out_paths, k, args, ckpt_dir=None):
        super(ACTRunner, self).__init__(None, None, None, optim_parm, device, out_paths=out_paths,
                                        resume_from=None, ckpt_dir=ckpt_dir)
        self.labels = None
        self.source_x_all = x_all['source']
        self.target_x_all = x_all['target']
        self.source_edge_idx = edge_all['source']
        self.target_edge_idx = edge_all['target']
        self.source_in_loader = None
        self.target_in_loader = self.target_train_loader = self.target_test_loader = None

        self.source_model = models['source'].to(self.device)
        self.target_model = models['target'].to(self.device)

        self.source_labels = torch.tensor(labels['source'].flatten()).long().to(device)
        self.target_labels = torch.tensor(labels['target'].flatten()).long().to(device)

        self.source_inlier_idx = None
        self.source_outlier_idx = None
        self.target_inlier_idx = None
        self.target_outlier_idx = None

        self.all_dists = None
        self.all_pred_ranks = None

        self.target_optimiser = get_optimiser(
            self.optim_parm['adapt']['name'],
            self.target_model.parameters(),
            lr=self.optim_parm['adapt']['lr'],
            weight_decay=self.optim_parm['adapt']['weight_decay'],
        )

        self.k = args.batch_size
        self.args = args

        self.single_layer = self.source_model.feature_extractor == 1

        self.target_detector = nn.Linear(self.args.target['ebd_dim'], 1)

        self.scheduler = None
        self.target_x_all_aug = None

        self.set_loaders()

    def set_loaders(self):
        self.source_inlier_idx = (self.source_labels == 0).nonzero(as_tuple=True)[0]
        self.source_outlier_idx = (self.source_labels == 1).nonzero(as_tuple=True)[0]

        # Randomly select k inliers as the known
        self.target_inlier_idx = (self.target_labels == 0).nonzero(as_tuple=True)[0]
        self.target_outlier_idx = (self.target_labels == 1).nonzero(as_tuple=True)[0]
        k_target_inlier_idx = self.target_inlier_idx[torch.randperm(self.target_inlier_idx.size(0))[:self.k]].sort()[0]

        print("Source\n inliers: %d    outlies: %d" %(self.source_inlier_idx.size(0), self.source_outlier_idx.size(0)))
        print("Target\n inliers: %d    outlies: %d" % (self.target_inlier_idx.size(0), self.target_outlier_idx.size(0)))

        self.target_inlier_idx = self.target_inlier_idx

        self.target_test_loader = NeighborSampler(
            self.target_edge_idx,
            node_idx=None,
            sizes=self.args.target['sampling_sizes'],
            batch_size=self.target_x_all.size(0),
            shuffle=False,
            num_workers=self.args.n_workers,
            drop_last=False
        )

    def single_pass(self, model, x_all, loader, single_layer):
        _, n_id, adjs = next(loader)

        if single_layer:
            adjs = [adjs]

        adjs = [adj.to(self.device) for adj in adjs]

        ebds = model(x_all[n_id].to(self.device), adjs)

        return ebds

    def sage_unsup_train(self, n_epochs, q=1):
        self.target_model.reset_parameters()

        self.target_model.train()

        train_loader = SageUnsupSampler(edge_index=self.target_edge_idx,
                                        sizes=self.args.target['sampling_sizes'],
                                        node_idx=None,
                                        num_nodes=self.target_x_all.size(0),
                                        q=q,
                                        batch_size=self.args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=self.args.n_workers)

        print("==== Adapting...")

        sinkhorn = SamplesLoss(loss='sinkhorn',
                               p=self.args.sh_p,
                               scaling=self.args.sh_scaling,
                               blur=self.args.sh_blur)

        n_batch_per_epoch = len(train_loader)

        print("Benchmarking the initial state   ...")
        auroc, aupr = self.val_target("src")
        print("Init. auroc: %.4f    aupr: %.4f" % (auroc, aupr))

        self.write_tb({"val_auroc": auroc}, 0)

        if self.args.adapt['scheduler'] is not None:
            print("getting scheduler")
            self.scheduler = get_scheduler(self.target_optimiser, self.args.adapt['scheduler'])

        for epoch in range(n_epochs):
            print("Epoch %2d" % (epoch))

            src_train_loader = iter(SageUnsupSampler(edge_index=self.source_edge_idx,
                                                     sizes=self.args.source['sampling_sizes'],
                                                     node_idx=None,
                                                     num_nodes=self.source_x_all.size(0),
                                                     q=q,
                                                     batch_size=self.args.batch_size,
                                                     shuffle=True,
                                                     drop_last=True,
                                                     num_workers=self.args.n_workers))

            self.target_model.train()
            for bn, (batch_size, n_id, adjs) in enumerate(train_loader):
                # print(batch_size)
                if bn >= int(self.source_x_all.size(0) / self.args.batch_size):
                    break

                adjs = [adj.to(self.device) for adj in adjs]

                domain_loss = 0.0
                if self.args.adapt['domain_loss']:
                    self.target_optimiser.zero_grad()
                    out = self.target_model(self.target_x_all[n_id].to(self.device), adjs)
                    src_train_ebd = self.single_pass(self.source_model.feature_extractor, self.source_x_all,
                                                     src_train_loader, self.single_layer==1)

                    domain_loss = sinkhorn(out, src_train_ebd)

                    domain_loss.backward()
                    self.target_optimiser.step()

                struct_loss = 0.0
                if self.args.adapt['struct_loss']:
                    self.target_optimiser.zero_grad()

                    out = self.target_model(self.target_x_all[n_id].to(self.device), adjs)
                    out, pos_out, neg_out = out.split((self.args.batch_size, self.args.batch_size * q,
                                                       self.args.batch_size * q), dim=0)

                    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
                    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()

                    struct_loss = - pos_loss - neg_loss

                    struct_loss.backward()
                    if self.args.adapt['clip'] is not None:
                        torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), self.args.adapt['clip'])
                    self.target_optimiser.step()

                batch_perf_dict = {
                    "adapt": {
                        "domain loss": domain_loss,
                        "struct. loss": struct_loss,
                    }
                }
                self.write_tb(batch_perf_dict, n_batch_per_epoch * epoch + bn)

            if (epoch+1) % 10 == 0:
                target_auroc, aupr = self.val_target("src")
                print("Init. auroc: %.4f    aupr: %.4f" % (target_auroc, aupr))

            if self.scheduler is not None:
                print("schduler step")
                self.scheduler.step()

        self.save_state({'epoch': n_epochs-1,
                         'model_state_dict': self.target_model.state_dict(),
                         'optimizer_state_dict': self.target_optimiser.state_dict(),
                         }, "unsup_act_%s" %str(n_epochs-1))

    def val_target(self, detector):
        print("Testing on the target...")

        self.target_model.eval()

        xs = []

        if detector == "src":
            linear = self.source_model.linear
        else:
            linear = self.target_detector

        linear.to(self.device)
        linear.eval()

        with torch.no_grad():
            for batch_size, n_id, adjs in self.target_test_loader:
                if self.target_model.num_layers == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                ebds = self.target_model(self.target_x_all[n_id].to(self.device), adjs)
                out = self.source_model.linear(ebds)
                x = out
                xs.append(x.cpu())

            y_pred = torch.cat(xs, dim=0)

        auroc = roc_auc_score(self.target_labels.cpu().numpy(), y_pred)
        aupr = average_precision_score(self.target_labels.cpu().numpy(), y_pred)
        self.target_model.train()

        return auroc, aupr

    def get_all_embedding(self, model, loader, x_all, n_batches):
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            loader_iter = iter(loader)
            for bn in range(len(loader)):
                if bn == n_batches:
                    break

                ebds = self.single_pass(model, x_all, loader_iter, self.single_layer)

                if bn == 0:
                    all_ebds = ebds
                else:
                    all_ebds = torch.cat((all_ebds, ebds), 0)

        return all_ebds