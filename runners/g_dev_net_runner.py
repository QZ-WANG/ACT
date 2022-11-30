# Source training runner

from runners.base_runner import BaseRunner
from utils.optim import get_optimiser, get_scheduler, get_lr

from models.losses import build_criterion

import torch

from utils.evaluation import SupEvaluator


class GraphDevNetRunner(BaseRunner):
    def __init__(self, x_all, dataloaders, split, model, optim_parm, device, labels, out_paths, resume_from, criterion, args=None, ckpt_dir=None):
        super(GraphDevNetRunner, self).__init__(x_all, split, model, optim_parm, device, out_paths=out_paths,
                                                resume_from=resume_from, ckpt_dir=ckpt_dir)
        self.labels = torch.tensor(labels.flatten()).squeeze().to(self.device)
        self.inlier_train_loader = dataloaders['train_inlier']
        self.outlier_train_loader = dataloaders['train_outlier']
        self.test_loader = dataloaders['test']
        self.val_loader = dataloaders['val']
        self.evaluator = SupEvaluator(split=self.split, labels=self.labels)
        self.criterion = build_criterion(criterion)
        self.scheduler = None
        self.single_layer = self.model.feature_extractor.num_layers == 1
        self.args = args

    def train(self):
        self.model.to(self.device)
        self.optimiser = get_optimiser(self.optim_parm['name'], self.model.parameters(), lr=self.optim_parm['lr'],
                                       weight_decay=self.optim_parm['weight_decay'])

        if self.optim_parm['scheduler'] is not None:
            self.scheduler = get_scheduler(self.optimiser, self.optim_parm['scheduler'],
                                           step_size=self.optim_parm['step_size'], gamma=self.optim_parm['gamma'])

        self.model.feature_extractor.reset_parameters()

        best_val_auroc = start_epoch = 0
        self.model.train()
        self.labels = self.labels.long()

        n_batches = len(self.outlier_train_loader)

        print("Graph Deviation Network training...")
        print("%3d batches per epoch" % n_batches)

        for epoch in range(start_epoch, self.optim_parm['n_epochs'] + start_epoch):
            epoch_str = "Epoch %2d, lr: %f" % (epoch, get_lr(self.optimiser))
            print(epoch_str)

            total_loss = batch_num = 0
            inlier_train_loader = iter(self.inlier_train_loader)
            for batch_size, n_id, adjs in self.outlier_train_loader:
                batch_num += 1

                if self.model.feature_extractor.num_layers == 1:
                    adjs = [adjs]

                adjs = [adj.to(self.device) for adj in adjs]
                self.optimiser.zero_grad()
                out = self.model(self.x_all[n_id].to(self.device), adjs)
                loss = self.criterion(out, self.labels[n_id[:batch_size]].unsqueeze(1).float(), self.device)

                batch_size1, n_id_1, adjs_1 = next(inlier_train_loader)

                if self.model.feature_extractor.num_layers == 1:
                    adjs_1 = [adjs_1]

                adjs_1 = [adj.to(self.device) for adj in adjs_1]
                out1 = self.model(self.x_all[n_id_1].to(self.device), adjs_1)
                loss1 = self.criterion(out1, self.labels[n_id_1[:self.args.in_batch_size]].unsqueeze(1).float(), self.device)

                loss += loss1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimiser.step()
                total_loss += float(loss)

            if self.scheduler is not None:
                self.scheduler.step()

            loss = total_loss / len(self.outlier_train_loader)

            auroc_train = auroc_val = .0

            if epoch % 10 == 0:
                if self.args.use_train_split:
                    auroc_train, auroc_val = self.val()
                else:
                    auroc_val = self.val()

                eval_str = "loss: %.4f, auroc_train: %.4f, auroc_val: %.4f" % (loss, auroc_train, auroc_val)
                print(eval_str)

            if auroc_val > best_val_auroc or epoch % self.args.save_every == 0:
                state_dict = {'epoch': epoch,
                              'model_state_dict': self.model.state_dict(),
                              'optimizer_state_dict': self.optimiser.state_dict(),
                              'loss': loss,
                              'auroc_best': auroc_val,
                              'auroc_last': best_val_auroc,}

                state_name = "best" if auroc_val > best_val_auroc else str(epoch)

                self.save_state(state_dict, state_name)

                best_val_auroc = auroc_val

            if epoch == start_epoch + self.optim_parm['n_epochs'] - 1 and batch_num == n_batches:
                self.save_state({'epoch': epoch,
                                 'model_state_dict': self.model.state_dict(),
                                 'optimizer_state_dict': self.optimiser.state_dict(),
                                 'loss': loss,
                                 'auroc_best': auroc_val,
                                 'auroc_last': best_val_auroc,
                                 }, "last")

    def val(self):
        print("\nValidating...")
        self.model.eval()
        xs = []

        with torch.no_grad():
            for batch_size, n_id, adjs in self.val_loader:
                if self.model.feature_extractor.num_layers == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                out = self.model(self.x_all[n_id].to(self.device), adjs)
                # x  = F.relu(out)
                x = out
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

            # y_pred = torch.exp(x_all).permute(1, 0)[1]

            y_pred = x_all

        if self.args.use_train_split:
            train_pred, val_pred = y_pred[self.split['train']], y_pred[self.split['val']]
            aurocs = self.evaluator.validate((train_pred, val_pred))
        else:
            aurocs = self.evaluator.validate(y_pred)

        self.model.train()

        return aurocs

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

    def single_pass(self, model, x_all, loader, single_layer):
        _, n_id, adjs = next(loader)

        if single_layer:
            adjs = [adjs]

        adjs = [adj.to(self.device) for adj in adjs]

        ebds = model(x_all[n_id].to(self.device), adjs)

        return ebds
