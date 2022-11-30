from sklearn.metrics import roc_auc_score, average_precision_score
import torch


def get_auroc_aupr(scores, ad_labels):
    if isinstance(scores, torch.Tensor):
        if scores.requires_grad:
            scores = scores.detach()

        if 'cuda' in str(scores.device):
            scores = scores.cpu()

        if 'cuda' in str(ad_labels.device):
            ad_labels = ad_labels.cpu()

    return roc_auc_score(ad_labels, scores), average_precision_score(ad_labels, scores)


class SupEvaluator:
    def __init__(self, split, labels):
        self.whole_graph = False
        self.split = split
        if split['test'] is None and split['val'] is None:
            self.all_labels = labels
            self.whole_graph = True
        else:
            self.y_train, self.y_test, self.y_val = labels[split['train']], labels[split['test']], labels[split['val']]

    def validate(self, pred):
        if self.whole_graph:
            return roc_auc_score(self.all_labels.cpu(), pred)
        else:
            train_pred, val_pred = pred
            train_auroc, val_auroc = roc_auc_score(self.y_train.cpu(), train_pred), \
                                     roc_auc_score(self.y_val.cpu(), val_pred)
            train_aupr, val_aupr = average_precision_score(self.y_train.cpu(), train_pred), \
                                   average_precision_score(self.y_val.cpu(), val_pred)
            print("train_auroc:", train_auroc, "val_auroc: ", val_auroc, "train aupr: ", train_aupr,
                  "val_aupr: ", val_aupr)
            return train_auroc, val_auroc

