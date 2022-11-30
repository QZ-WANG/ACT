import numpy as np
from os.path import join
import scipy.io as sio
import scipy.sparse as sp

import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import yaml


def load_mat(file_dir, fn):
    fp = join(file_dir, fn)
    data = sio.loadmat(fp)
    return {
        "features": sp.lil_matrix(data['Attributes']),
        "adj": sp.csr_matrix(data['Network']),
        "ad_labels": np.squeeze(np.array(data['Label']))
    }


def mat_to_pyg_data(data, undirected=True):
    features = torch.from_numpy(data["features"].todense()).float()

    adj = data["adj"]
    edge_index, _ = from_scipy_sparse_matrix(adj)

    ad_labels = data['ad_labels']

    if undirected:
        print("Processing the graph as undirected...")
        if data.is_directed():
            edge_index = to_undirected(data.edge_index)

    data = Data(x=features, edge_index=edge_index)

    if undirected:
        assert data.is_undirected()

    return data, ad_labels


def train_test_split(ad_labels, train_size, test_size):
    inlier_idx, outlier_idx = np.where(ad_labels == 0)[0], np.where(ad_labels == 1)[0]
    n_inlier, n_outlier = inlier_idx.shape[0], outlier_idx.shape[0]
    print('num inliers: %d, num outliers %d' % (n_inlier, n_outlier))

    n_inlier_train, n_inlier_test = int(n_inlier * train_size), int(n_inlier * test_size)

    n_outlier_train, n_outlier_test = int(n_outlier * train_size), int(n_outlier * test_size)

    np.random.shuffle(inlier_idx)
    np.random.shuffle(outlier_idx)

    inlier_train_idx = np.sort(inlier_idx[:n_inlier_train])
    outlier_train_idx= np.sort(outlier_idx[:n_outlier_train])

    test_idx = np.sort(np.hstack((inlier_idx[n_inlier_train : n_inlier_train + n_inlier_test],
                         outlier_idx[n_outlier_train : n_outlier_train + n_outlier_test])))

    val_idx = np.sort(np.hstack((inlier_idx[n_inlier_train + n_inlier_test:n_inlier],
                         outlier_idx[n_outlier_train + n_outlier_test:n_outlier])))

    print("num train inliers: %4d   num train outliers %4d" % (inlier_train_idx.shape[0], outlier_train_idx.shape[0]))
    print("num val nodes: %4d" % val_idx.shape[0])
    return (inlier_train_idx, outlier_train_idx), test_idx, val_idx


def load_yaml(fn):
    with open(fn) as fp:
        config = yaml.safe_load(fp)
    return config