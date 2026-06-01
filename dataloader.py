import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from os.path import join
from config import device


class BasicDataset(Dataset):
    def __init__(self):
        pass

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def _split_A_hat(self, A):
        from config import args
        folds = getattr(args, 'A_n_fold', 100)

        A_fold = []
        fold_len = (self.n_users + self.m_items) // folds
        for i_fold in range(folds):
            start = i_fold * fold_len
            if i_fold == folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(device))
        return A_fold

    def getSparseGraph(self):
        raise NotImplementedError


class Loader(BasicDataset):

    def __init__(self, path="../data/Beibei", behavior="pv"):
        self.path = path
        self.behavior = behavior

        count_file = os.path.join(path, 'count.txt')
        try:
            with open(count_file, 'r') as f:
                count_data = json.load(f)
                self.n_user = count_data['user']
                self.m_item = count_data['item']
        except Exception as e:
            raise ValueError(f"Failed to read count.txt from {count_file}: {e}")

        train_file = os.path.join(path, f'{behavior}.txt')
        if not os.path.exists(train_file):
            raise ValueError(f"Behavior file {train_file} does not exist")

        trainUser, trainItem = [], []
        self.traindataSize = 0

        print(f"Loading behavior data from: {train_file}")
        with open(train_file) as f:
            for line in f.readlines():
                if len(line.strip()) > 0:
                    items = line.strip().split(' ')
                    uid = int(items[0])
                    items = [int(i) for i in items[1:]]

                    if uid >= self.n_user:
                        print(f"Warning: User ID {uid} exceeds max user count {self.n_user}")
                        continue

                    valid_items = []
                    for item in items:
                        if item >= self.m_item:
                            print(f"Warning: Item ID {item} exceeds max item count {self.m_item}")
                        else:
                            valid_items.append(item)

                    if valid_items:
                        trainUser.extend([uid] * len(valid_items))
                        trainItem.extend(valid_items)
                        self.traindataSize += len(valid_items)

        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.Graph = None

        print(f"Loaded {self.traindataSize} interactions for behavior '{behavior}'")
        print(f"Users: {self.n_user}, Items: {self.m_item}")
        print(f"Actual max user ID: {self.trainUser.max() if len(self.trainUser) > 0 else 'N/A'}")
        print(f"Actual max item ID: {self.trainItem.max() if len(self.trainItem) > 0 else 'N/A'}")

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)),
                                       (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        if self.Graph is None:
            cache_file = os.path.join(self.path, f's_pre_adj_mat_{self.behavior}.npz')
            try:
                pre_adj_mat = sp.load_npz(cache_file)
                norm_adj = pre_adj_mat
                print(f"Loaded cached adjacency matrix for behavior '{self.behavior}'")
            except:
                print(f"Generating adjacency matrix for behavior '{self.behavior}' from scratch...")
                start_time = time()

                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()

                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                end_time = time()
                print(f"Adjacency matrix generation completed in {end_time - start_time:.2f}s")

                try:
                    sp.save_npz(cache_file, norm_adj)
                    print(f"Saved adjacency matrix cache for behavior '{self.behavior}'")
                except:
                    print("Warning: Could not save adjacency matrix")

            from config import args
            if getattr(args, 'A_split', False):
                self.Graph = self._split_A_hat(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(device)

        return self.Graph

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


def load_dataset(dataset_name, data_path="../data", behavior="pv"):
    if dataset_name in ['Beibei', 'IJCAI_15', 'Tmall']:
        return Loader(path=os.path.join(data_path, dataset_name), behavior=behavior)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: ['Beibei', 'IJCAI_15', 'Tmall']")


def get_available_behaviors(dataset_name, data_path="../data"):
    dataset_path = os.path.join(data_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist")

    behaviors = []
    for file in os.listdir(dataset_path):
        if file.endswith('.txt') and file != 'count.txt':
            behavior = file[:-4]
            behaviors.append(behavior)

    return sorted(behaviors)