import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

import dgl
from scipy import sparse as sp
from .data import DatasetFactory, BaseDataset
import logging
import os
from ogb.nodeproppred import DglNodePropPredDataset

@DatasetFactory.register('Ogb_products')
class Ogb_productsDataset(BaseDataset):

    def __init__(self, **kwargs):
        self.name = "Ogb_products"
        self.path = kwargs['path']
        self.device = kwargs['device']
        self.self_loop = kwargs['self_loop']

        self.g, self.labels = None, None
        self.train_mask, self.val_mask, self.test_mask = None, None, None
        self.num_classes, self.n_feats = None, None
        
        self._load()
    
    def _load(self):
        dataset = DglNodePropPredDataset(name='ogbn-products', root=self.path)
        self.g, self.labels = dataset[0]
        if self.self_loop:
            self.g = dgl.to_simple(dgl.add_self_loop(self.g))
        self.g = self.g.to(self.device)
        self.labels = self.labels.squeeze().to(self.device)
        self.g.ndata['label'] = self.labels

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        self.train_mask = torch.zeros(self.g.number_of_nodes(), dtype=torch.bool).to(self.device)
        self.train_mask[train_idx] = True
        self.val_mask = torch.zeros(self.g.number_of_nodes(), dtype=torch.bool).to(self.device)
        self.val_mask[valid_idx] = True
        self.test_mask = torch.zeros(self.g.number_of_nodes(), dtype=torch.bool).to(self.device)
        self.test_mask[test_idx] = True
    
        
        self.n_feats = self.g.ndata['feat'].shape[1]
        self.num_classes = len(set(self.g.ndata['label'].tolist()))

        self.g.ndata['feat'] = self.g.ndata['feat']
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1).to(self.device)
    
    def _add_positional_encodings(self, pos_enc_dim):
        g = self.g

        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float().to(self.device)
        
        self.g = g

    def train_data(self):
        return self.g, self.g.ndata['feat'], self.g.edata['feat']
    
    def train_label(self):
        return self.train_mask
    
    def val_data(self):
        return self.train_data()
    
    def val_label(self):
        return self.val_mask
    
    def test_data(self):
        return self.train_data()
    
    def test_label(self):
        return self.test_mask