import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

import dgl
from scipy import sparse as sp
from .data import DatasetFactory, BaseDataset
import logging
import os


@DatasetFactory.register('Cora')
class CoraDataset(BaseDataset):
    
    def __init__(self, **kwargs):
        self.name = "Cora"
        self.data = dgl.data.CoraGraphDataset()
        
        self.g, self.labels = None, None
        self.train_mask, self.val_mask, self.test_mask = None, None, None
        self.num_classes, self.n_feats = None, None
        
        self._load()
    
    def _load(self):
        self.g = self.data[0]
        self.labels = self.g.ndata['label']
        self.train_mask = self.g.ndata['train_mask']
        self.val_mask = self.g.ndata['val_mask']

        self.test_mask = self.g.ndata['test_mask']
        
        self.n_feats = self.g.ndata['feat'].shape[1]
        self.num_classes = len(set(self.g.ndata['label'].tolist()))
        self.g.ndata['feat'] = self.g.ndata['feat']
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)
    
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
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
        
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

@DatasetFactory.register('CoraFeature')
class CoraFeatureDataset(BaseDataset):

    def __init__(self,**kwargs) -> None:
        self.path = kwargs['path']
        self.max_similarity = kwargs['max_similarity']
        self._load()

    def _load(self):
        data = dgl.data.CoraGraphDataset()
        og = data[0]
        self.labels = og.ndata['label']
        self.train_mask = og.ndata['train_mask']
        self.val_mask = og.ndata['val_mask']
        self.test_mask = og.ndata['test_mask']
        self.n_feats = og.ndata['feat'].shape[1]
        self.num_classes = len(set(og.ndata['label'].tolist()))
        og.ndata['feat'] = og.ndata['feat']
        og.edata['feat'] = torch.zeros(og.number_of_edges(), 1)
        if os.listdir(self.path) == []:
            og = data[0]
            tmp_feats = og.ndata['feat']
            adj_matrix = torch.zeros(og.num_nodes(), og.num_nodes())

            cs = nn.CosineSimilarity()
            for i in range(og.num_nodes()):
                for j in range(og.num_nodes()):
                    adj_matrix[i] = cs(tmp_feats[i], tmp_feats)
            torch.save(adj_matrix, self.path + '/adj_matrix.pt')
        else:
            adj_matrix = torch.load(self.path + '/adj_matrix.pt')
        adj_matrix_b = (adj_matrix > self.max_similarity)
        u = torch.nonzero(adj_matrix_b)[:, 0]
        v = torch.nonzero(adj_matrix_b)[:, 1]
        logging.log(logging.INFO,"=================graph edge num:{}".format(len(u)))
        self.g = dgl.graph((u, v), num_nodes=adj_matrix.shape[0])
        self.g.ndata['feat'] = og.ndata['feat']
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)
    
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
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
        
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