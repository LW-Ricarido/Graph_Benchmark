import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

import dgl
from scipy import sparse as sp
from .data import DatasetFactory, BaseDataset
import logging
import os

@DatasetFactory.register('Flickr')
class FlickrDataset(BaseDataset):

    def __init__(self, **kwargs):
        self.name = "Flickr"
        self.data = dgl.data.FlickrDataset()
        self.device = kwargs['device']
        self.self_loop = kwargs['self_loop']

        self.g, self.labels = None, None
        self.train_mask, self.val_mask, self.test_mask = None, None, None
        self.num_classes, self.n_feats = None, None
        
        self._load()
    
    def _load(self):
        self.g = self.data[0]
        if self.self_loop:
            self.g = dgl.to_simple(dgl.add_self_loop(self.g))
        self.g = self.g.to(self.device)

        self.labels = self.g.ndata['label']
        self.train_mask = self.g.ndata['train_mask']
        self.val_mask = self.g.ndata['val_mask']

        self.test_mask = self.g.ndata['test_mask']
        
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
    
@DatasetFactory.register('FlickrFeature')
class FlickrFeatureDataset(FlickrDataset):

    def __init__(self, **kwrags):
        self.path = kwrags['path']
        self.device = kwrags['device']
        self.max_similarity = kwrags['max_similarity']
        self.self_loop = kwrags['self_loop']
        self._load()
    
    def _load(self):
        data = dgl.data.FlickrDataset()
        og = data[0].to(self.device)
        self.labels = og.ndata['label']
        self.train_mask = og.ndata['train_mask']
        self.val_mask = og.ndata['val_mask']
        self.test_mask = og.ndata['test_mask']
        self.n_feats = og.ndata['feat'].shape[1]
        self.num_classes = len(set(og.ndata['label'].tolist()))
        og.ndata['feat'] = og.ndata['feat']
        og.edata['feat'] = torch.zeros(og.number_of_edges(), 1).to(self.device)
        tmp_feats = og.ndata['feat']
        cs = nn.CosineSimilarity()
        u  = []
        v = []
        for i in range(og.num_nodes()):

            similarity = cs(tmp_feats[i], tmp_feats)
            u += [i] * torch.nonzero(similarity > self.max_similarity).shape[0]
            v += torch.nonzero(similarity > self.max_similarity)[:,0].tolist()
            
        
        self.g = dgl.graph((u, v), num_nodes=og.num_nodes())
        
        if self.self_loop:
            self.g = dgl.to_simple(dgl.add_self_loop(self.g))
        self.g = self.g.to(self.device)
        
        logging.log(logging.INFO,"=================graph edge num:{}".format(len(u)))
        self.g.ndata['feat'] = og.ndata['feat']
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1).to(self.device)
    