from .preprocessor import PreprocessorFactory, BasePreprocessor
import math
import torch
import torch.nn as nn
import logging
@PreprocessorFactory.register('DegreeBaseRewiring')
class DegreeBaseRewiringPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.self_loop = kwargs['self_loop']

    def preprocess(self, dataset):
        average_degree = math.ceil(dataset.g.number_of_edges() / dataset.g.number_of_nodes())
        add_edges = set()
        ndata = dataset.g.ndata['feat']
        cs = nn.CosineSimilarity()
        for i in range(dataset.g.number_of_nodes()):
            current_degree = dataset.g.successors(i).size()[0]
            if current_degree < average_degree:
                similarity = cs(ndata[i], ndata)
                top_similarities = torch.topk(similarity, average_degree * 2 + 1).indices
                for j in top_similarities:
                    if not self.self_loop and i == j:
                        continue
                    if dataset.g.has_edges_between(i, j):
                        continue
                    add_edges.add((i, j))
                    add_edges.add((j, i))
                    current_degree += 1
                    if current_degree >= average_degree:
                        break
        logging.info('=================before add edge num:{}'.format(dataset.g.number_of_edges()))
        logging.info('=================graph edge add num:{}'.format(len(add_edges)))
        dataset.g.add_edges(*zip(*add_edges))
        logging.info('=================after add edge num:{}'.format(dataset.g.number_of_edges()))
        

