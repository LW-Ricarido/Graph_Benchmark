import numpy as np
import os
import time
import random

import torch
import torch.optim as optim
import dgl

from tensorboardX import SummaryWriter
from tqdm import tqdm
from .runner import RunnerFactory
import os, sys
sys.path.append('..')
from data import DatasetFactory
from model import ModelFactory
from evaluator import EvaluatorFactory
from loss import LossFactory
from preprocessor import PreprocessorFactory
import logging

@RunnerFactory.register('NodeClassification')
class NodeClassifierRunner():

    def __init__(self, **kwargs) -> None:
        self.args = kwargs
        self._set_device()
        self._set_reproducibility()
        self._load_data()
        self._build_preprocessor()
        self._build_model()
        self._build_evaluator()
        self._set_loss_fn()
    
    def _set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.args['device'] = self.device
        self.args['net_params']['device'] = self.device
        self.args['dataset']['device'] = self.device
    
    def _set_reproducibility(self):
        if self.args['reproducibility']:
            random.seed(self.args['seed'])
            np.random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])
            torch.cuda.manual_seed(self.args['seed'])

    def _load_data(self):
        self.dataset = DatasetFactory.create_dataset(self.args['dataset']['name'], self.args['dataset'])
        self.args['net_params']['in_dim'] = self.dataset.n_feats
        self.args['net_params']['n_classes'] = self.dataset.num_classes

    def _build_preprocessor(self):
        if 'preprocessor' in self.args.keys():
            self.preprocessor = PreprocessorFactory.create_preprocessor(self.args['preprocessor']['name'], self.args['preprocessor'])
            self.preprocessor.preprocess(self.dataset)
        else:
            self.preprocessor = None

    def _build_model(self):
        self.model = ModelFactory.create_model(self.args['model'], self.args['net_params']).to(self.device)
        total_param = 0
        for param in self.model.parameters():
            total_param += np.prod((list(param.data.size())))
        logging.info( "Total number of parameters: {}".format(total_param))
    
    def _build_evaluator(self):
        self.evaluator = EvaluatorFactory.create_evaluator(self.args['evaluator_params']['name'], self.args['evaluator_params'])

    def _set_loss_fn(self):
        self.loss_fn = LossFactory.create_loss(self.args['loss_params']['name'], self.args['loss_params'])

    def run(self):
        test_accuracys, val_accuracys = [], []
        for round in range(self.args['test_rounds']):
            optimizer = optim.Adam(self.model.parameters(), lr=self.args['params']['init_lr'], weight_decay=self.args['params']['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=self.args['params']['lr_reduce_factor'], 
                                                         patience=self.args['params']['lr_schedule_patience'], 
                                                         verbose=True)
            self.model.reset_parameters()
            round_path = os.path.join(self.args['save_path'], "round{}".format(round))
            if not os.path.exists(round_path):
                os.makedirs(round_path)
            writer = SummaryWriter(round_path)
            try:
                with tqdm(range(self.args['params']['epochs']),position=0, leave=True) as t:
                    best_val_accuracy = 0
                    for epoch in t:
                        t.set_description('{} Round {}/{} Epoch {}'.format(self.args['task_name'], round, self.args['test_rounds'],epoch))
                        start = time.time()
                        epoch_loss, epoch_train_accuracy = self.evaluator.train(self.model, self.dataset,self.loss_fn) #TODO: evaluator get train loss
                        optimizer.zero_grad()
                        epoch_loss.backward()
                        optimizer.step()
                        epoch_val_loss, epoch_val_accuracy = self.evaluator.evaluate(self.model, self.dataset, self.loss_fn) #TODO: evaluator get val accuracy
                        scheduler.step(epoch_val_loss)
                        if epoch_val_accuracy > best_val_accuracy:
                            best_val_accuracy = epoch_val_accuracy
                            torch.save(self.model, round_path+'/best_model.pkl')
                        if epoch % self.args['log_interval'] == 0:
                            logging.info('Round:{}/{} Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Train Accuracy {:.4f} | Val Loss {:.4f} | Val Accuracy {:.4f}'.format(
                                round, self.args['test_rounds'], epoch, time.time() - start, epoch_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy))
                            writer.add_scalar('train_loss', epoch_loss, epoch)
                            writer.add_scalar('train_accuracy', epoch_train_accuracy, epoch)
                            writer.add_scalar('val_loss', epoch_val_loss, epoch)
                            writer.add_scalar('val_accuracy', epoch_val_accuracy, epoch)
                    test_model = torch.load(round_path+'/best_model.pkl')
                    test_loss, test_accuracy = self.evaluator.evaluate(test_model, self.dataset, self.loss_fn, val=False) #TODO: evaluator get test accuracy
                    writer.add_scalar('test_accuracy', test_accuracy, epoch)
                    writer.add_scalar('best_val_accuracy', best_val_accuracy, epoch)
                    logging.info('Round {} Test Accuracy {:.4f}'.format(round,test_accuracy))
                    logging.info('Round {} Best Val Accuracy {:.4f}'.format(round, best_val_accuracy))
                    writer.close()
                    #TODO: log test accuracy, TensorboardX log
                test_accuracys.append(test_accuracy)
                val_accuracys.append(best_val_accuracy)
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt")
                break
        logging.info("{} Average Val accuracy: {:.4f}".format(self.args['task_name'], np.mean(val_accuracys)))
        logging.info("{} Average Test accuracy: {:.4f}".format(self.args['task_name'], np.mean(test_accuracys)))