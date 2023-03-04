import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

class BaseLayer(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def reset_parameters(self):
        pass