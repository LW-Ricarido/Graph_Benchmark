from .loss import LossFactory
import torch
import torch.nn as nn

@LossFactory.register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)

@LossFactory.register('NLLLoss')
class NLLLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.loss_fn = nn.NLLLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)