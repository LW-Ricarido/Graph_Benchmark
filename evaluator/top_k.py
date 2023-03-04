from .evaluator import EvaluatorFactory, BaseEvaluator
import torch

def top_k(logits, labels, k):
    _, indices = torch.topk(logits, k)
    correct = torch.sum(labels.unsqueeze(1) == indices)
    return correct.item() * 1.0 / len(labels)

@EvaluatorFactory.register('Top_K_Evaluator')
class Top_K_Evaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        self.k = kwargs['k']

    def train(self,model, dataset, loss_fn):
        
        logits = model(*dataset.train_data())
        train_mask = dataset.train_mask
        labels = dataset.labels
        loss = loss_fn(logits[train_mask], labels[train_mask])
        accuracy = top_k(logits[train_mask], labels[train_mask], self.k)
        return loss, accuracy

    def evaluate(self, model, dataset, loss_fn,val=True):
        model.eval()
        with torch.no_grad():
            
            labels = dataset.labels
            if val:
                mask = dataset.val_mask
                logits = model(*dataset.val_data())
            else:
                mask = dataset.test_mask
                logits = model(*dataset.test_data())
            loss = loss_fn(logits[mask], labels[mask])
            accuracy = top_k(logits[mask], labels[mask], self.k)
            return loss, accuracy