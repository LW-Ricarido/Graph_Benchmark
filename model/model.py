from abc import ABCMeta, abstractmethod
import torch.nn as nn

class ModelFactory(metaclass=ABCMeta):
    """Abstract class for dataset factory"""
    registry = {}
    """Internal registry of dataset factories"""

    @classmethod
    def register(cls, name):
        """Decorator for registering dataset factories"""

        def decorator(wrapped_class):
            if name in cls.registry:
                raise ValueError('Cannot register duplicate dataset factory ({})'.format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def create_model(cls, name, model_config):
        """Create a dataset"""
        if name not in cls.registry:
            raise ValueError('Cannot find dataset factory ({})'.format(name))
        return cls.registry[name](**model_config)

class BaseModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def reset_parameters(self):
        pass