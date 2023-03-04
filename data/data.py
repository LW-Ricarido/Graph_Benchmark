from abc import ABCMeta, abstractmethod
import torch.utils.data

class DatasetFactory(metaclass=ABCMeta):
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
    def create_dataset(cls, name, dataset_config):
        """Create a dataset"""
        if name not in cls.registry:
            raise ValueError('Cannot find dataset factory ({})'.format(name))
        print('==========fucking name', name)
        print(dataset_config)
        return cls.registry[name](**dataset_config)

class BaseDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    @abstractmethod
    def train_data(self):
        pass
    
    @abstractmethod
    def train_label(self):
        pass

    @abstractmethod
    def val_data(self):
        pass

    @abstractmethod
    def val_label(self):
        pass

    @abstractmethod
    def test_data(self):
        pass

    @abstractmethod
    def test_label(self):
        pass
