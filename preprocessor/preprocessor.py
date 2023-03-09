from abc import ABCMeta, abstractmethod

class PreprocessorFactory(metaclass=ABCMeta):
    """Abstract class for preprocessor factory"""
    registry = {}
    """Internal registry of preprocessor factories"""

    @classmethod
    def register(cls, name):
        """Decorator for registering preprocessor factories"""

        def decorator(wrapped_class):
            if name in cls.registry:
                raise ValueError('Cannot register duplicate preprocessor factory ({})'.format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def create_preprocessor(cls, name, preprocessor_config):
        """Create a preprocessor"""
        if name not in cls.registry:
            raise ValueError('Cannot find preprocessor factory ({})'.format(name))
        return cls.registry[name](**preprocessor_config)

class BasePreprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, dataset):
        pass