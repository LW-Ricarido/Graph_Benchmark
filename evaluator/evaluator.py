from abc import ABCMeta, abstractmethod

class EvaluatorFactory(metaclass=ABCMeta):
    """Abstract class for evaluator factory"""
    registry = {}
    """Internal registry of evaluator factories"""

    @classmethod
    def register(cls, name):
        """Decorator for registering evaluator factories"""

        def decorator(wrapped_class):
            if name in cls.registry:
                raise ValueError('Cannot register duplicate evaluator factory ({})'.format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def create_evaluator(cls, name, evaluator_config):
        """Create an evaluator"""
        if name not in cls.registry:
            raise ValueError('Cannot find evaluator factory ({})'.format(name))
        return cls.registry[name](**evaluator_config)


class BaseEvaluator(metaclass=ABCMeta):

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def evaluate():
        pass