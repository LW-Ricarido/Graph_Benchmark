from abc import ABCMeta, abstractmethod

class LossFactory(metaclass=ABCMeta):
    """Abstract class for Loss factory"""
    registry = {}
    """Internal registry of Loss factories"""

    @classmethod
    def register(cls, name):
        """Decorator for registering Loss factories"""

        def decorator(wrapped_class):
            if name in cls.registry:
                raise ValueError('Cannot register duplicate loss factory ({})'.format(name))
            cls.registry[name] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def create_loss(cls, name, loss_config):
        """Create an loss"""
        if name not in cls.registry:
            raise ValueError('Cannot find loss factory ({})'.format(name))
        return cls.registry[name](**loss_config)