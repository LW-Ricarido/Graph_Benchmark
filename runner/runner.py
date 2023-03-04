from abc import ABCMeta


class RunnerFactory(metaclass=ABCMeta):
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
    def create_runner(cls, name, runner_config):
        """Create a dataset"""
        if name not in cls.registry:
            raise ValueError('Cannot find dataset factory ({})'.format(name))
        return cls.registry[name](**runner_config)