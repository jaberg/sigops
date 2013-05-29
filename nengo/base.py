"""
base.py: model description classes

These classes are used to describe a nengo model (Model).
Model is the input to a *simulator* (see e.g. simulator.py).

"""
import numpy as np

random_weight_rng = np.random.RandomState(12345)


class Signal(object):
    """Interpretable, vector-valued quantity within NEF"""
    def __init__(self, n=1):
        self.n = n


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, n, value):
        Signal.__init__(self, n)
        self.value = value


class Transform(object):
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        self.alpha = alpha
        self.insig = insig
        self.outsig = outsig


class CustomTransform(object):
    """An arbitrary transform from a decoded signal to the signals buffer"""
    def __init__(self, func, insig, outsig):
        self.func = func
        self.insig = insig
        self.outsig = outsig


class Filter(object):
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        self.oldsig = oldsig
        self.newsig = newsig
        self.alpha = alpha


class Model(object):
    """
    A container for model components.
    """
    def __init__(self, dt):
        self.dt = dt
        self.signals = []
        self.transforms = []
        self.filters = []
        self.custom_transforms = []

    def signal(self, n=1, value=None):
        """Add a signal to the model"""
        if value is None:
            rval = Signal(n)
        else:
            rval = Constant(n, value)
        self.signals.append(rval)
        return rval

    def transform(self, alpha, insig, outsig):
        """Add a transform to the model"""
        rval = Transform(alpha, insig, outsig)
        self.transforms.append(rval)
        return rval

    def filter(self, alpha, oldsig, newsig):
        """Add a filter to the model"""
        rval = Filter(alpha, oldsig, newsig)
        self.filters.append(rval)
        return rval

    def custom_transform(self, func, insig, outsig):
        """Add a custom transform to the model"""
        rval = CustomTransform(func, insig, outsig)
        self.custom_transforms.append(rval)
        return rval
