import numpy as np


class Signal(object):
    def __init__(self, n=1):
        self.n = n


class Constant(Signal):
    def __init__(self, n, value):
        Signal.__init__(self, n)
        self.value = value


class Transform(object):
    def __init__(self, alpha, insig, outsig):
        self.alpha = alpha
        self.insig = insig
        self.outsig = outsig


class CustomTransform(object):
    def __init__(self, func, insig, outsig):
        self.func = func
        self.insig = insig
        self.outsig = outsig


class Filter(object):
    def __init__(self, alpha, oldsig, newsig):
        self.oldsig = oldsig
        self.newsig = newsig
        self.alpha = alpha


class Model(object):
    def __init__(self, dt):
        self.dt = dt
        self.signals = []
        self.transforms = []
        self.filters = []
        self.custom_transforms = []

    def signal(self, n=1, value=None):
        if value is None:
            rval = Signal(n)
        else:
            rval = Constant(n, value)
        self.signals.append(rval)
        return rval

    def transform(self, alpha, insig, outsig):
        rval = Transform(alpha, insig, outsig)
        self.transforms.append(rval)
        return rval

    def filter(self, alpha, oldsig, newsig):
        rval = Filter(alpha, oldsig, newsig)
        self.filters.append(rval)
        return rval

    def custom_transform(self, func, insig, outsig):
        rval = CustomTransform(func, insig, outsig)
        self.custom_transforms.append(rval)
        return rval
