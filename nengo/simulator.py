"""
simulator.py: Simple reference simulator for base.Model

"""

import numpy as np

def get_signal(signals_dct, obj):
    # look up a Signal or SignalView
    # in a `signals_dct` such as self.signals
    if obj in signals_dct:
        return signals_dct[obj]
    elif obj.base in signals_dct:
        base_array = signals_dct[obj.base]
        try:
            # wtf?
            itemsize = int(obj.dtype.itemsize)
        except TypeError:
            itemsize = int(obj.dtype().itemsize)
        byteoffset = itemsize * obj.offset
        bytestrides = [itemsize * s for s in obj.elemstrides]
        view = np.ndarray(shape=obj.shape,
                          dtype=obj.dtype,
                          buffer=base_array.data,
                          offset=byteoffset,
                          strides=bytestrides,
                         )
        view[...]
        return view
    else:
        raise TypeError()


def zero_array_dct(dct):
    for arr in dct.values():
        arr[...] = 0


class Simulator(object):
    def __init__(self, model):
        self.model = model

        self.n_steps = 0
        self.signals = {}
        self.signals_tmp = {}
        self.signals_copy = {}

        for sig in self.model.signals:
            if hasattr(sig, 'value'):
                self.signals[sig] = np.asarray(sig.value)
            else:
                self.signals[sig] = np.zeros(sig.n)
            self.signals_tmp[sig] = np.zeros(sig.n)
            self.signals_copy[sig] = np.zeros(sig.n)

    def step(self):
        zero_array_dct(self.signals_tmp)

        # -- copy: signals -> signals_copy
        for sig in self.model.signals:
            self.signals_copy[sig][...] = self.signals[sig]

        # -- filters: signals_copy -> signals
        zero_array_dct(self.signals)
        for filt in self.model.filters:
            new, old = filt.newsig, filt.oldsig
            inc =  np.dot(filt.alpha, get_signal(self.signals_copy, old))
            targ = get_signal(self.signals, new)
            # -- we check for size mismatch,
            #    because incrementing scalar to len-1 arrays is ok
            #    if the shapes are not compatible, we'll get a
            #    problem in targ[...] += inc
            if inc.shape != targ.shape:
                if inc.size == targ.size == 1:
                    inc = np.asarray(inc).reshape(targ.shape)
                else:
                    raise ValueError('shape mismatch in filter',
                        (filt, inc.shape, targ.shape))
            targ[...] += inc

        # -- transforms: signals_tmp -> signals
        for tf in self.model.transforms:
            get_signal(self.signals, tf.outsig)[...] += np.dot(
                tf.alpha,
                get_signal(self.signals_tmp, tf.insig))

        self.n_steps += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()
            if verbose:
                print self.signals
