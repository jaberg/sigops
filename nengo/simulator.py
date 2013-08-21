import logging

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


def dot_inc(a, b, targ):
    # -- we check for size mismatch,
    #    because incrementing scalar to len-1 arrays is ok
    #    if the shapes are not compatible, we'll get a
    #    problem in targ[...] += inc
    try:
        inc =  np.dot(a, b)
    except Exception, e:
        e.args = e.args + (a.shape, b.shape)
        raise
    if inc.shape != targ.shape:
        if inc.size == targ.size == 1:
            inc = np.asarray(inc).reshape(targ.shape)
        else:
            raise ValueError('shape mismatch', (inc.shape, targ.shape))
    targ[...] += inc


class Simulator(object):
    def __init__(self, model):
        self.model = model

        self.n_steps = 0
        self.signals = {}
        self.signals_tmp = {}
        self.signals_copy = {}
        self.probe_outputs = {}
        self.constant_signals = []
        self.dynamic_signals = []

        for sig in self.model.signals:
            if hasattr(sig, 'value'):
                self.signals[sig] = np.asarray(sig.value)
                self.signals_tmp[sig] = np.asarray(sig.value)
                self.signals_copy[sig] = np.asarray(sig.value)
                self.constant_signals.append(sig)
            else:
                self.signals[sig] = np.zeros(sig.n)
                self.signals_tmp[sig] = np.zeros(sig.n)
                self.signals_copy[sig] = np.zeros(sig.n)
                self.dynamic_signals.append(sig)

        for probe in self.model.probes:
            self.probe_outputs[probe] = []

    def step(self):
        # -- reset: 0 -> signals_tmp
        for sig in self.dynamic_signals:
            self.signals_tmp[sig][...] = 0

        # -- copy: signals -> signals_copy
        for sig in self.dynamic_signals:
            self.signals_copy[sig][...] = self.signals[sig]

        # -- reset: 0 -> signals
        for sig in self.dynamic_signals:
            self.signals[sig][...] = 0

        # -- filters: signals_copy -> signals
        for filt in self.model.filters:
            #print
            # print 'old sig: ', filt.oldsig.name, get_signal(self.signals_copy, filt.oldsig)
            try:
                dot_inc(filt.alpha,
                        get_signal(self.signals_copy, filt.oldsig),
                        get_signal(self.signals, filt.newsig))
            except Exception, e:
                e.args = e.args + (filt.oldsig, filt.newsig)
                raise
            # print 'new sig: ', filt.newsig.name, get_signal(self.signals, filt.newsig)

        # -- transforms: signals_tmp -> signals
        for tf in self.model.transforms:
            #print
            #print 'old sig: ', tf.insig.name, get_signal(self.signals_copy, tf.insig)
            dot_inc(tf.alpha,
                    get_signal(self.signals_tmp, tf.insig),
                    get_signal(self.signals, tf.outsig))
            #print 'new sig: ', tf.outsig.name, get_signal(self.signals, tf.outsig)
        #print 'POST'
        #for k, v in self.signals.items():
        #    print k, v

        # -- probes signals -> probe buffers
        for probe in self.model.probes:
            period = int(probe.dt / self.model.dt)
            if self.n_steps % period == 0:
                tmp = get_signal(self.signals, probe.sig).copy()
                self.probe_outputs[probe].append(tmp)

        self.n_steps += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()
            if verbose:
                print self.signals

    def probe_data(self, probe):
        return np.asarray(self.probe_outputs[probe])
