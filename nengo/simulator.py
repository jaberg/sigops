"""
simulator.py: Simple reference simulator for base.Model

"""

import numpy as np

class Simulator(object):
    def __init__(self, model):
        self.model = model

        self.n_steps = 0
        self.signals = {}
        self.signals_tmp = {}
        self.signals_copy = {}
        self.probe_outputs = {}

        for sig in self.model.signals:
            if hasattr(sig, 'value'):
                self.signals[sig] = np.asarray(sig.value)
            else:
                self.signals[sig] = np.zeros(sig.n)
            self.signals_tmp[sig] = np.zeros(sig.n)
            self.signals_copy[sig] = np.zeros(sig.n)

        for probe in self.model.signal_probes:
            self.probe_outputs[probe] = []

    def step(self):
        # -- copy: signals -> signals_copy
        for sig in self.model.signals:
            self.signals_copy[sig] = 1.0 * self.signals[sig]

        # -- reset: 0 -> signals
        for sig in self.model.signals:
            self.signals[sig][...] = 0

        # -- filters: signals_copy -> signals
        for filt in self.model.filters:
            new, old = filt.newsig, filt.oldsig
            self.signals[new] += filt.alpha * self.signals_copy[old]

        # -- transforms: signals_tmp -> signals
        for tf in self.model.transforms:
            self.signals[tf.outsig] += tf.alpha * self.signals_tmp[tf.insig]

        # -- customs: signals -> signals
        for ct in self.model.custom_transforms:
            self.signals[ct.outsig][...] = ct.func(self.signals[ct.insig])

        # -- probes signals -> probe buffers
        for probe in self.model.signal_probes:
            period = int(probe.dt / self.model.dt)
            if self.n_steps % period == 0:
                tmp = self.signals[probe.sig].copy()
                self.probe_outputs[probe].append(tmp)

        self.n_steps += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()
            if verbose:
                print self.signals

    def signal_probe_output(self, probe):
        return self.probe_outputs[probe]
