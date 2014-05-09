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
            self.signals[sig] = np.zeros(sig.n)
            self.signals_tmp[sig] = np.zeros(sig.n)
            self.signals_copy[sig] = np.zeros(sig.n)

        for probe in self.model.signal_probes:
            self.probe_outputs[probe] = []

    def step(self):
        for sig in self.model.signals:
            self.signals_copy[sig] = self.signals[sig]

        for filt in self.model.filters:
            self.signals[filt.newsig] = \
                    filt.alpha * self.signals_copy[filt.oldsig]

        for tf in self.model.transforms:
            self.signals[tf.outsig] += tf.alpha * self.signals_tmp[tf.insig]

        for ct in self.model.custom_transforms:
            self.signals[ct.outsig] = ct.func(self.signals[ct.insig])

        for probe in self.model.signal_probes:
            period = int(probe.dt / self.model.dt)
            if self.n_steps % period == 0:
                self.probe_outputs[probe].append(self.signals[probe.sig])

        self.n_steps += 1

    def run_steps(self, N):
        for i in xrange(N):
            self.step()

    def signal_probe_output(self, probe):
        return self.probe_outputs[probe]
