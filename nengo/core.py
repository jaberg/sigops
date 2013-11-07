"""
Low-level objects
=================

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.

"""
import copy
import logging

import numpy as np
import simulator as sim


logger = logging.getLogger(__name__)


"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


def filter_coefs(pstc, dt):
    pstc = max(pstc, dt)
    decay = np.exp(-dt / pstc)
    return decay, (1.0 - decay)


class ShapeMismatch(ValueError):
    pass


class TODO(NotImplementedError):
    """Potentially easy NotImplementedError"""
    pass


class SignalView(object):
    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert base
        self.base = base
        self.shape = tuple(shape)
        self.elemstrides = tuple(elemstrides)
        self.offset = int(offset)
        if name is not None:
            self._name = name

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    def __repr__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    def view_like_self_of(self, newbase, name=None):
        if newbase.base != newbase:
            raise NotImplementedError()
        if newbase.structure != self.base.structure:
            raise NotImplementedError('technically ok but should not happen',
                                     (self.base, newbase))
        return SignalView(newbase,
                          self.shape,
                          self.elemstrides,
                          self.offset,
                          name)

    @property
    def structure(self):
        return (
            self.shape,
            self.elemstrides,
            self.offset)

    def same_view_as(self, other):
        return self.structure == other.structure \
           and self.base == other.base

    @property
    def dtype(self):
        return np.dtype(self.base._dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self.elemstrides == (1,):
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        else:
            # -- there are cases where reshaping can still work
            #    but there are limits too, because we can only
            #    support view-based reshapes. So the strides have
            #    to work.
            raise TODO('reshape of strided view')

    def transpose(self, neworder=None):
        raise TODO('transpose')

    def __getitem__(self, item):
        # -- copy the shape and strides
        shape = list(self.shape)
        elemstrides = list(self.elemstrides)
        offset = self.offset
        if isinstance(item, (list, tuple)):
            dims_to_del = []
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * elemstrides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * elemstrides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                elemstrides.pop(dim)
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, (int, np.integer)):
            if len(self.shape) == 0:
                raise IndexError()
            if not (0 <= item < self.shape[0]):
                raise NotImplementedError()
            shape = self.shape[1:]
            elemstrides = self.elemstrides[1:]
            offset = self.offset + item * self.elemstrides[0]
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, slice):
            return self.__getitem__((item,))
        else:
            raise NotImplementedError(item)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            if self.base is self:
                return '<anon%d>' % id(self)
            else:
                return 'View(%s[%d])' % (self.base.name, self.offset)

    @name.setter
    def name(self, value):
        self._name = value

    def add_to_model(self, model):
        if self.base not in model.signals:
            raise TypeError("Cannot add signal views. Add the signal instead.")

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'base': self.base.name,
            'shape': list(self.shape),
            'elemstrides': list(self.elemstrides),
            'offset': self.offset,
        }

    def shares_memory_with(self, other):
        # Terminology: two arrays *overlap* if the lowermost memory addressed
        # touched by upper one is higher than the uppermost memory address
        # touched by the lower one.
        #
        # np.may_share_memory returns True iff there is overlap.
        # Overlap is a necessary but insufficient condition for *aliasing*.
        #
        # Aliasing is when two ndarrays refer a common memory location.
        #
        if self.base is not other.base:
            return False
        if self is other or self.same_view_as(other):
            return True
        if self.ndim < other.ndim:
            return other.shares_memory_with(self)

        assert self.ndim >= other.ndim
        if self.ndim == 0:
            # self.same_view_as(other) would have
            # returned above if this were True.
            return False
        elif self.ndim == 1:
            raise NotImplementedError()
        elif self.ndim == 2:
            raise NotImplementedError()
        else:
            raise NotImplementedError()


class Signal(SignalView):
    """Interpretable, vector-valued quantity within NEF"""
    def __init__(self, n=1, dtype=np.float64, name=None):
        self.n = n
        self._dtype = dtype
        if name is not None:
            self._name = name
        if assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", " + str(self.n) + "D)"
        except AttributeError:
            return "Signal (id " + str(id(self)) + ", " + str(self.n) + "D)"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return (self.n,)

    @property
    def size(self):
        return self.n

    @property
    def elemstrides(self):
        return (1,)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self

    def add_to_model(self, model):
        model.signals.append(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'n': self.n,
            'dtype': str(self.dtype),
        }


class Probe(object):
    """A model probe to record a signal"""
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt

    def __str__(self):
        return "Probing " + str(self.sig)

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        model.probes.append(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'dt': self.dt,
        }


class Constant(Signal):
    """A signal meant to hold a fixed value"""
    def __init__(self, value, name=None):
        self.value = np.asarray(value)
        
        Signal.__init__(self, self.value.size, name=name)

    def __str__(self):
        if self.name is not None:
            return "Constant(" + self.name + ")"
        return "Constant(id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.value.shape

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'value': self.value.tolist(),
        }


def is_signal(sig):
    return isinstance(sig, SignalView)


def is_constant(sig):
    """
    Return True iff `sig` is (or is a view of) a Constant signal.
    """
    return isinstance(sig.base, Constant)

class Transform(object): #to be removed?
    """A linear transform from a decoded signal to the signals buffer"""
    def __init__(self, alpha, insig, outsig):
        alpha = np.asarray(alpha)
        if hasattr(outsig, 'value'):
            raise TypeError('transform destination is constant')
        if is_constant(insig):
            raise TypeError('constant input (use filter instead)')

        name = insig.name + ">" + outsig.name + ".tf_alpha"

        self.alpha_signal = Constant(alpha, name=name)
        self.insig = insig
        self.outsig = outsig
        if self.alpha_signal.size == 1:
            if self.insig.shape != self.outsig.shape:
                raise ShapeMismatch()
        else:
            if self.alpha_signal.shape != (
                    self.outsig.shape + self.insig.shape):
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.insig.shape,
                        self.outsig.shape,
                        )

    def __str__(self):
        return ("Transform (id " + str(id(self)) + ")"
                " from " + str(self.insig) + " to " + str(self.outsig))

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.append(self.alpha_signal)
        dst = model._get_output_view(self.outsig)

        # XXX: Complicated lookup still necessary?
        if self.insig in model._decoder_outputs:
            insig = model._decoder_outputs[self.insig]
        elif self.insig.base in model._decoder_outputs:
            insig = self.insig.view_like_self_of(
                model._decoder_outputs[self.insig.base])
        else:
            insig = self.insig

        model._operators.append(
            sim.DotInc(self.alpha_signal, insig, dst,
                       tag='transform'))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'alpha': self.alpha.tolist(),
            'insig': self.insig.name,
            'outsig': self.outsig.name,
        }


class Filter(object): #to be removed?
    """A linear transform from signals[t-1] to signals[t]"""
    def __init__(self, alpha, oldsig, newsig):
        if hasattr(newsig, 'value'):
            raise TypeError('filter destination is constant')
        alpha = np.asarray(alpha)

        name = oldsig.name + ">" + newsig.name + ".f_alpha"

        self.alpha_signal = Constant(alpha, name=name)
        self.oldsig = oldsig
        self.newsig = newsig

        if self.alpha_signal.size == 1:
            if self.oldsig.shape != self.newsig.shape:
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.oldsig.shape,
                        self.newsig.shape,
                        )
        else:
            if self.alpha_signal.shape != (
                    self.newsig.shape + self.oldsig.shape):
                raise ShapeMismatch(
                        self.alpha_signal.shape,
                        self.oldsig.shape,
                        self.newsig.shape,
                        )

    def __str__(self):
        return ("Filter (id " + str(id(self)) + ")"
                " from " + str(self.oldsig) + " to " + str(self.newsig))

    def __repr__(self):
        return str(self)

    @property
    def alpha(self):
        return self.alpha_signal.value

    @alpha.setter
    def alpha(self, value):
        self.alpha_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.append(self.alpha_signal)
        dst = model._get_output_view(self.newsig)

        model._operators.append(
            sim.DotInc(self.alpha_signal, self.oldsig, dst,
                       tag='transform'))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'alpha': self.alpha.tolist(),
            'oldsig': self.oldsig.name,
            'newsig': self.newsig.name,
        }


class Encoder(object): #to be removed?
    """A linear transform from a signal to a population"""
    def __init__(self, sig, pop, weights):
        self.sig = sig
        self.pop = pop
        weights = np.asarray(weights)
        if weights.shape != (pop.n_in, sig.size):
            raise ValueError('weight shape', weights.shape)
        name = sig.name + ".encoders"
        self.weights_signal = Constant(weights, name=name)

    def __str__(self):
        return ("Encoder (id " + str(id(self)) + ")"
                " of " + str(self.sig) + " to " + str(self.pop))

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.append(self.weights_signal)
        model._operators.append(
            sim.DotInc(
                self.sig,
                self.weights_signal,
                self.pop.input_signal,
                xT=True,
                tag='encoder'))


    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'sig': self.sig.name,
            'pop': self.pop.name,
            'weights': self.weights.tolist(),
        }


class Decoder(object): #to be removed?
    """A linear transform from a population to a signal"""
    def __init__(self, pop, sig, weights):
        self.pop = pop
        self.sig = sig
        weights = np.asarray(weights)
        if weights.shape != (sig.size, pop.n_out):
            raise ValueError('weight shape', weights.shape)
        name = sig.name + ".decoders"
        self.weights_signal = Constant(weights, name=name)

    def __str__(self):
        return ("Decoder (id " + str(id(self)) + ")"
                " of " + str(self.pop) + " to " + str(self.sig))

    def __repr__(self):
        return str(self)

    @property
    def weights(self):
        return self.weights_signal.value

    @weights.setter
    def weights(self, value):
        self.weights_signal.value[...] = value

    def add_to_model(self, model):
        model.signals.append(self.weights_signal)
        if self.sig.base not in model._decoder_outputs:
            sigbase = Signal(self.sig.base.n, name=self.sig.name + '-decbase')
            model.signals.append(sigbase)
            model._decoder_outputs[self.sig.base] = sigbase
            model._operators.append(
                sim.Reset(sigbase))
        else:
            sigbase = model._decoder_outputs[self.sig.base]
            
        if self.sig == self.sig.base:
            dec_sig = sigbase
        else:
            dec_sig = self.sig.view_like_self_of(sigbase)
            
        model._decoder_outputs[self.sig] = dec_sig
        model._operators.append(
            sim.DotInc(
                self.pop.output_signal,
                self.weights_signal,
                dec_sig,
                xT=True,
                tag='Decoder'))

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'pop': self.pop.name,
            'sig': self.sig.name,
            'weights': self.weights.tolist(),
        }


class Nonlinearity(object):
    def __str__(self):
        return "Nonlinearity (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def add_to_model(self, model):
        # XXX: do we still need to append signals to model?
        model.signals.append(self.bias_signal)
        model.signals.append(self.input_signal)
        model.signals.append(self.output_signal)
        model._operators.append(
            self.operator(
                output=self.output_signal,
                J=self.input_signal,
                nl=self))
        # -- encoders will be scheduled between this copy
        #    and nl_op
        model._operators.append(
            sim.Copy(dst=self.input_signal, src=self.bias_signal))


class Direct(Nonlinearity):

    operator = sim.SimDirect

    def __init__(self, n_in, n_out, fn, name=None):
        if name is None:
            name = "<Direct%d>" % id(self)
        self.name = name

        self.input_signal = Signal(n_in, name=name + '.input')
        self.output_signal = Signal(n_out, name=name + '.output')
        self.bias_signal = Constant(np.zeros(n_in),
                                    name=name + '.bias')

        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def __deepcopy__(self, memo):
        try:
            return memo[id(self)]
        except KeyError:
            rval = self.__class__.__new__(
                    self.__class__)
            memo[id(self)] = rval
            for k, v in self.__dict__.items():
                if k == 'fn':
                    rval.fn = v
                else:
                    rval.__dict__[k] = copy.deepcopy(v, memo)
            return rval

    def __str__(self):
        return "Direct (id " + str(id(self)) + ")"

    def __repr__(self):
        return str(self)

    def fn(self, J):
        return J

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'fn': self.fn.__name__,
        }


class _LIFBase(Nonlinearity):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, name=None):
        if name is None:
            name = "<%s%d>" % (self.__class__.__name__, id(self))
        self.input_signal = Signal(n_neurons, name=name + '.input')
        self.output_signal = Signal(n_neurons, name=name + '.output')
        self.bias_signal = Constant(np.zeros(n_neurons), name=name + '.bias')

        self.name = name
        self.n_neurons = n_neurons
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = None

    def __str__(self):
        return "%s (id %d, %dN)" % (
            self.__class__.__name__, id(self), self.n_neurons)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'input_signal': self.input_signal.name,
            'output_signal': self.output_signal.name,
            'bias_signal': self.bias_signal.name,
            'n_neurons': self.n_neurons,
            'tau_rc': self.tau_rc,
            'tau_ref': self.tau_ref,
            'gain': self.gain.tolist(),
        }

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.input_signal.name = value + '.input'
        self.output_signal.name = value + '.output'
        self.bias_signal.name = value + '.bias'

    @property
    def bias(self):
        return self.bias_signal.value

    @bias.setter
    def bias(self, value):
        self.bias_signal.value[...] = value

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_neurons(self):
        return self._n_neurons

    @n_neurons.setter
    def n_neurons(self, _n_neurons):
        self._n_neurons = _n_neurons
        self.input_signal.n = _n_neurons
        self.output_signal.n = _n_neurons
        self.bias_signal.n = _n_neurons
        self.bias_signal.value = np.zeros(_n_neurons)

    @property
    def n_out(self):
        return self.n_neurons

    def set_gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.

        """
        logging.debug("Setting gain and bias on %s", self.name)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts

    def rates(self, J_without_bias):
        """Return LIF firing rates for current J in Hz

        Parameters
        ---------
        J: ndarray of any shape
            membrane voltages
        tau_rc: broadcastable like J
            XXX
        tau_ref: broadcastable like J
            XXX
        """
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            J = J_without_bias + self.bias
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A


class LIFRate(_LIFBase):
    operator = sim.SimLIFRate
    def math(self, J):
        """Compute rates for input current (incl. bias)"""
        old = np.seterr(divide='ignore')
        try:
            j = np.maximum(J - 1, 0.)
            r = 1. / (self.tau_ref + self.tau_rc * np.log1p(1./j))
        finally:
            np.seterr(**old)
        return r


class LIF(_LIFBase):
    operator = sim.SimLIF
    def __init__(self, n_neurons, upsample=1, **kwargs):
        _LIFBase.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample

    def to_json(self):
        d = _LIFBase.to_json(self)
        d['upsample'] = self.upsample
        return d

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # N.B. J here *includes* bias

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        old = np.seterr(all='ignore')
        try:
            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked[:] = (v > 1) * 1.0

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = spiked * (spiketime + self.tau_ref) \
                                  + (1 - spiked) * (refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time
