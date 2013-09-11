"""
Low-level objects
=================

These classes are used to describe a nengo model (Model).
Model is the input to a *simulator* (see e.g. simulator.py).

"""
import numpy as np


"""
Set assert_named_signals True to raise an Exception
if model.signal is used to create a signal with no name.

This can help to identify code that's creating un-named signals,
if you are trying to track down mystery signals that are showing
up in a model.
"""
assert_named_signals = False


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

    def to_json(self):
        return {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            'n': self.n,
            'dtype': str(self.dtype),
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
