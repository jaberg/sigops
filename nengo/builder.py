import numpy as np


class ShapeMismatch(ValueError):
    pass


class SignalView(object):

    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert base is not None
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
        return (self.shape, self.elemstrides, self.offset)

    def same_view_as(self, other):
        return self.structure == other.structure and self.base == other.base

    @property
    def dtype(self):
        return np.dtype(self.base.dtype)

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
        elif self.size == 1:
            # -- scalars can be reshaped to any number of (1, 1, 1...)
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1] * len(shape)
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
            raise NotImplementedError('reshape of strided view')

    def transpose(self, neworder=None):
        if neworder:
            raise NotImplementedError()
        return SignalView(
            self.base,
            reversed(self.shape),
            reversed(self.elemstrides),
            self.offset,
            self.name + '.T'
        )

    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return self.transpose()

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

    def is_contiguous(self, return_range=False):
        def ret_false():
            if return_range:
                return False, None, None
            else:
                return False
        shape, strides, offset = self.structure
        if not shape:
            if return_range:
                return True, offset, offset + 1
            else:
                return True
        if len(shape) == 1:
            if strides[0] == 1:
                if return_range:
                    return True, offset, offset + shape[0]
                else:
                    return True
            else:
                return ret_false()
        if len(shape) == 2:
            if strides == (1, shape[0]) or strides == (shape[1], 1):
                if return_range:
                    return True, offset, offset + shape[0] * shape[1]
                else:
                    return True
            else:
                return ret_false()

        raise NotImplementedError()
        # if self.ndim == 1 and self.elemstrides[0] == 1:
            # return self.offset, self.offset + self.size

    def shares_memory_with(self, other):
        # XXX: WRITE SOME UNIT TESTS FOR THIS FUNCTION !!!
        # Terminology: two arrays *overlap* if the lowermost memory addressed
        # touched by upper one is higher than the uppermost memory address
        # touched by the lower one.
        #
        # np.may_share_memory returns True iff there is overlap.
        # Overlap is a necessary but insufficient condition for *aliasing*.
        #
        # Aliasing is when two ndarrays refer a common memory location.
        if self.base is not other.base:
            return False
        if self is other or self.same_view_as(other):
            return True
        if self.ndim < other.ndim:
            return other.shares_memory_with(self)
        if self.size == 0 or other.size == 0:
            return False

        assert self.ndim > 0
        if self.ndim == 1:
            # -- self is a vector view
            #    and other is either a scalar or vector view
            ae0, = self.elemstrides
            be0, = other.elemstrides
            amin = self.offset
            amax = amin + self.shape[0] * ae0
            bmin = other.offset
            bmax = bmin + other.shape[0] * be0
            if amin <= amax <= bmin <= bmax:
                return False
            elif bmin <= bmax <= amin <= amax:
                return False
            if ae0 == be0 == 1:
                # -- strides are equal, and we've already checked for
                #    non-overlap. They do overlap, so they are aliased.
                return True
            # TODO: look for common divisor of ae0 and be0
            raise NotImplementedError('1d', (self.structure, other.structure))
        elif self.ndim == 2:
            # -- self is a matrix view
            #    and other is either a scalar, vector or matrix view
            a_contig, amin, amax = self.is_contiguous(return_range=True)
            if a_contig:
                # -- self has a contiguous memory layout,
                #    from amin up to but not including amax
                b_contig, bmin, bmax = other.is_contiguous(return_range=True)
                if b_contig:
                    # -- other is also contiguous
                    if amin <= amax <= bmin <= bmax:
                        return False
                    elif bmin <= bmax <= amin <= amax:
                        return False
                    else:
                        return True
                raise NotImplementedError(
                    '2d self:contig, other:discontig',
                    (self.structure, other.structure))
            raise NotImplementedError('2d', (self.structure, other.structure))
        else:
            raise NotImplementedError()


class Signal(SignalView):
    """Interpretable, vector-valued quantity within Nengo"""

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(self, value, name=None):
        self.value = np.asarray(value, dtype=np.float64)
        if name is not None:
            self._name = name
        if Signal.assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", shape=" + str(self.shape) + ")"
        except AttributeError:
            return ("Signal(id " + str(id(self)) + ", shape="
                    + str(self.shape) + ")")

    def __repr__(self):
        return str(self)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(map(int, s / self.dtype.itemsize))

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self


class Operator(object):
    """Base class for operator instances understood by nengo.Simulator.

    The lifetime of a Signal during one simulator timestep:
    0) at most one set operator (optional)
    1) any number of increments
    2) any number of reads
    3) at most one update

    A signal that is only read can be considered a "constant".

    A signal that is both set *and* updated can be a problem:
    since reads must come after the set, and the set will destroy
    whatever were the contents of the update, it can be the case
    that the update is completely hidden and rendered irrelevant.
    There are however at least two reasons to use both a set and an update:
    (a) to use a signal as scratch space (updating means destroying it)
    (b) to use sets and updates on partly overlapping views of the same
        memory.

    N.B.: It is done on purpose that there are no default values for
    reads, sets, incs, and updates.

    Each operator should explicitly set each of these properties.
    """

    @property
    def reads(self):
        """Signals that are read and not modified"""
        return self._reads

    @reads.setter
    def reads(self, val):
        self._reads = val

    @property
    def sets(self):
        """Signals assigned by this operator

        A signal that is set here cannot be set or updated
        by any other operator.
        """
        return self._sets

    @sets.setter
    def sets(self, val):
        self._sets = val

    @property
    def incs(self):
        """Signals incremented by this operator

        Increments will be applied after this signal has been
        set (if it is set), and before reads.
        """
        return self._incs

    @incs.setter
    def incs(self, val):
        self._incs = val

    @property
    def updates(self):
        """Signals assigned their value for time t + 1

        This operator will be scheduled so that updates appear after
        all sets, increments and reads of this signal.
        """
        return self._updates

    @updates.setter
    def updates(self, val):
        self._updates = val

    @property
    def all_signals(self):
        return self.reads + self.sets + self.incs + self.updates

    def init_signals(self, signals, dt):
        """Initialize simulator.signals

        Install any buffers into the signals view that
        this operator will need. Classes for nonlinearities
        that use extra buffers should create them here.
        """
        for sig in self.all_signals:
            if sig.base not in signals:
                signals.init(sig.base,
                             np.asarray(
                                 np.zeros(sig.base.shape,
                                          dtype=sig.base.dtype)
                                 + sig.base.value))


class Reset(Operator):
    """Assign a constant value to a Signal."""

    def __init__(self, dst, value=0):
        self.dst = dst
        self.value = float(value)

        self.reads = []
        self.incs = []
        self.updates = []
        self.sets = [dst]

    def __str__(self):
        return 'Reset(%s)' % str(self.dst)

    def make_step(self, signals, dt):
        target = signals[self.dst]
        value = self.value

        def step():
            target[...] = value
        return step


class Copy(Operator):
    """Assign the value of one signal to another."""

    def __init__(self, dst, src, as_update=False, tag=None):
        self.dst = dst
        self.src = src
        self.tag = tag
        self.as_update = True

        self.reads = [src]
        self.sets = [] if as_update else [dst]
        self.updates = [dst] if as_update else []
        self.incs = []

    def __str__(self):
        return 'Copy(%s -> %s, as_update=%s)' % (
            str(self.src), str(self.dst), self.as_update)

    def make_step(self, signals, dt):
        dst = signals[self.dst]
        src = signals[self.src]

        def step():
            dst[...] = src
        return step


def reshape_dot(A, X, Y, tag=None):
    """Checks if the dot product needs to be reshaped.

    Also does a bunch of error checking based on the shapes of A and X.
    """
    badshape = False
    ashape = (1,) if A.shape == () else A.shape
    xshape = (1,) if X.shape == () else X.shape
    if A.shape == ():
        incshape = X.shape
    elif X.shape == ():
        incshape = A.shape
    elif X.ndim == 1:
        badshape = ashape[-1] != xshape[0]
        incshape = ashape[:-1]
    else:
        badshape = ashape[-1] != xshape[-2]
        incshape = ashape[:-1] + xshape[:-2] + xshape[-1:]

    if (badshape or incshape != Y.shape) and incshape != ():
        raise ValueError('shape mismatch in %s: %s x %s -> %s' % (
            tag, A.shape, X.shape, Y.shape))

    # If the result is scalar, we'll reshape it so Y[...] += inc works
    return incshape == ()


class DotInc(Operator):
    """Increment signal Y by dot(A, X)"""

    def __init__(self, A, X, Y, tag=None):
        self.A = A
        self.X = X
        self.Y = Y
        self.tag = tag

        self.reads = [self.A, self.X]
        self.incs = [self.Y]
        self.sets = []
        self.updates = []

    def __str__(self):
        return 'DotInc(%s, %s -> %s "%s")' % (
            str(self.A), str(self.X), str(self.Y), self.tag)

    def make_step(self, signals, dt):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            inc = np.dot(A, X)
            if reshape:
                inc = np.asarray(inc).reshape(Y.shape)
            Y[...] += inc
        return step


class ProdUpdate(Operator):
    """Sets Y <- dot(A, X) + B * Y"""

    def __init__(self, A, X, B, Y, tag=None):
        self.A = A
        self.X = X
        self.B = B
        self.Y = Y
        self.tag = tag

        self.reads = [self.A, self.X, self.B]
        self.updates = [self.Y]
        self.incs = []
        self.sets = []

    def __str__(self):
        return 'ProdUpdate(%s, %s, %s, -> %s "%s")' % (
            str(self.A), str(self.X), str(self.B), str(self.Y), self.tag)

    def make_step(self, signals, dt):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        B = signals[self.B]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            val = np.dot(A, X)
            if reshape:
                val = np.asarray(val).reshape(Y.shape)
            Y[...] *= B
            Y[...] += val
        return step
