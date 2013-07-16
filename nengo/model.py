import inspect
import math
import random

from . import logger
from .model_objects import *
from .simulator_objects import *


class Model(object):
    """A model contains a single network and the ability to
    run simulations of that network.

    Model is the first part of the API that modelers
    become familiar with, and it is possible to create
    many of the models that one would want to create simply
    by making a model and calling functions on that model.

    For example, a model that implements a communication channel
    between two ensembles of neurons can be created with::

        import nengo
        model = nengo.Model("Communication channel")
        input = model.make_node("Input", values=[0])
        pre = model.make_ensemble("In", neurons=100, dimensions=1)
        post = model.make_ensemble("Out", neurons=100, dimensions=1)
        model.connect(input, pre)
        model.connect(pre, post)

    Parameters
    ----------
    name : str
        Name of the model.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the model
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire model changes.
    fixed_seed : int, optional
        Random number seed that will be fed to the random number generator
        before each random process. Unlike setting ``seed``,
        each new ensemble in the network will use ``fixed_seed``,
        meaning that ensembles with the same properties will have the same
        set of neurons generated.
    backend : str, optional
        The backend that this model should use.

        If you have installed a Nengo backend, such as the Theano backend,
        then pass in the appropriate string to use that backend for this model.

        **Default**: ``'numpy'``, the Python reference implementation.


    Attributes
    ----------
    name : str
        Name of the model
    seed : int
        Random seed used by the model.
    backend : str
        The backend that is implementing this model.
    time : float
        The amount of time that this model has been simulated.
    metadata : dict
        An editable dictionary that modelers can use to store
        extra information about a network.
    properties : read-only dict
        A collection of basic information about
        a network (e.g., number of neurons, number of synapses, etc.)

    """

    BACKENDS = {
        'numpy': 'nengo.simulator',
    }

    def __init__(self, name, seed=None, fixed_seed=None, backend='numpy',
                 dt=0.001):
        self.dt = dt

        self.signals = set()
        self.transforms = set()
        self.filters = set()
        self.probes = set()

        self.objs = {}
        self.aliases = {}
        self.probed = {}
        self.probe_data = {}

        self.simtime = self.add(Signal(name='simtime'))
        self.steps = self.add(Signal(name='steps'))
        self.one = self.add(Constant(1, value=[1.0], name='one'))

        # -- steps counts by 1.0
        self.add(Filter(1.0, self.one, self.steps))
        self.add(Filter(1.0, self.steps, self.steps))

        # simtime <- dt * steps
        self.add(Filter(dt, self.one, self.simtime))
        self.add(Filter(dt, self.steps, self.simtime))

        self.name = name
        self.backend = backend

        if seed is None:
            self.seed = 123
        else:
            self.seed = seed

        if fixed_seed is not None:
            raise NotImplementedError()

    def __str__(self):
        return "Model: " + self.name

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if hasattr(self, '_backend') and backend == self._backend:
            return

        try:
            toimport = Model.BACKENDS[backend]
            self.simulator = __import__(toimport, globals(), locals(),
                                        ['simulator'], -1)
            self.sim_obj = None
            self._backend = backend

        except KeyError:
            print (backend + " not a registered backend. "
                   "Falling back to numpy.")
            self.backend = 'numpy'

        except ImportError:
            if backend == 'numpy':
                raise ImportError("Cannot import numpy backend!")
            print (backend + " cannot be imported. "
                   "Falling back to numpy.")
            self.backend = 'numpy'

    @property
    def time(self):
        if self.sim_obj is None:
            return None
        return self.sim_obj.simulator_time

    @property
    def objects(self):
        return self.objs.values()

    ### Simulation methods

    def reset(self):
        """Reset the state of the simulation.

        Runs through all nodes, then ensembles, then connections and then
        probes in the network and calls thier reset functions.

        """
        logger.debug("Resetting simulator")
        self.sim_obj.reset()

    def run(self, time, dt=0.001, output=None, stop_when=None):
        """Runs a simulation of the model.

        Parameters
        ----------
        time : float
            How long to run the simulation, in seconds.

            If called more than once, successive calls will continue
            the simulation for ``time`` more seconds.
            To reset the simulation, call :func:`nengo.Model.reset()`.
            Typical use cases are to either simply call it once::

              model.run(10)

            or to call it multiple times in a row::

              time = 0
              dt = 0.1
              while time < 10:
                  model.run(dt)
                  time += dt
        dt : float, optional
            The length of a timestep, in seconds.

            **Default**: 0.001
        output : str or None, optional
            Where probed data should be output.

            If ``output`` is None, then probed data will be returned
            by this function as a dictionary.

            If ``output`` is a string, it is interpreted as a path,
            and probed data will be written to that file.
            The file extension will be parsed to determine the type
            of file to write; any unrecognized extension
            will be ignored and a comma-separated value file will
            be created.

            **Default**: None, so this function returns a dictionary
            of probed data.

        Returns
        -------
        data : dictionary
            All of the probed data. This is only returned if
            ``output`` is None.

        """
        if self.sim_obj is None:
            logger.debug("No simulator object yet. Building.")
            self.sim_obj = self.simulator.Simulator(self)
        if stop_when is not None:
            raise NotImplementedError()
        if output is not None:
            raise NotImplementedError()

        steps = int(time / dt)
        logger.debug("Running simulator for " + str(steps) + " steps")
        self.sim_obj.run_steps(steps)

        for k in self.probed:
            self.probe_data[k] = self.sim_obj.probe_data(self.probed[k])

        return self.probe_data

    ### Model manipulation

    def add(self, obj):
        """Adds a Nengo object to this model.

        This is generally only used for manually created nodes, not ones
        created by calling :func:`nef.Model.make_ensemble()` or
        :func:`nef.Model.make_node()`, as these are automatically added.
        A common usage is with user created subclasses, as in the following::

          node = net.add(MyNode('name'))

        Parameters
        ----------
        obj : Nengo object
            The Nengo object to add.

        Returns
        -------
        obj : Nengo object
            The Nengo object that was added.

        See Also
        --------
        Network.add : The same function for Networks

        """
        if hasattr(obj, 'name') and self.objs.has_key(obj.name):
            raise ValueError("Something called " + obj.name + " already exists."
                             " Please choose a different name.")
        obj.add_to_model(self)
        if hasattr(obj, 'name'):
            self.objs[obj.name] = obj
        return obj

    def get(self, target, default=None):
        """Return the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The ``target`` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a Nengo object is passed, ``get`` just confirms
            that ``target`` is a part of the model.

        default : optional
            If ``target`` is not in the model, then ``get`` will
            return ``default``.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by ``target``.

        """
        if isinstance(target, str):
            if self.aliases.has_key(target):
                return self.aliases[target]
            elif self.objs.has_key(target):
                return self.objs[target]
            print "Cannot find " + target + " in this model."
            return default

        if not target in self.objs.values():
            print "Cannot find " + str(target) + " in this model."
            return default

        return target

    def get_string(self, target, default=None):
        """Return the canonical string of the Nengo object specified.

        Parameters
        ----------
        target : string or Nengo object
            The ``target`` can be specified with a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object.
            If a string is passed, ``get_string`` returns
            the canonical version of it; i.e., if it is
            an alias, the non-aliased version is returned.

        default : optional
            If ``target`` is not in the model, then ``get`` will
            return ``default``.

        Returns
        -------
        target : Nengo object
            The Nengo object specified by ``target``.

        Raises
        ------
        ValueError
            If the ``target`` does not exist and no ``default`` is specified.

        """
        if isinstance(target, str):
            if self.aliases.has_key(target):
                obj = self.aliases[target]
            elif self.objs.has_key(target):
                return target

        for k, v in self.objs.iteritems():
            if v == target:
                return k

        print "Cannot find " + str(target) + " in this model."
        return default

    def remove(self, target):
        """Removes a Nengo object from the model.

        Parameters
        ----------
        target : str, Nengo object
            A string referencing the Nengo object to be removed
            (see `string reference <string_reference.html>`)
            or node or name of the node to be removed.

        Returns
        -------
        target : Nengo object
            The Nengo object removed.

        """
        obj = self.get(target)
        if obj is None:
            print target + " not in this model."
            return

        obj.remove_from_model(self)

        for k, v in self.objs.iteritems():
            if v == obj:
                del self.objs[k]
        for k, v in self.aliases.iteritem():
            if v == obj:
                del self.aliases[k]

        return obj

    def alias(self, alias, target):
        """Adds a named shortcut to an existing Nengo object
        within this model.

        This is designed to simplify :func:`nengo.Model.connect()`,
        :func:`nengo.Model.get()`, and :func:`nengo.Model.remove()` calls.
        For example, you can do::

            model.make_alias('vision', 'A.B.C.D.E')
            model.make_alias('motor', 'W.X.Y.Z')
            model.connect('vision', 'motor')

        Parameters
        ----------
        alias : str
            The alias to assign to ``target``.
        target : str or Nengo object
            Identifies the Nengo object to be aliased.

        Raises
        ------
        ValueError
            If ``target`` can't be found in the model.

        """
        obj = self.get(target)
        if obj is None:
            raise ValueError(target + " cannot be found.")
        self.aliases[alias] = obj
        return obj


    # Model creation methods

    def probe(self, target, sample_every=None, pstc=None, static=False):
        """Probe a piece of data contained in the model.

        When a piece of data is probed, it will be recorded through
        the course of the simulation.

        Parameters
        ----------
        target : str, Nengo object
            The piece of data being probed.
            This can specified as a string
            (see `string reference <string_reference.html>`_)
            or a Nengo object. Each Nengo object will emit
            what it considers to be the most useful piece of data
            by default; if that's not what you want,
            then specify the correct data using the string format.
        sample_every : float, optional
            How often to sample the target data, in seconds.

            Some types of data (e.g. connection weight matrices)
            are very large, and change relatively slowly.
            Use ``sample_every`` to limit the amount of data
            being recorded. For example::

              model.probe('A>B.weights', sample_every=0.5)

            records the value of the weight matrix between
            the ``A`` and ``B`` ensembles every 0.5 simulated seconds.

            **Default**: Every timestep (i.e., ``dt``).
        static : bool, optional
            Denotes if a piece of data does not change.

            Some data that you would want to know about the model
            does not change over the course of the simulation;
            this includes things like the properties of a model
            (e.g., number of neurons or connections) or the random seed
            associated with a model. In these cases, to record that data
            only once (for later being written to a file),
            set ``static`` to True.

            **Default**: False

        """
        def _filter_coefs(pstc, dt):
            pstc = max(pstc, dt)
            decay = math.exp(-dt / pstc)
            return decay, (1.0 - decay)

        if sample_every is None:
            sample_every = self.dt

        obj = self.get(target)
        obj_s = self.get_string(target)

        if pstc is not None and pstc > self.dt:
            fcoef, tcoef = _filter_coefs(pstc=pstc, dt=self.dt)
            probe_sig = self.signal(obj.sig.n)
            self.filter(fcoef, probe_sig, probe_sig)
            self.transform(tcoef, obj.sig, probe_sig)
            p = SimModel.probe(self, probe_sig, sample_every)
        else:
            p = SimModel.probe(self, obj.sig, sample_every)

        i = 0
        while self.probed.has_key(obj_s):
            i += 1
            obj_s = self.get_string(target) + "_" + str(i)

        self.probed[obj_s] = p
        return p
