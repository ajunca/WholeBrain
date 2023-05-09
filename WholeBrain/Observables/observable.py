#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/tree/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################

import warnings
import numpy as np


# This is a form of standardizing Observable result. It holds a dictionary "_data" with its results values in it.
# Different Observables can derive from it and then define properties for easy access.
# For example:
#   class FooObservableResult(ObservableResult):
#       def __init__(self):
#           super().__init__('Foo')
#       @property
#       def value1(self):
#           return self._data['value1']
class ObservableResult:
    def __init__(self, name):
        self._name = name
        self._data = dict()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def data(self):
        return self._data


# Abstract class for Observables. At the moment it has a main method "from_fmri" that takes the signal and the filter
# as parameters and outputs the result if computable (or None if some problem occurred). Each implementation has to
# define "_compute_from_fmri" method.
#
# NOTE: At the moment every Observable subclass outputs its own result format, so maybe at a future we have to
# standardize it somehow...
#
# MORE NOTES: Maybe it should be called ObservableOperator as it operates on signals and spits the result but is not
# itself and Observable(?). Also note that the implementation is as this to maximize the portability with the none
# class based library.
class Observable:
    # Main method to compute the Observable from an fMRI BOLD signal.
    def from_fmri(self, bold_signal, bold_filter=None) -> ObservableResult:
        # First check that there are no NaNs in the signal. If NaNs found, rise a warning and return None
        if np.isnan(bold_signal).any():
            warnings.warn(f'############ Warning!!! {self.__class__.__name__}.from_fmri: NAN found ############')
            return None

        # Compute bold filter if needed, if not leave the signal as it is
        s = bold_signal
        if bold_filter is not None:
            s = bold_filter.apply_filter(bold_signal)

        # We still check that s is not None cause some possible future filters may return None even if nor NaNs
        # are present in the signal
        if s is None:
            return None

        return self._compute_from_fmri(s)

    # Virtual function. Performs the observable computation and returns the result.
    # Needs to be implemented on the deriving classes.
    def _compute_from_fmri(self, bold_signal) -> ObservableResult:
        pass
