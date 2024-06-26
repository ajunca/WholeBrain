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
#           super().__init__(name='Foo')
#       @property
#       def value1(self):
#           return self._data['value1']
class ObservableResult:
    def __init__(self, name, data=dict()):
        assert isinstance(data, dict)
        self._name = name
        self._data = data

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
# define "_compute_from_fMRI" method.
#
# NOTES: Implementation is as this to maximize the portability with the none
# class based library.
class Observable:
    # Main method to compute the Observable from an fMRI BOLD signal.
    def from_fMRI(self, BOLD_signal, BOLD_filter=None) -> ObservableResult:
        # First check that there are no NaNs in the signal. If NaNs found, rise a warning and return None
        if np.isnan(BOLD_signal).any():
            warnings.warn(f'############ Warning!!! {self.__class__.__name__}.from_fMRI: NAN found ############')
            return None

        # Compute bold filter if needed, if not leave the signal as it is
        s = BOLD_signal
        if BOLD_filter is not None:
            s = BOLD_filter.apply_filter(BOLD_signal)

        # We still check that s is not None cause some possible future filters may return None even if nor NaNs
        # are present in the signal
        if s is None:
            return None

        return self._compute_from_fMRI(s)

    # Main method to compute the Observable from the adjacency matrix matrix.
    def from_adjacency_matrix(self, M):
        # Perform some check on the SC matrix. Check it is a np matrix and that it is square
        if not isinstance(M, np.ndarray) or not (M.shape[0] == M.shape[1]):
            warnings.warn(f'############ Warning!!! {self.__class__.__name__}.from_adjacency_matrix: Invalid matrix input ############')
            return None

        return self._compute_from_adjacency_matrix(M)

    # Virtual function. Performs the computation using the BOLD signal and returns the result.
    # Implemented in the deriving class if needed.
    def _compute_from_fMRI(self, bold_signal) -> ObservableResult:
        raise NotImplementedError()

    # Virtual function. Perform the computation using an adjacency matrix and return the result.
    # Implemented in the deriving class if needed.
    def _compute_from_adjacency_matrix(self, M) -> ObservableResult:
        raise NotImplementedError()