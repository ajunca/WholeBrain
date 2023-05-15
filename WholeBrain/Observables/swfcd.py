#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/blob/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain/Observables/swFCD.py
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################

from WholeBrain.Observables.observable import Observable, ObservableResult
import numpy as np
import warnings


class swFCDResult(ObservableResult):
    def __init__(self):
        super().__init__(name='swFCD')

    @property
    def swFCD(self):
        return self._data['swFCD']


# Computes the sliding-window Functional Connectivity Dynamics (swFCD)
class swFCD(Observable):
    def __init__(self, window_size=30, window_step=3):
        self._window_size = window_size
        self._window_step = window_step

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self._window_size = value

    @property
    def window_step(self):
        return self._window_step

    @window_step.setter
    def window_step(self, value):
        self._window_step = value

    @staticmethod
    def __calc_length(start, end, step) -> int:
        # This fails for a negative step e.g., range(10, 0, -1).
        # From https://stackoverflow.com/questions/31839032/python-how-to-calculate-the-length-of-a-range-without-creating-the-range
        return (end - start - 1) // step + 1

    @staticmethod
    def __pearson_r(x, y):
        # Compute correlation matrix
        corr_mat = np.corrcoef(x.flatten(), y.flatten())
        return corr_mat[0, 1]

    def _compute_from_fMRI(self, signal) -> swFCDResult:
        (n, t_max) = signal.shape
        last_window = t_max - self._window_size
        n_windows = self.__calc_length(0, last_window, self._window_step)

        if np.isnan(signal).any():
            warnings.warn('############ Warning!!! swFCD.from_fMRI: NAN found ############')
            return None

        isubdiag = np.tril_indices(n, k=-1)  # Indices of triangular lower part of the matrix

        # For each pair of sliding windows calculate the FC at t and t2 and
        # compute the correlation between the two.
        cotsampling = np.zeros((int(n_windows * (n_windows - 1) / 2)))
        kk = 0
        ii2 = 0
        for t in range(0, last_window, self._window_step):
            jj2 = 0
            # Extracts a (sliding) window between t and t+windowSize (included)
            sfilt = (signal[:, t:t + self._window_size + 1])
            # Pearson correlation coefficients. No need to transpose if rowvar=True
            cc = np.corrcoef(sfilt, rowvar=True)
            for t2 in range(0, last_window, self._window_step):
                # Extracts a (sliding) window between t2 and t2+windowSize (included)
                sfilt2 = (signal[:, t2:t2 + self._window_size + 1])
                # Pearson correlation coefficients. No need to transpose if rowvar=True
                cc2 = np.corrcoef(sfilt2, rowvar=True)
                ca = self.__pearson_r(cc[isubdiag], cc2[isubdiag])  # Correlation between both FC
                if jj2 > ii2:  # Only keep the upper triangular part
                    cotsampling[kk] = ca
                    kk = kk + 1
                jj2 = jj2 + 1
            ii2 = ii2 + 1

        result = swFCDResult()
        result.data['swFCD'] = cotsampling
        return result
