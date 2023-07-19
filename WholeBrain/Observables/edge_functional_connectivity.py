#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/blob/8434c0640e99a4042e0958fabd51e3de86c6692a/WholeBrain/Observables/eFC.py
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################

# ----------------------------------- ORIGINAL HEADER ----------------------------------
# --------------------------------------------------------------------------------------
# eFC: Edge Functional Connectivity,
#
# Faskowitz, J., Esfahlani, F.Z., Jo, Y. et al.
# Edge-centric functional network representations of human cerebral cortex reveal overlapping system-level architecture.
# Nat Neurosci 23, 1644–1654 (2020). https://doi.org/10.1038/s41593-020-00719-y
#
# Code at https://github.com/brain-networks/edge-centric_demo
#
# Code by Facundo Roffet,
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

from WholeBrain.Observables.observable import Observable, ObservableResult
import numpy as np
from scipy.stats import zscore

class EdgeFunctionalConnectivityResult(ObservableResult):
    def __init__(self, eFC, eTS):
        super().__init__(
            name='EdgeFunctionalConnectivity',
            data={
                'eFC': eFC, # Edge functional connectivity
                'eTS': eTS  # Edge time series
            }
        )

    @property
    def eFC(self):
        return self._data['eFC']

    @property
    def eTS(self):
        return self._data['eTS']


class EdgeFunctionalConnectivity(Observable):
    @staticmethod
    def __edge_ts(bold_signal):
        # Number of nodes
        n = bold_signal.shape[1]
        # Normalization
        z = zscore(bold_signal)
        # Indexes of the upper triangular matrix
        index = np.nonzero(np.triu(np.ones((n, n)), 1))
        u = index[0]
        v = index[1]
        # edge time series
        e_ts = np.multiply(z[:, u], z[:, v])
        return e_ts

    @staticmethod
    def __edge_ts_2_edge_corr(e_ts):
        b = np.matmul(np.transpose(e_ts), e_ts)
        c = np.sqrt(np.diagonal(b))
        c = np.expand_dims(c, axis=1)
        d = np.matmul(c, np.transpose(c))
        e = np.divide(b, d)
        return e


    # Apply the observable operator
    def _compute_from_fMRI(self, bold_signal) -> EdgeFunctionalConnectivityResult:
        e_ts = EdgeFunctionalConnectivity.__edge_ts(bold_signal)
        eFC = EdgeFunctionalConnectivity.__edge_ts_2_edge_corr(e_ts)
        result = EdgeFunctionalConnectivityResult(eFC=eFC, eTS=e_ts)
        return result
