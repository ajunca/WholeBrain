#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/blob/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain/Observables/intrinsicIgnition.py
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################


from WholeBrain.Observables.intrinsic_ignition import IntrinsicIgnition
from WholeBrain.Utils import demean
import numpy as np
# from numba import jit
from scipy import signal


class PhaseBasedIntrinsicIgnition(IntrinsicIgnition):

    # TODO: Code repeated in metastability.py.
    @staticmethod
    # @jit(nopython=False)
    def compute_phases(node_signal, n, t_max):
        phases = np.zeros((n, t_max))
        for seed in range(n):  # obtain phases for each seed, i.e., each node
            x_analytic = signal.hilbert(demean.demean(node_signal[seed, :]))
            phases[seed, :] = np.angle(x_analytic)
        return phases

    # TODO: Maybe better in Utils?
    # @jit(nopython=True)
    @staticmethod
    # @jit(nopython=True)
    def adif(a, b):
        if np.abs(a - b) > np.pi:
            c = 2 * np.pi - np.abs(a - b)
        else:
            c = np.abs(a - b)
        return c

    # @jit(nopython=True)
    def _compute_integration(self, node_signal, events, n, t_max):
        # Integration
        # -----------
        # obtain 'events connectivity matrix' and integration value (integ):
        # for each time point:
        #    Compute the phase lock matrix P_{ij}(t), which describes the state of pair-wise phase synchronization
        #    at time t between regions i and k (from [EscrichsEtAl2021]).

        # First we need the phases. On the original code, phases are computed before calling "computePhaseBasedIntegration"
        phases = self.compute_phases(node_signal, n, t_max)

        phase_matrix = np.zeros((n, n))  # nodes x nodes
        integ = np.zeros(t_max)
        for t in range(t_max):  # (t_max-1, t_max)
            print(f"===========  Computing for t={t}/{t_max} ")
            for i in range(n):
                for j in range(n):
                    phase_matrix[i, j] = np.exp(-3.0 * self.adif(phases[i, t], phases[j, t]))
            cc = phase_matrix - np.eye(n)
            pr = np.arange(0, 0.99, 0.01)
            cs = np.zeros(len(pr))
            for pos, p in enumerate(pr):
                print(f'Processing PR={p} (t={t}/{t_max})')
                A = (np.abs(cc) > p).astype(np.int64)
                comps, csize = self.get_components(A)
                cs[pos] = np.max(csize)
            integ[t] = np.sum(cs) / 100 / n  # area under the curve / n = mean integration
        return integ
