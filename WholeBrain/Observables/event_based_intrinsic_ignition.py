#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/blob/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain/Observables/intrinsicIgnition.py
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################


from WholeBrain.Observables.intrinsic_ignition import IntrinsicIgnition
import numpy as np
# from numba import jit


class EventBasedIntrinsicIgnition(IntrinsicIgnition):
    # # @jit(nopython=True)
    def _compute_integration(self, node_signal, events, n, t_max):
        # Integration
        # -----------
        # obtain 'events connectivity matrix' and integration value (integ)
        events_matrix = np.zeros([n, n])
        integ = np.zeros(t_max)
        for t in range(t_max):
            for i in range(n):
                for j in range(n):
                    events_matrix[i, j] = events[i, t] * events[j, t]
            cc = events_matrix - np.eye(n)
            comps, csize = self.get_components(cc)
            integ[t] = max(csize)/n
        return integ
