#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/blob/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain/Observables/Metastability.py
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################

# ----------------------------- ORIGINAL HEADER ----------------------------
# --------------------------------------------------------------------------
#  Computes the Metastability (network synchrony) of a signal, using the Kuramoto order paramter
#
#  Explained at
#  [Shanahan2010] Metastable chimera states in community-structured oscillator networks,
#             Shanahan, M.,
#             Chaos 20 (2010), 013108.
#             DOI: 10.1063/1.3305451
#  [Cabral2011] Role of local network oscillations in resting-state functional connectivity,
#             Joana Cabral, Etienne Hugues, Olaf Sporns, Gustavo Deco,
#             NeuroImage 57 (2011) 130–139,
#             DOI: 10.1016/j.neuroimage.2011.04.010
#  [Cabral2014] Exploring mechanisms of spontaneous functional connectivity in MEG: How delayed network interactions lead to structured amplitude envelopes of band-pass filtered oscillations,
#             Joana Cabral, Henry Luckhoo, Mark Woolrich, Morten Joensson, Hamid Mohseni, Adam Baker, Morten L. Kringelbach, Gustavo Deco,
#             NeuroImage 90 (2014) 423–435
#             DOI: 10.1016/j.neuroimage.2013.11.047
#  and probably many others
#
#  Code by... probably Gustavo Deco, provided by Xenia Kobeleva
#  Translated by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


from scipy import signal
from WholeBrain.Utils import demean
from WholeBrain.Observables.observable import Observable, ObservableResult
import numpy as np


class MetastabilityResult(ObservableResult):
    def __init__(self):
        super().__init__(name='Metastability')

    @property
    def metastability(self):
        return self._data['metastability']

class Metastability(Observable):
    def _compute_from_fMRI(self, bold_signal) -> MetastabilityResult:
        (n, t_max) = bold_signal.shape
        npattmax = t_max - 19    # Calculates the size of phfcd vector

        # Some data structures we are going to need
        phases = np.zeros([n, t_max])

        # Parent class already applies the bold filter if not None, so no need in here

        # Compute phases
        for n in range(n):
            # TODO: Is demean really necessary? Done when bold filter is applied? What if bold filter is not applied?
            x_analytic = signal.hilbert(demean.demean(bold_signal[n, :]))
            phases[n, :] = np.angle(x_analytic)

        T = np.arange(10, t_max - 10 + 1)
        sync = np.zeros(T.size)
        for t in T:
            ku = np.sum(np.cos(phases[:, t-1]) + 1j * np.sin(phases[:, t-1])) / n
            sync[t - 10] = abs(ku)

        # Return the metastability value
        # return np.std(sync)
        result = MetastabilityResult()
        result.data['metastability'] = np.std(sync)
        return result
