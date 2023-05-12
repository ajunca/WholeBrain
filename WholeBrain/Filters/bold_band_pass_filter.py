#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/blob/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain/BOLDFilters.py
#
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################

import warnings
import numpy as np
from scipy.signal import butter, detrend, filtfilt
from WholeBrain.Utils import demean
from WholeBrain.Filters.signal_filter import SignalFilter

# At the moment this is the only signal filter we have. It takes 5 configuration parameters and transforms an input
# signal as needed.
# Parameters:
#   tr: fMRI TR time in seconds. Default 2.0s
#   flp: lowpass frequency of filter. Default: 0.02Hz
#   fhi: highpass frequency of filter. Default: 0.1Hz
#   k: k order butterworth filter. Default: 2
#   remove_strong_artifacts: Clamp artifacts "remove_strong_artifacts" standard deviations apart. Not applied if None.
#       Default: 3.0
class BOLDBandPassFilter(SignalFilter):
    def __init__(self, tr=2.0, flp=0.02, fhi=0.1, k=2, remove_strong_artifacts=3.0):
        self._tr = tr
        self._flp = flp
        self._fhi = fhi
        self._k = k
        self._remove_strong_artifacts = remove_strong_artifacts

    @property
    def tr(self):
        return self._tr

    @tr.setter
    def tr(self, value):
        self._tr = value

    @property
    def flp(self):
        return self._flp

    @flp.setter
    def flp(self, value):
        self._flp = value

    @property
    def fhi(self):
        return self._fhi

    @fhi.setter
    def fhi(self, value):
        self._fhi = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value

    @property
    def remove_strong_artifacts(self):
        return self._remove_strong_artifacts

    @remove_strong_artifacts.setter
    def remove_strong_artifacts(self, value):
        self._remove_strong_artifacts = value

    # Transforms the input signal based on the configuration parameters
    def apply_filter(self, bold_signal):
        (N, Tmax) = bold_signal.shape
        fnq = 1. / (2. * self.tr)  # Nyquist Frequency
        wn = [self.flp / fnq, self.fhi / fnq]  # Butterworth bandpass non-dimensional frequency
        bfilt, afilt = butter(self.k, wn, btype='band', analog=False)  # Construct the filter
        signal_filt = np.zeros(bold_signal.shape)

        for seed in range(N):
            if not np.isnan(bold_signal[seed, :]).any():  # Check that bold_signal data is OK
                ts = demean.demean(detrend(bold_signal[seed, :]))  # demean necessary? detrend probably does the job.

                if self.remove_strong_artifacts is not None:
                    r = self.remove_strong_artifacts
                    ts[ts > r * np.std(ts)] = r * np.std(ts)  # Remove high strong artifacts
                    ts[ts < -r * np.std(ts)] = -r * np.std(ts)  # Remove low strong artifacts

                # Band pass filter. padlen modified to get the same result as in Matlab
                signal_filt[seed, :] = filtfilt(bfilt, afilt, ts, padlen=3 * (max(len(bfilt), len(afilt)) - 1))

                # TODO: This is needed? For the moment lets comment it out.
                # if finalDetrend:  # Only for compatibility reasons. By default, don't!
                #    signal_filt[seed, :] = detrend(signal_filt[seed, :])

            else:
                # Signal is not good, we found nans in it... TODO: What is the best way to treat these errors?
                # At the moment we fire a warning and return None? Originally returning nan.
                warnings.warn(f'############ Warning!!! BandPassFilter: NAN found at region {seed} ############')
                return None

        # All done, we can return the result
        return signal_filt
