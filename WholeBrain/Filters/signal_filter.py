#####################################################################################
# Adapted/Refactored from Gustavo Patow code by Albert Juncà
#####################################################################################


# Abstract class so in the future we can implement more filters. At the moment only BoldBandPassFilter is implemented.
class SignalFilter:
    def apply_filter(self, signal):
        pass
