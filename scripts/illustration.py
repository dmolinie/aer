""" Simple illustration image """

import numpy as np

import aer.data_tk as dtk
import matplotlib.pyplot as plt
from aer.plot import set_margins
import aer.display as disp
from aer.filters import FilterBank

rng = np.random.default_rng()
plt.rcParams.update({'font.size': 12})

#-------------------------   Generate Dummy Data   --------------------------#
# Signal Parameters
frate = 200.                    # Sampling frequency
#noise = 0.001 * frate * 0.5    # Power of the signal noise

tstp = np.arange(0, 2, 1./frate)

bins = 4
off = len(tstp) // bins
amps = 2.5*(rng.random(bins)+0.25)
freqs = 0.5*frate*rng.random(bins)

DOUBLEPI = 2.0*np.pi
def sinus(amp, freq, tstp):
    return amp * np.sin(DOUBLEPI * freq * tstp)

pos = 0
data = np.empty_like(tstp)
for amp, freq in zip(amps, freqs):
    data[pos:pos+off] = sinus(amp, freq, tstp[pos:pos+off])
    pos += off
#----------------------------------------------------------------------------#

#--------------------   Filter Bank Frequency Response   --------------------#
# Build the frequency-domain sub-bands on a given scale
nb_filters = 11
bandwidth = (0., 0.5*frate)

# Compute the filters' temporal impulse response
fbank = FilterBank(nb_filters, bandwidth, 'rect')
fbank.build_bands(overlap=True, scale='hz')
fbank.build_filters(tstp, rescale=True, sym=True)

# Compute the spectral representation of the filters (FFT)
filters_fft = fbank.spectrum(onesided=True, cmplx=False)

# Plot the filters' frequency response
freq = np.fft.fftfreq(len(filters_fft[0]))*frate
fig = disp.filterbank_spectrum(
    freq, filters_fft,
    fig_params={'figsize': (19.20, 3.20)},
    title=None,
    labels=None,
#    labels=("Filterbank frequency response", None),
#    label_params={'size': 20, 'labelpad': 12},
)
axis = fig.gca()
## Remove the bordering box
#plt.box(False)
## Remove the x & y margins
#set_margins(axis, x=0.0, y=0.0)
## Remove the x & y ticks
# Deactivate the axes (and their ticks)
#axis.set_xticks([])
#axis.set_yticks([])
axis.axis('off')
# Save the figure
plt.savefig(fname="Fbank_Resp.svg", bbox_inches='tight')
#fig.show()
#----------------------------------------------------------------------------#
