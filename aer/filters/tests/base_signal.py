""" Generate & save (as NumPy binary files) some dummy 1D data """

import os
import numpy as np
rng = np.random.default_rng()

# Folder where to save the data
DATA_FOLDER = "dummy_data/"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
os.path.join(DATA_FOLDER, '')   # Ensure compat. between OSes

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

# Save the data in binary files
np.save(DATA_FOLDER+"tstp.npy", tstp)
np.save(DATA_FOLDER+"data.npy", data)
np.save(DATA_FOLDER+"frate.npy", frate)
np.save(DATA_FOLDER+"amps.npy", amps)
np.save(DATA_FOLDER+"freqs.npy", freqs)

## Time-domain signal
#import aer.display as disp
#fig = disp.signal(tstp, data,
#    fname="Signal.png",
#    fig_params={'figsize': (19.20, 10.80)})
#fig.show()

