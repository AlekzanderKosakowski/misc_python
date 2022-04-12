from numba import njit
import numpy as np

@njit
def get_phase(odata, peak_freq):
    #
    # Create phase points for data and model.
    # Sort the model points so it appears as a smooth line with ax.plt()
    # Copy/Pasted from eleanor_astropy.py code
    #
    #
    # Assign phase to each flux point for phase-folded light curve.
    #
    sdata = sorted(odata, key=lambda l:l[0], reverse=False)

    phase = np.zeros(len(sdata))
    for i,j in enumerate(sdata):
        phase[i] = (j[0]-sdata[0][0])*peak_freq - int( (j[0]-sdata[0][0])*peak_freq )

    pdata = np.zeros((2*len(odata),5))
    for i in range(len(pdata)/2):
        pdata[i] = [sdata[i][0],sdata[i][1],sdata[i][2],sdata[i][3],phase[i]]
        pdata[i+len(odata)] = [sdata[i][0],sdata[i][1],sdata[i][2],sdata[i][3],phase[i]+1.0]

    # pdata is then the phase-data with format: [[hjd, mag, merr, filter, phase], [.....], [.....], [.....]]

    return(pdata.T)
