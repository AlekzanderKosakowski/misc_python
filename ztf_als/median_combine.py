import numpy as np
from numba import njit

#@njit
def median_combine(sdata):
    #
    # Combine multi-filter data into a single data set.
    # Normalizes data to the ZTF g-band where available, else r-band
    #
    zg = sdata[np.where(sdata[:,3] == 1.)[0]]
    mg = np.median(zg[:,1]) if zg.size>0 else False # mg is the median of all g-band observations. This if False if there are no gband observations

    zr = sdata[np.where(sdata[:,3] == 2.)[0]] # Same for ZTF rband
    mr = np.median(zr[:,1]) if zr.size>0 else False
    if mr: # If there exists r-band data, calculate the correction to line it up with the gband
        rcorrection = mg - mr if mg else 0

    zi = sdata[np.where(sdata[:,3] == 3.)[0]] # Same for ZTF iband
    mi = np.median(zi[:,1]) if zi.size>0 else False
    if mi:
        if mg: # Calculate gband correction for iband
            icorrection = mg - mi
        else: # Calculate rband correction for iband if no gband data exists for this target
            icorrection = mr - mi if mr else 0

    odata = np.zeros((len(zg)+len(zr)+len(zi),4)) # odata is "Output Data" and is the final combined, g-band normalized, collection of data for one target.
                                                  # odata = [[hjd1, mag1, merr1, filter1], [hjd2, mag2, merr2, filter2]]. Filter is used for plot coloring later

    if zg.size>0:
        for g,k in enumerate(zg): # Arbitrarily include the gband data first
            odata[g] = [k[0], k[1], k[2], k[3]] # odata = [[hjd1, mag1, merr1, filter1], [hjd2, mag2, merr2, filter2]]. Filter is used for plot coloring later
    else:
        g = -1 # if there is no data in zg filter, then set the g index to -1 in order to cancel the +1 in the r and i band lines later

    if zr.size>0: # Include the rband data
        for r,k in enumerate(zr):
            odata[g+r+1] = [k[0], k[1]+(rcorrection), k[2], k[3]]
    else:
        r = -1 # if there is no data in zr filter, then set the r index to -1 in order to cancel the +1 in the i band lines later

    if zi.size>0: # Include the iband data.
        for i,k in enumerate(zi):
            odata[g+r+i+2] = [k[0], k[1]+(icorrection), k[2], k[3]]

    return(odata.T)
