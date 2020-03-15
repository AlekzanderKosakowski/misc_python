# Bin light curve data

# Takes input as 2D list or numpy array with columns time, mag, mag_err.
# Returns binned version of the input list or numpy array in same format.

# If only two columns in input, assume they are time and magnitude.
# If input length is not divisible by requested binning, the last element
#   will have fewer entries:
#   [[x1, x2, x3], [x4, x5, x6] . . . [x_N-1, x_N]]

import numpy as np
from more_itertools import chunked

def binned(input,b):

    # Bin light curve data by b.

    # Force input to be numpy array to allow .shape and .T
    input = np.array(input)
    s = input.shape

    # Check if reading by column or row.
    # Return same format as input.
    if s[0] < s[1]:

        x,y  = input[0],input[1]
        xbin = [np.mean(x) for x in list(chunked(x,b))]
        ybin = [np.mean(y) for y in list(chunked(y,b))]
        if s[0] == 3: # If 3 columns: include mag_err binning
            z    = input[2]
            zbin = [np.sqrt(sum(np.array(z)**2))/b for z in list(chunked(z,b))]
            return(list([xbin,ybin,zbin]))

        return(list([xbin,ybin]))

    else:

        # Transpose input to allow copy/paste code from above.
        input = input.T

        # Do the same stuff as above.
        x,y = input[0],input[1]
        xbin = [np.mean(x) for x in list(chunked(x,b))]
        ybin = [np.mean(y) for y in list(chunked(y,b))]
        if input.shape[0] == 3:
            z    = input[2]
            zbin = [np.sqrt(sum(np.array(z)**2))/b for z in list(chunked(z,b))]
            return(list(np.array([xbin,ybin,zbin]).T))

        return(list(np.array([xbin,ybin]).T))
