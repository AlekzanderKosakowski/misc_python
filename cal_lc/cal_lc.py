# Uses the output from the IRAF tasks PHOT or DAOPHOT to create
#   a calibration light curve applies it to the light curves in the PHOT
#   or DAOPHOT output
#
# The actually photometry is done in IRAF with PHOT or DAOPHOT.
#
# This code only automates the process of creating and applying a
#   weighted-mean calibration light curve.
#
# To obtain the output from PHOT or DAOPHOT in the proper format to use
#   for this code, you'll need to define the parameter 'OTIME' in the
#   PHOT or DAOPHOT parameter list. I usually assign HJD from SETJD to OTIME
# After running PHOT or DAOPHOT on all of your field stars at the same time,
#   type this command into your IRAF terminal:
#
#   txdump *.mag.1 OTIME,MAG,MERR yes > phot_file.txt
#
# This will create phot_file.txt in the proper format for this code.
#
# This file should come with 3 example light curves that I've taken using APO
#   phot_file_example1.txt (DOI: 10.1093/mnras/staa3571)
#   phot_file_example2.txt
#   phot_file_example3.txt
#
#     Example3 shows strong ellipsoidal variations.
#     This effect is overfit if you fit the calibration LC with order=2 or 3
#     I recommend using order=1 when running this code on example3
#     'order' is used in the flatten_lc() function



import numpy as np
import matplotlib.pyplot as plt
import sys # For testing purposes with sys.exit()


'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''


def count_stars(txdump):

    # Reads the input photfile generated from txdump to count the number
    #   of field stars and images photometry was performed on.

    nstars = 0
    for k in txdump[:,0]:
        if k == txdump[0][0]:
            nstars += 1
        else:
            break

    npts = int(len(txdump)/nstars)

    return(nstars,npts)

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''

def split_stars(txdump,nstars,npts):

    # Splits the input photfile into nstars different light curves with
    #   length npts each.
    # Replaces INDEF magnitudes with their respective star's median magnitute
    # Replaces INDEF mag errors with 99999.0

    lightcurves = np.zeros((nstars,3,npts)).astype(str)
    for k in range(nstars):

        hjd, mag, merr = txdump[k::nstars].T

        mask = np.where(mag != 'INDEF')
        med = np.median(mag[mask].astype(np.float64))

        mag  = np.array([m if m != 'INDEF' else str(med) for m in mag])
        merr = np.array([e if e != 'INDEF' else '99999.0' for e in merr])

        lightcurves[k] = np.array([hjd, mag, merr])

    return(np.array(lightcurves,dtype=np.float64))

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''

def mk_cal_lc(lightcurves):

    # Create a weighted-mean calibration light curve from all of the data
    #   in the PHOT or DAOPHOT output except the first star.
    # The first star is assumed to be the target of interest.
    # The resulting calibration light curve is centered at about 0 median

    for k in lightcurves:
        k[1] -= np.median(k[1])

    hjd = lightcurves[0][0]
    mags = lightcurves[:,1]
    weights = 1./lightcurves[:,2]**2

    weighted,werr = np.average(mags,axis=0,weights=weights,returned=True)
    werr = werr**-0.5

    cal_lc = np.array([hjd, weighted, werr],dtype=np.float64)

    return(cal_lc)

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''

def apply_cal_lc(lightcurves, cal_lc):

    # Subtracts the calibration light curve from all of the individual star
    #   light curves.
    # Note that the PHOT or DAOPHOT output is assumed to be in units of mag.
    # If the units are counts or flux, then you should change this to DIVIDE
    #   by the calibration light curve, not subtract.

    clightcurves = np.zeros_like(lightcurves)
    cal_hjd, cal_mag, cal_merr = cal_lc

    for i,k in enumerate(lightcurves):

        hjd, mag, merr = k
        cmag  = mag - cal_mag # Subtract calibration light curve
        cmerr = (merr**2 + cal_merr**2)**0.5

        clightcurves[i] = np.array([hjd, cmag, cmerr], dtype=np.float64)

    return(clightcurves)

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''

def flatten_lc(clightcurves, order, limiting_merr):

    # Detrends each light curve by fitting and subtracting a 3rd-order
    #   polynomial using numpy.polyfit.
    #
    # numpy.polyfit does not like numbers to be very large, such as HJD
    # To get around this, I subtract the minimum observed HJD from each
    #   data point, then multiply by 24 to convert from days to hours.
    #
    # If your data already uses small enough numbers for numpy.polyfit,
    #   then you will need to comment out the two lines that I use to convert
    #   my times to hours.

    flightcurves = np.zeros_like(clightcurves)

    for i,k in enumerate(clightcurves):
        hjd, mag, merr = k
        mhjd = min(hjd) ; hjd -= mhjd # Subtract minimum HJD
        hjd *= 24.                    # Convert days to hours

        hjd2 = np.copy(hjd)
        mag2 = np.copy(mag)
        merr2 = np.copy(merr)

        # Perform [iter] iterations of [nsig]-sigma clipping
        # Perform 3 iterations of 3-sigma clipping
        # Additionally ignores data points with uncertainty > limiting_merr
        iter = 3
        nsig = 3
        for k in range(iter):
            std = np.std(mag2)
            med = np.median(mag2)
            mask = np.where((mag2 < med+nsig*std) &
                              (mag2 > med-nsig*std) &
                              (merr2 < limiting_merr))
            hjd2 = hjd2[mask]
            mag2 = mag2[mask]
            merr2 = merr2[mask]

        # Fit the sigma-clipped data with an [order]-order polynomial
        polfit = np.polyfit(hjd2, mag2, order, rcond=None, full=False)

        # Build the best-fit polynomial
        fit = np.zeros_like(mag)
        for k in range(0, order+1):
            fit = fit + np.multiply(np.power(hjd, (order-k)), polfit[k])

        # Subtract the fit from each light curve and recenter each light curve
        #   back to their median values.
        mag3 = mag - fit + med

        flightcurves[i] = np.array([hjd/24. + mhjd, mag3, merr],
                                   dtype=np.float64)

    return(flightcurves)

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''

def save_lightcurves(lightcurves, limiting_merr):

    # Create an output file containing the light curve data.
    # Does not include data points with uncertainty > limiting_merr
    # Creates one file per star named 'starN_data.txt'
    # Star1_data.txt is the first light curve, and assumed to be the
    #   target of interest.
    # The remaining stars are calibration field stars used to create the
    #   calibration light curve.

    for i,k in enumerate(lightcurves):
        ofilename = f'star{i+1}_data.txt'
        with open(ofilename,'w') as ofile:
            for j in k.T:
                if j[2] >= limiting_merr:
                    continue
                ofile.write(f'{j[0]}    {j[1]}    {j[2]}\n')

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''

def plot_lc(lightcurves, star, limiting_merr):

    # Creates a plot of one or all lightcurves at once.
    # Assumes times are in HJD days and converts those to:
    #   "hours since first image was taken"
    # Comment out my conversion lines if your times are not in HJD or if you
    #   don't want your plot to be in hours on the x-axis.
    #
    # Does not plots points with uncertainty > limiting_merr


    fig, ax = plt.subplots(1,1,figsize=(12,4))

    if star == 'all' or star == 0:

        # Plot all stars at once on the same plot.
        # Not very useful usually.

        for i,k in enumerate(lightcurves):
            hjd, mag, merr = k

            mask = np.where(merr < limiting_merr)

            hjd = hjd[mask]
            mag = mag[mask]
            merr = merr[mask]

            hjd -= min(hjd)  # Subtract time of first image
            hjd *= 24.       # Convert HJD days to hours

            label = f'Star #{i+1}'
            ax.errorbar(hjd,mag,yerr=merr,marker='.',
                         markersize=3,elinewidth=1,capsize=2,alpha=0.5,
                         linestyle='None',label=label)
    else:
        # Plot a single star's light curve by itself.
        hjd, mag, merr = lightcurves[star-1]
        mask = np.where(merr < limiting_merr)

        hjd = hjd[mask]
        mag = mag[mask]
        merr = merr[mask]

        hjd -= min(hjd)  # Subtract time of first image
        hjd *= 24.       # Convert HJD days to hours

        label = f'Star #{star}'
        ax.errorbar(hjd,mag,yerr=merr,color='black',marker='.',
                     markersize=3,elinewidth=1,capsize=2,alpha=0.5,
                     linestyle='None',label=label)

    ax.invert_yaxis()
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Relative Magnitude')
    plt.legend()
    plt.show()

'''
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
'''


photfile = 'phot_file_example1.txt' # Filename of output from txdump

txdump = np.loadtxt(photfile,unpack=False,dtype=str) # As string for 'INDEF'

# Count the number of stars in the input phot_file
nstars,npts  = count_stars(txdump)

# Split up the input file into multiple subarrays holding one LC each
lightcurves  = split_stars(txdump,nstars,npts)

# Create the weighted-mean calibration light curve
cal_lc       = mk_cal_lc(np.copy(lightcurves[1:]))

# Apply the calibration light curve to each individual light curve
# Assumes magnitude units so it subtracts the calibration LC.
# You'll need to modify this to DIVIDE if you're using flux or counts
clightcurves = apply_cal_lc(lightcurves, cal_lc)

# Detrend/flatten the calibrated light curves.
# Fits fits a polynomial of order={order} and subtracts the best fit
# order=3 or order=2 are usually good.
# If your target shows strong sinusoidal motion, then order=1 might be best.
#   phot_file_example3.txt shows strong sinusoidal motion for example.
#   Try that example with order=3 and order=1 to see what I mean.
flightcurves = flatten_lc(clightcurves, order=3, limiting_merr=3)

# Save light curve files for each flattened, calibrated light curve
save_lightcurves(flightcurves, limiting_merr=3)

# Plot the lightcurve(s)
# star=0 or star='all' plots all LCs on the same plot
# star=1 only plots star1, etc
plot_lc(flightcurves, star=0, limiting_merr=2)
