import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from astropy.io import fits

def read_input(ifilename):
    #
    # Read the input photometry file.
    # This will be just the txdump output from IRAF for now.
    #
    times,fluxes,errors = np.loadtxt(ifilename,unpack=True,dtype=str)
    unique_times = np.unique(times)
    n_datapoints = len(unique_times)
    n_stars = int(len(times)/n_datapoints)

    column_headers = ["time"]
    for k in range(n_stars):
        column_headers.append(f"flux{k}")
        column_headers.append(f"error{k}")

    df = pd.DataFrame(index=range(n_datapoints),columns=column_headers)
    df['time'] = unique_times.astype(np.float64)

    for k in range(n_stars):
        flux = fluxes[k::n_stars].astype(np.float32)
        df[f'flux{k}'] = flux - np.median(flux)
        df[f'flux_error{k}'] = errors[k::n_stars].astype(np.float32)

    return(df, n_stars)

def make_weighted_mean_lc(df, n_stars):
    #
    # Create a weighted-mean calibration light curve using each of the calibration stars, but not the main star of interest.
    #
    fluxes = np.array([df[f'flux{k+1}'] for k in range(n_stars-1)])
    weights = np.array([1/(df[f'flux_error{k+1}'])**2 for k in range(n_stars-1)])
    weighted, weighted_error = np.average(fluxes, axis=0, weights=weights,returned=True)
    weighted_error = weighted_error**-0.5

    df['cal_lc'] = weighted
    df['cal_lc_error'] = weighted_error

    return(df)

def calibrate_flux(df, n_stars):
    #
    # Create columns for the calibrated flux values. Using magnitudes for now.
    #
    for k in range(n_stars):
        df[f'cal_flux{k}'] = df[f'flux{k}'] - df['cal_lc']
        df[f'cal_flux_error{k}'] = (df[f'flux_error{k}']**2 + df['cal_lc_error']**2)**0.5

    return(df)

def apply_airmass_lc(df, n_stars, order=0):
    #
    # Use the calibrated calibration star light curves to determine the effect of airmass on the general sky position
    # The effect of airmass changes depending on the color of an object, so its best to use only stars with a similar color to yours for this.
    # Using astroquery to look up magnitudes on the fly would be a nice upgrade, but it requires a WCS solution for the CCD to work.
    #
    # Apply the airmass correction by simply subtracting the fit from each lightcurve.
    #
    fluxes = np.array([df[f'cal_flux{k+1}'] for k in range(n_stars-1)])
    weights = np.array([1/(df[f'cal_flux_error{k+1}'])**2 for k in range(n_stars-1)])
    weighted, weighted_error = np.average(fluxes, axis=0, weights=weights,returned=True)
    weighted_error = weighted_error**-0.5

    polfit = np.polyfit(x=df['time'], y=weighted, w=1/weighted_error, deg=order, rcond=None, full=False)
    fit = np.zeros_like(df['time'])
    for k in range(0, order+1):
        fit = fit + np.multiply(np.power(df['time'], (order-k)), polfit[k])
    df['airmass_fit'] = fit

    for k in range(n_stars):
        df[f'airmass_cal_flux{k}'] = df[f'cal_flux{k}'] - df['airmass_fit']

    return(df)

def save_lightcurve(df, n_stars):
    #
    # Save a 3-column .txt file with the light curve.
    #

    ofilename = "output_lc.txt"
    with open(ofilename, 'w') as ofile:
        for i in range(len(df['time'])):
            line = f"{df['time'][i]} {df['airmass_cal_flux0'][i]} {df['cal_flux_error0'][0]}\n"
            ofile.write(line)



def plot_secondary_x(x):
    return(24*(x-df['time'][0]))

def plot_secondary_x_invert(x):
    return(x/24+df['time'][0])

def plot_lightcurve(df, n_stars):
    #
    # Create a simple plot showing final light curves
    #
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.secondary_xaxis('top',functions=(plot_secondary_x, plot_secondary_x_invert))
    ax2.set_xlabel('Time (hours)', fontsize='large')

    if False:
        plt.errorbar(df['time'],df['airmass_cal_flux0'], df['cal_flux_error0'],color='black',marker='.',linestyle='None',elinewidth=1,capsize=2,markersize=5)
        plt.gca().invert_yaxis()
        plt.ylabel("Relative Magnitude")
    else:
        plt.errorbar(df['time'],df['tflux0'], df[f'tflux_error0'],color='black',marker='.',linestyle='None',elinewidth=1,capsize=2,markersize=5)
        plt.ylabel("Relative Flux")

    plt.xlabel("BJD_TDB (days)")
    plt.savefig("output_lc.jpg", dpi=100)

    plt.show()

def mag2flux(df, n_stars):
    #
    # Convert relative magnitude to relative flux
    #
    # m1 - m2  =  -2.5*log10(F2/F1)
    #
    # F2/F1 = 10**(-(m1-m2)/2.5)
    #
    df[f'tflux0'] = 10**(-df['airmass_cal_flux0']/2.5)
    df[f'tflux_error0'] = df['cal_flux_error0']*df['tflux0']*np.log(10)/2.5
    return(df)



if __name__ == "__main__":
    #
    #
    #
    # Read the data into a pandas dataframe. One column per stellar flux, HJD/MJD saved to a separate column.
    ifilename = "txdump_mag.txt"
    df, n_stars = read_input(ifilename)

    # Create a weighted-mean light curve flux column using all of the calibration stars.
    df = make_weighted_mean_lc(df, n_stars)

    # Create new calibrated stellar flux columns, separate from the original columns in the input file.
    df = calibrate_flux(df, n_stars)

    # Create a weighted post-calibration light curve using only the field stars. This is used as an estimate of the effect of airmass
    # Fit a polynomial to the post-calibration light curve to obtain an airmass correction.
    # Apply the airmass correction factor to the first star in a new column.
    df = apply_airmass_lc(df, n_stars, order=1)

    # Convert relative magnitudes to relative flux
    df = mag2flux(df, n_stars)

    # Save an output file.
    save_lightcurve(df, n_stars)

    # Plot the final output.
    plot_lightcurve(df, n_stars)
