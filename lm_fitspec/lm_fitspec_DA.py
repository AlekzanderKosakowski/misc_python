import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font) # https://matplotlib.org/stable/gallery/text_labels_and_annotations/fonts_demo.html
from astropy.io import fits
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
import os
import sys
from numba import njit


'''
Personal code used to perform Levenberg-Marquardt minimization to DA white dwarf spectra from FITS files.
This code was inspired by a similar private Fortran code.

Model files used are private and are not available upon request. You'll need to create or find your own public models to use.
This means you'll also need to write your own "load_models()" function to put them in a similar format for the rest of the code to work.
'''

def load_fitsdata(filename):
    #
    # Use astropy.io.fits to load spectral data from a FITS file.
    # Use the header to determine wavelength values.
    # Assumes relevant data and header are located in HDU0.
    #   This may not be true.
    #   If you get an error about data shape, then check the data dimensions by uncommenting the lines: "print(fitsfile.info()) ; sys.exit()"
    #   For example: You may get the error: "ValueError: operands could not be broadcast together with shapes (4,) (4,1,4086)"
    #     You'll need to change "fitsfile[hdu].data" to "fitsfile[hdu].data[0][0]" or something similar
    #
    #   x = np.array of observed wavelength data
    #   y = np.array of observed spectral data
    #   e = np.array of observed errorbars. Errorbars not actually present in spectra. Just set all to 1 for equal weighting.
    #
    hdu = 0
    fitsfile = fits.open(filename)
    y_lambda = np.array(fitsfile[hdu].data[0][0], dtype=np.float64)

    # print(fitsfile.info()) ; sys.exit()

    x0 = fitsfile[hdu].header['CRVAL1'] # Based on SOAR Goodman Blue header
    dx = fitsfile[hdu].header['CD1_1']  # Based on SOAR Goodman Blue header
    x = np.array([x0 + k*dx for k in range(len(y_lambda))], dtype=np.float64)

    y_nu = flambda_to_fnu(x, y_lambda) # Convert data in F-lambda to F-nu

    # Weight each point equally.
    e = np.ones_like(y_nu)

    target = fitsfile[hdu].header['OBJECT']

    return(x, y_nu, e, target)

def flambda_to_fnu(x, y_lambda):
    #
    # Spectra are typically in units of erg/cm**2/s/Angstrom (F-lambda).
    # This function converts the data into erg/cm**2/s/Hz (F-nu)
    #
    c = 299792458*1e2 # Speec of light in cgs
    y_nu = (x**2)/c*y_lambda
    return(y_nu)


def split_data(x, y, include):
    #
    # Split the observed data into the appropriate Hydrogen Balmer lines defined by the user in the [include] array.
    # Includes call to a function that determines wavelength shifts
    # This function is designed specifically for observed data; not models.
    #
    # H_lines: Global dictionary of wavelength regions to use when trimming specific lines from the data.
    #    x, y: Observed wavelengths and fluxes
    #  xt, yt:  Trimmed wavelengths and fluxes
    #  shifts: Fitted wavelength shifts for individual lines
    #       a: Normalized and trimmed wavelengths and fluxes. Has format [[x], [y]]
    #
    global H_lines

    # Create a list of indices of Balmer lines being fit, 0=halpha, 1=hbeta, 4=hepsilon, etc.
    # Only used to allow njit with the split_models() function. Creating it here for convenience.
    include_i = [] ; i = 0
    for j,k in H_lines.items():
        if j in include:
            include_i.append(i)
        i += 1

    fit_shifts = ['halpha', 'hbeta', 'hgamma', 'hdelta']                    # Use only these lines for velocity fitting
    shifts = np.zeros(np.count_nonzero([k in fit_shifts for k in include])) # Wavelength shifts per line. Will median later and apply a uniform shift

    for l in range(2):                  # Two passes: Once to determine wavelength shift, then once again after the shift is applied. Required to do this at least twice or the models may end up with different dimensions than the data.
        shift = np.median(shifts)       # shift=0 for the first pass.
        x -= shift                      # Apply the wavelength correction.
        xt, yt = np.array([]), np.array([])
        for i,k in enumerate(include):
            index = np.ravel(np.where(  (x>=H_lines[k][0]) & (x<=H_lines[k][1]) ))
            a1,a2,_ = normalize_line(x[index], y[index], i)
            if k in fit_shifts:
                shifts = find_line_center(a1, a2, i, k, shifts)
            xt = np.concatenate( (a1, xt))
            yt = np.concatenate( (a2, yt))

    print(f"\nShifted observed wavelength values by {shift} A based on a multi-Gaussian fit to a few line centers.")

    return(xt, yt, np.array(include_i), shift)

def find_line_center(x, y, i, line, shifts):
    #
    # Fit a triple-Gaussian profile to the lines to find the wavelength shift due to systemic velocities
    # Return a single constant wavelength shift value to be applied to all wavelengths
    # Fits the sum of 3 Gaussians with varaible parameters based on which line is being fit.
    # The initial parameters for the fits are:
    #       mu = Constant line center based on the median of the wavelength range being considered
    #     sig1 = 1/8 of the total width of the wavelength range being considered
    #     sig2 = 1/4 of the total width of the wavelength range being considered
    #     sig3 = 1/2 of the total width of the wavelength range being considered
    #     amp1 = 0.5   (50% absorption depth)
    #     amp2 = 0.25  (25% absorption depth)
    #     amp3 = 0.125 (12.5% absorption depth)
    #
    # From Wikipedia "Balmer Series"
    H_center_air   = {"halpha":    6562.79,
                      "hbeta":     4861.35,
                      "hgamma":   4340.472,
                      "hdelta":   4101.734,
                      "hepsilon": 3970.075}

    # Using equations from [http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion] to convert from air to vacuum
    H_center_vac   = {"halpha":   6564.602977311409,
                      "hbeta":       4862.708025294,
                      "hgamma":   4341.692323417887,
                      "hdelta":   4102.891633248132,
                      "hepsilon": 3971.198206170563}

    try:
        # Try to fit a sum of three Gaussians to get line centers values.
        popt, pcov = curve_fit(gauss3, x, y, p0=[np.median(x), 0.5, (x[-1]-x[0])/8, 0.25, (x[-1]-x[0])/4, 0.125, (x[-1]-x[0])/2])
        model = gauss3(x, *popt)
    except:
        # If three Gaussians fails, do two instead.
        popt, pcov = curve_fit(gauss2, x, y, p0=[np.median(x), 0.5, (x[-1]-x[0])/8, 0.25, (x[-1]-x[0])/4])
        model = gauss2(x, *popt)


    # Plot each line being fit to check how well it performs.
    # plt.plot(x,y)
    # plt.plot(x,model)
    # plt.show()

    shifts[i] = (popt[0] - H_center_air[line])
    return(shifts)

def gauss3(x, mu, a1, sig1, a2, sig2, a3, sig3):
    #
    # Model used to find line centers.
    # Simple sum of three Gaussians with the same center.
    #
    a = [a1, a2, a3]
    sig = [sig1, sig2, sig3]

    g = 1
    for i in range(len(a)):
        g -= a[i]*np.exp(-0.5*((x-mu)/sig[i])**2)
    return(g)

def gauss2(x, mu, a1, sig1, a2, sig2):
    #
    # Model used to find line centers.
    # Simple sum of two Gaussians with the same center.
    #
    a = [a1, a2]
    sig = [sig1, sig2]

    g = 1
    for i in range(len(a)):
        g -= a[i]*np.exp(-0.5*((x-mu)/sig[i])**2)
    return(g)

def get_ignore_indices(xt, block_calcium3933, block_helium4026):
    #
    # Use np.where() to determine indices in the data where known absorption lines may be present.
    # If the user specified to ignore specific lines, then assign "error-bars" of 1e30 to essentially force the fit to ignore those data points.
    # Eventually add 'auto' as an option to block_calcium3933 and other lines to try to fit a feature at these locations.
    #   If the feature successfully fits, then deweight those points, if not then don't deweight.
    #
    if block_calcium3933:
        calcium_index = np.ravel(np.where( (xt>=3923.) & (xt<=3943.) ))
    else:
        calcium_index = np.ravel(np.where(x<0.))
    if block_helium4026:
        helium_index = np.ravel(np.where( (xt>=4020.) & (xt<=4032.) ))
    else:
        helium_index = np.ravel(np.where(x<0.))

    ignore_indices = np.concatenate((calcium_index, helium_index))
    return(ignore_indices)

def load_models(convolution, model_path):
    #
    # Load the collection of model spectra
    # This function depends entirely on the format of the model spectra.
    #   Therefore, this function will likely need to be entirely rewritten for different model file formats.
    #   This code is based on model files with the following format:
    #     One file per gravity value.
    #       The top line is a single string label and a gravity value where np.log10(value) is the logg  (i.e. "Gravity:  3.162E+04" is the only text on the first line)
    #       The 2nd line is split into N+1 columns: the first column is a space-filler header for wavelength, followed by one column for each of the N model temperatures: ("wave\teff 1000.0 1500.0 2000.0 2500.0 3000.0 5777.0 12500.0" etc)
    #       Every line afterwards contains wavelength in the first column, followed by N columns containing the flux at that wavelength and temperature corresponding to the column header and gravity specified in the file's first line.
    #         For example: "4333.46   2.83271E-07   4.77768E-07   7.79092E-07   1.18832E-06   1.71210E-06   2.37173E-06   3.16232E-06" etc
    #
    #   model_files: list of text files containing model spectrum data
    #        mgravs: Array containing the log10(g) values being used in the fitting.
    #        mteffs: Array containing the Teff values being used in the fitting.
    #    temp_mdata: temporary array holding all of the information in a single model file.
    #   mwavelength: Wavelength grid used in the model files.
    #    temp_mflux: temporary array of STRINGS holding the collection of all fluxes for a single model file (one log(g), all Teff)
    #                Includes typos generated from the original fortran formatting (1.234-100 should be 1.234E-100).
    #         mflux: Corrected array of np.float64 containing all fluxes for a single model file
    #    model_list: Array containing all flux data for all model files.
    #                Has the format: [log(g), Teff, wavelength]
    #                For example: if you wanted the entire 1D spectrum for a specific model, you'd use "model_list[12][15]" and use mwavelengths as your x-values
    #
    model_files = sorted([k for k in os.listdir(model_path) if f"ML1.8_c{convolution}" in k]) # You will need to change the "ML1.8_c" part if you are using different models.

    mgravs = np.empty_like(model_files)
    for i,file in enumerate(model_files):
        mfilename = f"{model_path}/{file}"
        with open(mfilename, 'r') as mfile:
            # print(f"Loading model file {mfilename}")
            mgravs[i] = np.log10(float(mfile.readline().split()[1]))
            if i == 0:
                mteffs = np.array(mfile.readline().split()[1:], dtype=np.float64)

            temp_mdata = np.loadtxt(mfilename, unpack=True, skiprows=1, dtype=str)

            mwavelength = np.array(temp_mdata[0][1:], dtype=np.float64)

            # Get flux values from a single log(g) file for all temperatures.
            # Correct typos generated from fortran formatting (1.234-100 is corrected to 1.234E-100).
            temp_mflux = np.array(temp_mdata[1:].T[1:].T)
            if i == 0:
                mflux = np.zeros( (len(mgravs), len(temp_mflux), len(temp_mflux[0])) )
            for t in range(len(temp_mflux)):
                mflux[i][t] = np.array([k if 'E' in k else k[:-4] + 'E' + k[-4:] for k in temp_mflux[t]])
            mflux = np.array(mflux, dtype=np.float64)

    mgravs = np.array(mgravs, dtype=np.float64)

    return(mgravs, mteffs, mwavelength, mflux)

def interp_models(x, mgravs, mteffs, mwavelength, mflux):
    #
    # Interpolate the models to the observed wavelength grid.
    # This is separate from the load_model() function to allow for multiple-targets to be run in sequence without having to reload all models every time.
    # This is because different targets will have a slightly different observed wavelength grid, so the load_model() function should be independent of the observed wavelength grid.
    #
    for i in range(len(mgravs)):
        if i == 0: # Initialize the model_list array
            # Build model array of shape [ngrav, ntemp, nflux_obs]
            model_list = np.zeros((len(mgravs), len(mteffs), len(x)))

        for k in range(len(mflux[i])):
            # Using linear interpolation for now to create an interpolation object over wavelength and flux for all models.
            # The model files have a duplicate wavelength point that prevents the use of 'cubic' interpolation. Probably not terribly important.
            z = [interp1d(mwavelength, l, kind='linear') for l in mflux[i]]

            model_list[i,k] = z[k](x)

    return(model_list)

def split_model(x, y, include, dmdq):
    #
    # Split a model spectrum into the appropriate Hydrogen Balmer lines defined by the user in the [include] array.
    # This function is designed specifically for model spectra; not observed data.
    #
    # H_lines: Global dictionary of wavelength regions to use when trimming specific lines from the data.
    #    x, y: Model wavelengths and fluxes
    #  xt, yt:  Trimmed wavelengths and fluxes
    #       a: Normalized and trimmed wavelengths and fluxes. Has format [[x], [y]]
    #    dmdq: Array containing first partial derivatives to the model spectra. Two fit parameters, so two first-derivatives
    #
    global H_lines

    xt, yt = np.array([]), np.array([])
    dmdqt1, dmdqt2 = np.array([]), np.array([])
    for i,k in enumerate(include):
        index = np.ravel(np.where(  (x>=H_lines[k][0]) & (x<=H_lines[k][1]) ))
        a = normalize_line(x[index], y[index], i, dmdq[:,index], True)
        dmdqt1 = np.concatenate( ( a[2][0], dmdqt1 ) )
        dmdqt2 = np.concatenate( ( a[2][1], dmdqt2 ) )
        yt = np.concatenate( (a[1], yt))

    dmdqt = np.array([dmdqt1, dmdqt2], dtype=np.float64)

    return(yt, dmdqt)

@njit
def split_model_njit(x, y, include_i, dmdq):
    #
    # Split a model spectrum into the appropriate Hydrogen Balmer lines defined by the user in the [include] array.
    # This function is designed specifically for model spectra; not observed data.
    #
    # H_lines: Array of wavelength regions to use when trimming specific lines from the data.
    #    x, y: Model wavelengths and fluxes
    #      yt: Trimmed fluxes
    #      a2: 1D array containing the normalized and trimmed model flux
    #      a3: 2D array containing the normalized and trimmed first partial derivatives. Has format [[dmdq1], [dmdq2]]
    #    dmdq: Array containing first partial derivatives to the model spectra. Two fit parameters, so two first-derivatives
    #
    global njit_H_lines

    yt = np.array([0.])
    dmdqt1, dmdqt2 = np.array([0.]), np.array([0.])

    for i in include_i:
        index = np.where(  (x>=njit_H_lines[i][0]) & (x<=njit_H_lines[i][1]) )[0]
        _,a2,a3 = normalize_line(x[index], y[index], i, dmdq[:,index], True)
        dmdqt1 = np.concatenate((a3[0], dmdqt1))
        dmdqt2 = np.concatenate((a3[1], dmdqt2))
        yt = np.concatenate((a2, yt))

    yt = yt[:-1]

    dmdqt = np.zeros((2,len(dmdqt1)-1))
    dmdqt[0] = dmdqt1[:-1]
    dmdqt[1] = dmdqt2[:-1]

    return(yt,dmdqt)

def create_2dinterp_object(mteffs, mgravs, model_list):
    #
    # Create a list containing one scipy.interpolate.interp2d() object per wavelength point.
    # The 0th element is the 2D interpolation object for the first wavelength point, etc.
    #
    z  = [interp2d(mteff, mgravs, models[:,:,k], kind='cubic') for k in range(len(model_list[0][0]))]
    return(z)

def get_model(teff, logg, interp_2d_object):
    #
    # Interpolate over Teff and log(g) on the observed model grid to an arbitrary teff and log(g) and return its model spectrum.
    # Obtain partial derivatives along each parameter as well.
    # If the attempted logg or teff is outside of the model grid, DO NOT extrapolate. Assign new values at 5% away from the model grid edges and continue.
    # Extrapolation is super inaccurate and shouldn't be bothered with. I've seen a model return logg=21 even though the models stop at 9.5
    #   dmdq = partial derivative of model m with respect to parameter q
    #        = has shape [N_parameters, N_datapoints]
    #        = first element is first partial derivitive to the entire model spectrum with respect to parameter 1
    #
    try:
        model = np.ravel([k(teff, logg) for k in interp_2d_object])
    except ValueError:
        teff = mteffs[0]*1.05  if teff <= mteffs[0]  else teff
        teff = mteffs[-1]*0.95 if teff >= mteffs[-1] else teff
        logg = mgravs[0]*1.05  if logg <= mgravs[0]  else logg
        logg = mgravs[-1]*0.95 if logg >= mgravs[-1] else logg

        print(f"Attempted model is outside of the defined model interpolation range.\nTo avoid inaccurate extrapolation, assigning Teff={teff} and log(g)={logg}")
        model = np.ravel([k(teff, logg) for k in interp_2d_object])


    dmdq = np.zeros((2, len(model)))
    dmdq[0] = np.ravel([k(teff, logg, dx=1, dy=0) for k in interp_2d_object])
    dmdq[1] = np.ravel([k(teff, logg, dx=0, dy=1) for k in interp_2d_object])

    return(model, dmdq)

@njit
def normalize_line(x, y, i, dmdq=np.zeros((2,2)), derivative=False):
    #
    # Normalize a Hydrogen line by dividing by a straight line.
    # Corrects the derivatives as well.
    # This is not fitting a straight line. It just uses dy/dx = slope, etc
    #
    #     i: Iteration number representing which line is being fit. i=1 for halpha, i=2 for hbeta, etc
    #     n: number of points on each end of the line to average for x1, x2, y1, y2
    #      : Average 5 points for lines below H8. For H8+, use 3
    # x1,x2: average x-position of first/last n data points
    # y1,y2: average y-position of first/last n data points
    #     a: linear slope
    #     b: linear y-intercept
    #    da: derivative of slope with respect to fitted parameter
    #    db: derivative of y-intercept with respect to fitted parameter
    # dmdq2: Normalized derivative of the flux (calc1 chain-rule on "normalized = flux/line_fit" for equation used)
    #
    n = 10 if i <= 4 else 4
    x1, x2 = np.mean(x[:n]), np.mean(x[-n:])
    y1, y2 = np.mean(y[:n]), np.mean(y[-n:])

    a = (y2 - y1) / (x2 - x1)
    b = y1 - a*x1
    fit = a*x + b  # Straight line "fit", but not actually obtained through any sort of fitting.

    dmdq2 = np.zeros((2,len(y)))
    if derivative:
        for j in range(len(dmdq)):
            dy1, dy2 = np.mean(dmdq[j][:n]), np.mean(dmdq[j][-n:])
            da = (dy2 - dy1) / (x2 - x1)
            db = dy1 - da*x1
            dmdq2[j] = dmdq[j]/fit - y/(fit**2)*(da*x+db)

    return(x, y/fit, dmdq2)

@njit
def get_chi2(y, e, m):
    #
    # A simple chi-squared merit function
    #
    chi2 = np.sum(((y-m)/e)**2)
    return(chi2)

@njit
def get_beta(y, e, m, dmdq):
    #
    # Simple beta vector calculator based on chi-squared merit function.
    # Will need a different beta for a different merit function.
    # See "Numerical Recipes in C++ Second Edition" chapter 15 for details.
    #
    beta = np.zeros(2)
    for k in range(len(dmdq)):
        beta[k] = np.sum((y-m)/e*dmdq[k])
    return(beta)

@njit
def get_alpha(e, dmdq, alambda=0):
    #
    # Simple alpha matrix calculator based on chi-squared merit function.
    # Will need a different alpha for a different merit function.
    # See "Numerical Recipes in C++ Second Edition" chapter 15 for details.
    #
    alpha = np.zeros((2,2))
    for k in range(2):
        for l in range(2):
            alpha[k][l] = np.sum(dmdq[k]*dmdq[l]/e**2)
        alpha[k][k] *= (1+alambda)
    return(alpha)

def get_rms(y, m):
    #
    # Calculates the Root Mean-Squared based on the residuals of a fitted model.
    #
    rms = (np.sum((y - m)**2)/len(y))**0.5
    e = rms*np.ones_like(y)

    return(e)

def plot_solution(x, y, m, teff, eteff, logg, elogg, ax, j, shift):
    #
    # Create the final plot of stacked Balmer lines and their fits
    # To center each line, the code subtracts the air-wavelength line center for each line calculated from the Rydberg equation.
    #   See equation 2-113 of Astrophysical Formulae Second Corrected and Enlarged edition (Kenneth R. Lang)
    # Vacuum-wavelength line centers were corrected to air-wavelength using the equations from the website: "https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion" which uses equations from various publications.
    #
    global H_lines

    colors = ['crimson', 'crimson']

    vshift = 0
    for k,v in H_lines.items():
        index = np.ravel(np.where( (x>=v[0]) & (x<=v[1]) ))
        if len(index) == 0:
            continue
        ax[j].plot(x[index]-(v[2]+shift), y[index]+vshift, color='black', linewidth=1)
        ax[j].plot(x[index]-(v[2]+shift), m[index]+vshift, color=colors[j], linewidth=1)
        vshift += 0.5

    ax[j].set_xlabel("$\Delta\lambda$ ($\AA$)")
    ax[j].set_xticks([-100, -50, 0, 50, 100])
    title = f'{"Teff":>6s} = {teff:>5.0f} +/- {eteff:>5.0f}\n{"log(g)":>6s} = {logg:>5.3f} +/- {elogg:>5.3f}'
    ax[j].set_title(title,loc='left')

if __name__ == "__main__":

    filename = "fits_files/20170323_1012p1427_0096.ms.fits" # FITS file of the spectrum to fit

    model_path = '/Users/kastra/code/python/fitspec/python_grids_ELM/' # System location of the model files.
    convolution = 2.3 # Spectral resolution of the data.
    include = ['hgamma', 'hdelta', 'hepsilon', 'h8', 'h9', 'h10', 'h11', 'h12'] # Fit these lines
    teff0_list, logg0 = [8000., 20000.], 6.0  # Initial guesses for Teff and log(g)
    ignore_calcium3933 = True   # Deweight the Ca II absorption at 3933 angstrom?  True/False
    ignore_helium4026  = False  # Deweight the He I absorption at 4026 angstrom?   True/False

    # Wavelength range for each Hydrogen line. Line centers (in air) from Rydberg formula given as element [2] of array
    H_lines = {"halpha":   [6361., 6763., 6562.8821],
               "hbeta":    [4721., 5001., 4861.3791],
               "hgamma":   [4220., 4460., 4340.5092],
               "hdelta":   [4031., 4171., 4101.7768],
               "hepsilon": [3925., 4015., 3970.1121],
               "h8":       [3859., 3919., 3889.0876],
               "h9":       [3815., 3855., 3835.4220],
               "h10":      [3782., 3812., 3797.9350],
               "h11":      [3760., 3780., 3770.6671],
               "h12":      [3741., 3759., 3750.1883]}
    # Create a np.array version of the H_lines dictionary since njit doesn't allow dictionaries.
    njit_H_lines = np.zeros((len(H_lines),3))
    for i,k in enumerate(H_lines):
        njit_H_lines[i] = H_lines[k]


    # Load the data and trim it to the regions around the 'include' Balmer lines. Fit and apply a wavelength shift correction from radial velocity
    x, y, e, target = load_fitsdata(filename)
    xt, yt, include_i, shift = split_data(x, y, include)

    # Load all of the model files at once. Save an array of gravities and temperatures as well.
    mgravs, mteffs, mwavelength, mflux = load_models(convolution, model_path)

    # Create a list of models using the observed wavelength grid.
    # This is the point where you'd throw in a for loop if you were running this code on a list of files in sequence.
    model_list = interp_models(x, mgravs, mteffs, mwavelength, mflux)

    # For each wavelength point, create the 'cubic' spline interpolation function using all models.
    # See "Numerical Recipes in C++ Second Edition" chapters 3 for details on Cubic Spline interpolation
    interp_2d_object = [interp2d(mteffs, mgravs, model_list[:,:,k], kind='cubic', bounds_error=True) for k in range(len(model_list[0][0]))]

    fig, ax = plt.subplots(nrows=1,ncols=len(teff0_list), sharex=True, sharey=True, figsize=(11.3, 7.1))
    plt.subplots_adjust(hspace=0, wspace=0)
    for j,t in enumerate(teff0_list):

        teff = t
        logg = logg0

        et = np.ones_like(yt) # Originally use ones as "error bars". Use RMS for final solution to get "correct" uncertainties
        # Determine which indices are to be weighted so low that they're essentially ignored in the fits.
        ignore_indices = get_ignore_indices(xt, ignore_calcium3933, ignore_helium4026)
        et[ignore_indices] += 1e30

        # Use the interpolated function to obtain a model and its derivatives at teff and logg.
        m, dmdq = get_model(teff, logg, interp_2d_object)

        # Trim the model and its derivatives to match the observed data, keeping only data surrounding the Balmer lines in 'include'
        mt, dmdqt  = split_model(x, m, include, dmdq)

        # Use a chi-squared merit function to quantify how well the data fits the model. Initialize alambda for iterative methods later.
        old_chi2 = get_chi2(yt, et, mt) ; alambda = 0.001
        print(f'{"Old_Chi2":>15s}{"lambda":>15s}{"dTeff":>15s}{"dlogg":>15s}{"New_Teff":>15s}{"New_logg":>15s}{"New_Chi2":>15s}')
        print(f'{"":>15s}{"":>15s}{"":>15s}{"":>15s}{teff:>15.0f}{logg:>15.3f}{old_chi2:>15f}')

        # Use Levenberg-Marquardt method to determine the best-fitting model parameters (see "Numerical Recipes in C++ Second Edition" chapter 15 for details)
        converged = False
        bcount = 0
        while not converged:

            beta  = get_beta(yt, et, mt, dmdqt)
            alpha = get_alpha(et, dmdqt, alambda)

            da = np.linalg.solve(alpha, beta)
            dteff, dlogg = da
            # Sometimes the change in parameters is so large that it skips over the global minimum and settles near an incorrect local minimum.
            # If the change in Teff is >5000 K or logg is > 1.0, then reduce it by 90% and continue. I assume this is enough to allow escaping from local minima while avoiding large jumps out of the global minimum.
            if abs(dteff) >= 5000:
                dteff *= 0.1
            if abs(dlogg) >= 1.0:
                dlogg *= 0.1

            # Do not allow the algorithm to attempt a model outside of the model grid parameter range.
            while teff+dteff >= mteffs[-1] or teff+dteff < mteffs[0]:
                dteff *= 0.1
            while logg+dlogg >= mgravs[-1] or logg+dlogg < mgravs[0]:
                dlogg *= 0.1

            m2, dmdq2 = get_model(teff+dteff, logg+dlogg, interp_2d_object)
            m2t, dmdq2t = split_model(x, m2, include, dmdq2)
            chi2 = get_chi2(yt, et, m2t)

            if chi2 < old_chi2:
                print(f'{old_chi2:>15f}{alambda:>15.2e}{dteff:>15.0f}{dlogg:>15.3f}{teff+dteff:>15.0f}{logg+dlogg:>15.3f}{chi2:>15f}')
                bcount = 0
                teff += dteff ; logg += dlogg
                alambda /= 2
                if abs(chi2 - old_chi2) < 0.01:    # Convergence criterion in chi2
                    if dlogg < 0.001 and dteff < 10: # Convergence criterion in logg and teff
                        converged = True ; alambda = 0
                old_chi2 = chi2
                m = np.copy(m2) ; mt = np.copy(m2t)
                dmdq = np.copy(dmdq2) ; dmdqt = np.copy(dmdq2t)
            elif chi2 > old_chi2:
                alambda *= 2
                bcount += 1
                if bcount > 100:
                    # If bad count > N fits in a row, then assume converged because you're clearly not finding a better solution anytime soon.
                    print(f"Failed to find a better solution after {bcount} attempts. Assuming converged.")
                    converged = True ; alambda = 0
                    teff += dteff ; logg += dlogg
                    old_chi2 = chi2
                    m = np.copy(m2) ; mt = np.copy(m2t)
                    dmdq = np.copy(dmdq2) ; dmdqt = np.copy(dmdq2t)
            elif chi2 == old_chi2:
                print(f"No change in chi-squared. Assuming converged.")
                converged = True ; alambda = 0


        et = get_rms(yt, mt) # Estimate parameter uncertainty based on RMS of the residuals to the data and the best-fitting model
        et[ignore_indices] += 1e30 # Re-ignore indices corresponding to common non-Hydrogen lines like Ca II 3933 or He I 4026
        beta  = get_beta(yt, et, mt, dmdqt)
        alpha = get_alpha(et, dmdqt,)
        covar = np.linalg.inv(alpha)
        eteff, elogg = covar[0][0]**0.5, covar[1][1]**0.5
        print(f'\nFinal Solution:\n{"Teff":>6s} = {teff:>5.0f} +/- {eteff:>5.0f}\n{"log(g)":>6s} = {logg:>5.3f} +/- {elogg:>5.3f}\n{"chi2":>6s} = {chi2:>5.3f}\n')

        plot_solution(xt, yt, mt, teff, eteff, logg, elogg, ax, j, shift)

    # plt.suptitle(f'{filename}')
    plt.show()
    # plt.savefig(f"{target}.png")
