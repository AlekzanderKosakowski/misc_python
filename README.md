# misc

binning.py: Average-bin light curve data. Accepts [time,mag,merr], [time,mag], and their transpose formats.

convert_wavelength.py: Convert wavelength values between their vacuum and air values. Requires Angstrom input units.
                       (Source: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion )

Coordinates.py: Convert between Right Ascension and Declination coordinate formats (HH:MM:SS.ss vs Decimal Degrees)

corner_dist.ipynb: Create fancy corner plot. Not very efficient, doesn't look good, and seaborn does it better, faster, and cleaner. This is only here for reference now.

eleanor_astropy.py: Use eleanor (https://adina.feinste.in/eleanor/) to obtain TESS FFI light curves. Process those lightcurves using astropy.timeseries.LombScargle to search for variability. Estimate amplitude and frequency uncertainty using multi-processed bootstrapping. While it is very easy to modify this code to take a list of targets instead of a single target, I don't recommend it because the code isn't written in a memory efficient way. This code doesn't appear to release memory after each loop, causing it to quickly hoard all of your available memory. It was much slower, but more memory efficient, to modify this code to use sys.argv[] for targetdec, ra, dec and throw it in a bash loop or awk-generated script instead.

plot_phased.py: Plot phased-lightcuve with model over two phases. Separate panels zoomed into phase=0.5 and phase=1.0 for secondary and primary eclipses.

skyplot.py: Plot the locations of astronomical objects on a mollweide projection of the sky. Allows quick multi-color plotting based on optional filter argument.
