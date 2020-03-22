# misc
binning.py: Average-bin light curve data. Accepts [time,mag,merr], [time,mag], and their transpose formats.

convert_wavelength.py: Convert wavelength values between their vacuum and air values. Requires Angstrom input units.
                       (Source: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion )

Coordinates.py: Convert between Right Ascension and Declination coordinate formats (HH:MM:SS.ss vs Decimal Degrees)

corner_dist.ipynb: Create fancy corner plot. Not very efficient.

skyplot.py: Plot the locations of astronomical objects on a mollweide projection of the sky. Allows quick multi-color plotting based on optional 
filter argument.

plot_phased.py: Plot phased-lightcuve with model over two phases. Separate panel zoomed into phase=1.0 for eclipses.
