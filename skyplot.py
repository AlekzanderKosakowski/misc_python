# Plot a list of objects on a mollweide projection of the sky.
# Requires RA and Declination to be in decimal degrees.
# Allows each object to be plotted in a different color, based on an optional
#   filter keyword argument.
# filter should have the same length as ra and dec.
# Called with skyplot(ra,dec) or skyplot(ra,dec,filter)

import numpy as np
import matplotlib.pyplot as plt

def skyplot(ra,dec,filter=[]):

    # Define colors to plot for each filter
    # Dictionary key is the string being searched for in optional 'filter'
    #   argument. filter should be a list or array with the same length as
    #   ra and dec.
    # Dictionary value is the color you want that filter to show up as in the
    #   the figure.
    #   See: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    colors = {'u':    'tab:blue',
              'zg':   'g',
              'zr':   'y',
              'zi':   'tab:orange',
              'z':    'red',
              'bg40': 'limegreen'}

    ra  = np.array(ra,dtype=np.float64)
    dec = np.array(dec,dtype=np.float64)

    # Wrap RA around the projection if larger than 180 degrees.
    for k in range(len(ra)):
        if ra[k]>180.0:
          ra[k] -= 360.0
    ra = np.array(ra,dtype=float)
    dec = np.array(dec,dtype=float)

    fsize = (12,6) # Figure dimensions in inches
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111, projection="mollweide")
    if len(filter) == len(ra): # Use special colors if filters provided.
        ax.scatter(np.radians(ra), np.radians(dec),marker=".",color=[colors[k] for k in filter], s=10, alpha=0.34)
    else: # Use color='black' if no filters provided.
        ax.scatter(np.radians(ra), np.radians(dec),marker=".",color="black", s=10, alpha=0.34)
    ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
    ax.grid(True)
    plt.show()
