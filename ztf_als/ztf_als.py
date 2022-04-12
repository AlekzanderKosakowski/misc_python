from median_combine import median_combine
from als import run_als
from fancy_plot import fancy_plot
from get_phase import get_phase
from get_ztf10_lc import get_ztf_lc
from Coordinates import RA2Decimal, Dec2Decimal, Decimal2RA, Decimal2Dec

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def run_periodograms(coords):
  #
  # Run Astropy Lomb Scargle.
  # Will add forced-photometry Box Least Squares later.
  #
  ra, dec = coords

#  print(ra, dec)

  skip_completed = True
  if "".join([str(ra),str(dec)]) in completed and skip_completed:
    print(f"Skipping {ra} {dec}; target already completed.")
    return()


  data = get_ztf_lc(ra, dec, False)
  if len(data[0]) < 30: # No ZTF data found or not enough ZTF data found. Require at least 30 data points per object.
    return

  mdata = median_combine(data.T)

  minp = 3./1440. # Minimum period to search (in days)
  maxp = 684./1440. # Maximum period to search (in days)
  nfreq = 5_000_000 # Number of frequency points to search.
  fgrid = np.linspace(1/maxp, 1/minp, nfreq) # Evenly-spaced frequency grid.
  power, peak_freq, peak_power, times, model, amp = run_als(mdata[0], mdata[1], mdata[2], fgrid)

  # Conditions for saving an output figure.
  # The noise level changes with frequency, so we hard-code the noise level here for different frequency ranges.
  save_fig = False
  if (peak_power >= 0.75 and peak_power <= 1.0 and np.median(power) <= 0.15):
    save_fig = True
  elif 24./peak_freq < 2:
    if(peak_power / np.mean(power) >= 18.0):
      save_fig = True
  elif 24./peak_freq < 3:
    if(peak_power / np.mean(power) >= 28.0):
      save_fig = True
  elif 24./peak_freq < 4:
    if(peak_power / np.mean(power) >= 38.0):
      save_fig = True
  elif 24./peak_freq < 5:
    if(peak_power / np.mean(power) >= 48.0):
      save_fig = True
  elif 24./peak_freq < 6:
    if(peak_power / np.mean(power) >= 58.0):
      save_fig = True
  elif 24./peak_freq < 7:
    if(peak_power / np.mean(power) >= 48.0):
      save_fig = True
  elif 24./peak_freq < 7.85: # Between ~7.85h and 8.1h, the ZTF observation sampling creates a huge spike in the power spectrum that produces a ton of junk results, eating up computation time to save junk figures.
    if(peak_power / np.mean(power) >= 38.0):
      save_fig = True
  elif 24./peak_freq < 7.95:
    if(peak_power / np.mean(power) >= 43.0):
      save_fig = True
  elif 24./peak_freq < 7.98:
    if(peak_power / np.mean(power) >= 148.0):
      save_fig = True
  elif 24./peak_freq < 8.0:
    if(peak_power / np.mean(power) >= 78.0):
      save_fig = True
  elif 24./peak_freq < 8.01:
    if(peak_power / np.mean(power) >= 68.0):
      save_fig = True
  elif 24./peak_freq < 8.075:
    if(peak_power / np.mean(power) >= 38.0):
      save_fig = True
  elif 24./peak_freq < 8.085:
    if(peak_power / np.mean(power) >= 58.0):
      save_fig = True
  elif 24./peak_freq < 10:
    if(peak_power / np.mean(power) >= 58.0):
      save_fig = True
  elif 24./peak_freq < 12:
    if(peak_power / np.mean(power) >= 48.0):
      save_fig = True
  else:
    save_fig = False

  pdata = get_phase(mdata.T, peak_freq)

  # Save the .txt file
  # ra_degrees, dec_degrees, ra, dec, mag, mag_err, peak_freq, peak_power, mean_power, amplitude
  with open("als_output10.txt", 'a') as ofile:
    stats = f"{ra} {dec} {':'.join(Decimal2RA(ra))} {':'.join(Decimal2Dec(dec))} {np.median(pdata[1])} {np.std(pdata[1])} {peak_freq} {peak_power} {np.mean(power)} {amp}\n"
    ofile.write(stats)

  fancy_plot(pdata.T, ra, dec, fgrid, peak_freq, power, amp, save_fig)

if __name__ == "__main__":

  global completed
  if not os.path.isfile("als_output10.txt"):
    os.system("touch als_output10.txt")
  completed = ["".join(k) for k in np.loadtxt('als_output10.txt', unpack=False, usecols=(0,1), dtype=str)]

  input = sys.argv[1] # User input. Can be a file containing a list of coordinates or a single RA+Dec pair

  if input in os.listdir("./"): # If the user input is a file in the current working directory, then read that file for coordinates.
    ra0, dec0 = np.loadtxt(input, dtype=str, unpack=True)
    ra = np.array([RA2Decimal(k) if ":" in k else k for k in ra0], dtype=np.float64) # Convert RA to decimal degrees.
    dec = np.array([Dec2Decimal(k) if ":" in k else k for k in dec0], dtype=np.float64) # Convert Dec to Decimal degrees.
    coords = np.array([ra, dec]).T

    from multiprocessing import Pool
    ncores = len(ra) if len(ra) < 128 else 128
    ncores = 1 # Force 1 core for users using the login nodes. Comment this line out if you want to submit a job on slurm for up to 128 cores.
    pool = Pool(ncores)
    results = pool.map(run_periodograms, coords)

  else: # The user's input is a single RA+Dec pair.
    ra = np.float64(RA2Decimal(sys.argv[1])) if ":" in sys.argv[1] else np.float64(sys.argv[1]) # Convert RA to decimal degrees.
    dec = np.float64(Dec2Decimal(sys.argv[2])) if ":" in sys.argv[2] else np.float64(sys.argv[2]) # Convert Dec to decimal degrees.
    #print(ra,dec)
    run_periodograms([ra, dec])

