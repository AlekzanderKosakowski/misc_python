# Python3-based script used to update the headers for the FITS images generated by the 82-inch telescope at McDonald Observatory.
# Uses the timestamps.csv file created by the OLDMAID software.

from astropy.time import Time
from astropy import time, coordinates as coord, units as u
import astropy # Changing Date/Time into JD for headers
from astropy.io import fits # Opening and manipulating fits files. Adding header key/value
from astropy.coordinates import EarthLocation # Obtaining site location of McDonald observatory for barycentric correction
import numpy as np # General use
import os # Search directory for filenames
import sys # Command line arguments

mcdonald = EarthLocation.of_site("McDonald Observatory")

timestamps_file = [k for k in os.listdir() if "timestamps" in k][0] # Find the timestamps.csv file
frames, times_start, times_stop =  np.loadtxt(timestamps_file, skiprows=1,delimiter=',',unpack=True,dtype=str,usecols=(0,1,2)) # Use only the frame#, EXP_START, and EXP_STOP timings

for i in range(len(times_start)):
  times_start[i] = times_start[i].replace(' ', "T").strip('"') # Modify the formatting into standard DATE/TIME format for astropy.
  times_stop[i]  = times_stop[i].replace(' ', "T").strip('"') # Modify the formatting into standard DATE/TIME format for astropy.
atimes_start = Time(times_start, format="isot", scale="utc", precision=6, location=mcdonald) # Store the astropy version of the times
atimes_stop  = Time(times_stop, format="isot", scale="utc", precision=6, location=mcdonald) # Store the astropy version of the times

exptime = atimes_stop - atimes_start

atimes = atimes_start + exptime/2

# Calculate barycentric correction for light travel times
# https://docs.astropy.org/en/stable/time/index.html#barycentric-and-heliocentric-light-travel-time-corrections
try:
  ra, dec = sys.argv[2], sys.argv[3]
except IndexError:
  print("Provide the command line arguments when running this script.\n\npython3 update_header82.py <file_prefix> <R.A.> <Dec.> <Filter>\n")
  print("Example:   \"python3 update_header82.py 0353p4315 08:22:39.54 +30:48:57.19 BG40\"\n")
  sys.exit()
ip_peg = coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
barycentric_correction = atimes.light_travel_time(ip_peg)

# Apply the barycentric correction to the barycentric dynamical time version of the times recorded by the telescope software.
times_bjd_tdb = atimes.tdb + barycentric_correction

object = sys.argv[1]
files = sorted([k for k in os.listdir() if object in k and k[-5:] == ".fits"]) # List of fits files to add image timings to.


for i,f in enumerate(files):

  file = fits.open(f, mode='update') # Open the fits file and prepare to update its header
  header = file[0].header # Read the header

  if header.get('bjd_tdb',False): # If the header already has bjd_tdb, then skip the file.
    print(f"Skipping {f}; BJD_TDB header value already exists: {header['BJD_TDB']}")
    #continue # Comment out this line if you want to overwrite the headers if they already exist.

  header.set('object', object, "User-supplied object name")
  header.set('ra', ra, "User-supplied R.A.")
  header.set('dec', dec, "User-supplied Dec.")
  header.set('bjd_tdb', times_bjd_tdb[i].jd, "UTC based mid-exposure BJD_TDB") # Assign bjd_tdb to the header. Assumes the files and timings lists are in the same order, which is a safe assumption based on file names.

  filter = sys.argv[4]
  header.set('Filter', filter, "User-supplied Filter")

  # Setup the readnoise and gain settings.
  # The EM gain is surprisingly already in the header.
  # The readnoise is published online: https://www.princetoninstruments.com/products/proem-family/pro-em
  #   See info on "ProEM HS: 1024BX3"
  #   Still need to determine these experimentally to be sure it hasn't changed.
  emgain_key = "PI Camera Adc EMGain"
  emgain = header.get("PI Camera Adc EMGain", "Unknown") # Ranges from 1-1000
  readrate = header.get("PI Camera Adc Speed", "Unknown")
  readnoise = {"1": 4.0, # Confirmed units MHz
               "100": 3.5, # Assumed units kHz
               "Unknown": "Unknown"}

  header.set('gain', float(emgain), "EM Gain from header")
  header.set('rdnoise', readnoise[readrate], "Readnoise based on readrate and documentation")

  now    = Time.now().value
  year   = now.year
  month  = now.month
  day    = now.day
  hour   = now.hour
  minute = now.minute
  second = now.second

  header.set('modify', f"{year}-{month:0>2d}-{day:0>2d}T{hour:0>2d}:{minute:0>2d}:{second:0>2d}", "Time of last modification.")
  file.close() # Close the fits file. This line is what saves the changes.
