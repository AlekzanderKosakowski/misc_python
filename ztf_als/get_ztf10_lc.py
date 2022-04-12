from Coordinates import Decimal2RA, Decimal2Dec, RA2Decimal, Dec2Decimal # Keep RA and Dec in decimal format
import numpy as np # Read in files
import sys # Use command-line inputs
import os # Check if command-line input is a file
from numba import njit

def get_lc(ra1, dec1, files, save_lc):
  #
  # Search the ZTF parquet files for light curve data.
  # Include only "good" data (catflags<32768)
  # Include only objects with >30 observations.
  #
  import pandas as pd
  hjd, mag, merr, filter, catflag = [[] for k in range(5)]


  for f in files:
    fieldnumber = f"{int(f.split('_')[1]):0>6d}"
    path0 = f"/lustre/research/kupfer/catalogs/ZTF8/data/ZTF/lc_dr8/0/field{fieldnumber}/"
    path1 = f"/lustre/research/kupfer/catalogs/ZTF8/data/ZTF/lc_dr8/1/field{fieldnumber}/"
    try:
      data = pd.read_parquet(f"{path0}{f}", columns=['objra', 'objdec', 'hmjd', 'mag', 'magerr', 'filterid', 'nepochs', 'catflags'])
    except:
      data = pd.read_parquet(f"{path1}{f}", columns=['objra', 'objdec', 'hmjd', 'mag', 'magerr', 'filterid', 'nepochs', 'catflags'])

    ra = np.array(data.objra, dtype=np.float64)
    dec = np.array(data.objdec, dtype=np.float64)
    nepochs = np.array(data['nepochs'], dtype=np.float64)
    for i in range(len(ra)):
      if nepochs[i] < 30:
        continue
      ra2 = ra[i]
      dec2 = dec[i]
      dist = np.degrees(np.arccos( np.cos(np.radians(dec1))*np.cos(np.radians(dec2))*np.cos(np.radians(ra1)-np.radians(ra2)) + np.sin(np.radians(dec1))*np.sin(np.radians(dec2))))*3600
      if dist < 2.5:
        hjd.append(list(map(float, list(data['hmjd'])[i])))
        mag.append(list(data['mag'])[i])
        merr.append(list(data['magerr'])[i])
        catflag.append(list(data['catflags'])[i])
        f0 = []
        for l in range(len(hjd[-1])):
          f0.append(list(data['filterid'])[i])
        filter.append(f0)

  catflag0 = np.array([k for j in catflag for k in j])
  good_obs = np.where(catflag0<32768)[0]

  hjd0 = np.array([k for j in hjd for k in j], dtype=np.float64)[good_obs]
  mag0 = np.array([k for j in mag for k in j])[good_obs]
  merr0 = np.array([k for j in merr for k in j])[good_obs]
  filter0 = np.array([k for j in filter for k in j])[good_obs]

  ra2 = Decimal2RA(ra1)
  dec2 = Decimal2Dec(dec1)
  ra_name = "".join(ra2[:2]) + str(int(float(ra2[-1])))
  dec_name = "".join(dec2[:2]).replace('+','p').replace('-','m') + str(int(float(dec2[-1])))
  if save_lc:
    with open(f'{lcdir}/{ra_name}{dec_name}_lc.txt', 'w') as ofile:
      for l in range(len(mag0)):
        ofile.write(f"{hjd0[l]}  {mag0[l]}  {merr0[l]}  {filter0[l]}\n")
  lc_data = []
  for i,k in enumerate(hjd0):
    lc_data.append([hjd0[i], mag0[i], merr0[i], filter0[i]])
  lc_data = np.array(lc_data)
  #print('len(lc_data) ',len(lc_data))
  if len(lc_data) == 0:
    return([[],[],[]])
  else:
    return(lc_data.T)

@njit
def check_limits(ra, dec, field_limits, files):
  arcsec = 1./3600. # 1 arcsec in degrees
  buffer = 2.5*arcsec
  outfiles = []
  for i in range(len(field_limits[0])): # For each field
    ra_min, ra_max = field_limits[0][i], field_limits[1][i] # RA min and max of the field.
    if ra_min <= 10 and ra_max >= 350: # If the range crosses over the 360.0 degree mark (min=359, max=1), then unloop the range by adding 360 to the min and swapping min-max so the range becomes (min=359, max=361)
      ra_min2 = ra_max
      ra_max2 = ra_min + 360
      if ra <= 10: # If the object's RA was after 360.0 degrees, add 360.0 to it.
        ra += 360
      if ra >= ra_min2 - buffer and ra <= ra_max + buffer:
        dec_min, dec_max = field_limits[2][i], field_limits[3][i]
        if dec >= dec_min - buffer and dec <= dec_max + buffer:
          #print(f"Coordinates located in: {files[i]}")
          #print(ra, ra_min, ra_max, dec, files[i])
          outfiles.append(files[i])
    else:
      if ra >= ra_min - buffer and ra <= ra_max + buffer:
        dec_min, dec_max = field_limits[2][i], field_limits[3][i]
        if dec >= dec_min - buffer and dec <= dec_max + buffer:
          #print(f"Coordinates located in: {files[i]}")
          #print(ra, ra_min, ra_max, dec, files[i])
          outfiles.append(files[i])

  return(outfiles)


def get_ztf_lc(ra, dec, save_lc):
  if save_lc:
    global lcdir
    lcdir = "ztf_lc_files"
    if not os.path.isdir(lcdir):
        os.system(f"mkdir {lcdir}")

  field_limits = np.loadtxt("/lustre/research/kupfer/catalogs/ZTF8/field_limits.txt", unpack=True, usecols=(0,1,2,3), dtype=np.float64) # RA and Dec limits of each field.
  files = np.loadtxt("/lustre/research/kupfer/catalogs/ZTF8/field_limits.txt", unpack=True, usecols=(4), dtype=str) # Filenames associated with each field.

  outfiles = check_limits(ra, dec, field_limits, files)
  if len(outfiles) == 0:
    return([[],[],[]])

  lc_data = get_lc(ra, dec, outfiles, save_lc)
  return(lc_data)

if __name__ == "__main__":

  input = sys.argv[1] # User input. Can be a file containing a list of coordinates or a single RA+Dec pair

  if input in os.listdir("./"): # If the user input is a file in the current working directory, then read that file for coordinates.
    ra0, dec0 = np.loadtxt(input, dtype=str, unpack=True)
    ra = np.array([RA2Decimal(k) if ":" in k else k for k in ra0], dtype=np.float64) # Convert RA to decimal degrees.
    dec = np.array([Dec2Decimal(k) if ":" in k else k for k in dec0], dtype=np.float64) # Convert Dec to Decimal degrees.

    for i in range(len(ra)):
      get_ztf_lc(ra[i], dec[i], True)

  else: # The user's input is a single RA+Dec pair.
    ra = np.float64(RA2Decimal(sys.argv[1])) if ":" in sys.argv[1] else np.float64(sys.argv[1]) # Convert RA to decimal degrees.
    dec = np.float64(Dec2Decimal(sys.argv[2])) if ":" in sys.argv[2] else np.float64(sys.argv[2]) # Convert Dec to decimal degrees.
    get_ztf_lc(ra, dec, True)
