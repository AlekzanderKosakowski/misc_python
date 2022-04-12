import numpy as np
from astropy.timeseries import LombScargle # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html

def run_als(hjd, flux, ferr, freq_grid):

  t1 = np.min(hjd) # Earliest HJD measured. Used for phase calculation
  t2 = np.max(hjd)
  npts = max(len(hjd),2000) # Create at least 1000 model points for a well-sampled model light curve.
  times = np.linspace(t1,t2,num=npts)

  # Astropy Lomb Scargle. Auto centers data at mag=0, fits a single sine term.
  # Using method='fast' is about 100x faster than 'cython' but apparently not as accurate and can result in negative or >1 power.
  power = LombScargle(hjd,flux,dy=ferr,center_data=True,nterms=1).power(freq_grid,method='fast',assume_regular_frequency=True) # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.power
  #power = LombScargle(hjd,flux,dy=ferr,center_data=True,nterms=1).power(freq_grid,method='cython',assume_regular_frequency=True) # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.power
  if np.min(power) < 0.0 or np.max(power) > 1.0:
      power = LombScargle(hjd,flux,dy=ferr,center_data=True,nterms=1).power(freq_grid,method='cython',assume_regular_frequency=True) # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.power
  peak_index = np.argmax(power)
  peak_freq = freq_grid[peak_index]
  peak_power = power[peak_index]
  # peak_freq = freq_grid[np.where(power==np.max(power))][0]
  # peak_power = power[np.where(power==np.max(power))][0]
  model = LombScargle(hjd,flux,dy=ferr,center_data=True,nterms=1).model(times,peak_freq) # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.model
  amp = abs(np.median(model)-np.max(model))

  return(power, peak_freq, peak_power, times, model, amp)
