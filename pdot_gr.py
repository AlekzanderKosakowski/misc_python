#https://iopscience.iop.org/article/10.1088/0004-637X/700/2/965

import numpy as np

def get_offset(p, perr, pdot, pdot_err, t=10):
  #
  #
  #
  offset = ((t*365.25*86400)**2)/2 * pdot / p

  A = 0.5
  pdot_hat = pdot_err * ((t*365.25*86400)**2) / p
  p_hat = ((t*365.25*86400)**2) * pdot * (-p_err / (p**2))

  offset_err = A * np.sqrt( p_hat**2 + pdot_hat**2)

  return(offset, offset_err)


if __name__ == "__main__":
    rsol = 7e8 # m


    # https://ui.adsabs.harvard.edu/abs/2014MNRAS.438.3399B/abstract
    a = 0.886*rsol # m
    a_err = 0.014*rsol # m
    p = 0.1160154352*86400 # s
    p_err = 0.0000000015*86400 # s

    omega = np.radians(90.209) # rad
    omega_err = np.radians(0.002) # rad

    e = 0.034 # 3sigma upper limit
    e_err = 0

    c = 299792458 # m/s


    # Equation 20

    A = 576 * (np.pi**5) / (c**4) # (s/m)**4

    x1 = e*np.cos(omega)
    x2 = (1+e*np.sin(omega))**-3
    x3 = (a/p)**4 # (m/s)**4
    x4 = (1-e**2)**-0.5

    pdot = A*x1*x2*x3*x4 # s/s
    #print(f"Pdot_GW = {pdot:>7.3e} s/s")

    omega_hat = omega_err *( ((a/p)**4)*x4 * (-3*e*np.cos(omega)) * ((1+e*np.sin(omega))**-4) * e*np.cos(omega) - ((1+e*np.sin(omega))**-3) * e*np.sin(omega))
    a_hat = x1*x2*x4 * (4*x3/a*a_err)
    p_hat = x1*x2*x4 * (-4*x3/p*p_err)

    pdot_err = A * np.sqrt( (omega_hat)**2 + (a_hat)**2 + (p_hat)**2)
    #print(f"Pdot_GW_err = {pdot_err:>7.3e} s/s")

    print(f"Pdot_GW = {pdot:>7.3e} +- {pdot_err:>7.3e} s/s")

    t = 10
    offset, offset_err = get_offset(p, p_err, pdot, pdot_err,t)

    print(f"Eclipsing timing offset after {t} years: {offset:>6.3f} +- {offset_err:>6.3f}.")
