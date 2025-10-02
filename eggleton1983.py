import numpy as np
import sys

def eggleton_density(q, q_error, period, period_error=0):
    '''
    https://ui.adsabs.harvard.edu/abs/1983ApJ...268..368E/abstract

    Estimate the mean density of a roche-filling donor star
      based on the system mass ratio and orbital period
      using Eggleton1983 calculations

    Estimate on R_L is good to "better than 1%",
      so we add 1% error in quadtrature to our RL_error variable.

    Inputs:
        q:            Mass ratio
        q_error:      Error in mass ratio
        period:       Orbital period (seconds)
        period_error: Error in orbital period (seconds)

    Returns:
        rho:          Estimated mean density (g cm^-3)
        rho_error:    Error in estimated mean density (g cm^-3)
    '''
    A = 0.49
    B = 0.60
    beta = B*q**(2./3) + np.log(1+q**(1./3))

    RL = A*q**(2./3) / beta

    gamma = 2 * (q**(1./3)) * (1+q**(1./3))
    RL_error = (2./3)*(q_error/q)*RL*(1 - B + gamma**-1)
    RL_error = (RL_error**2 + (RL*0.01)**2)**0.5 


    period_days = period/86400
    period_days_error = period_error/86400
    C = 0.1375
    rho = (C * (q/(1+q))**0.5 * RL**(-3./2) / period_days)**2
    base = 2*C * rho
    part_q = (1./2) / (q/(1+q)) * q_error * (1+q) * (1 - q*(1+q)**(-3))
    part_R = (-3./2) / RL * RL_error
    part_P = (-1./2) / period_days * period_days_error

    rho_error = base*(part_q**2 + part_R**2 + part_P**2)**0.5

    return rho, rho_error


if __name__ == "__main__":

    q = float(sys.argv[1])
    q_error = float(sys.argv[2])
    period = float(sys.argv[3])
    period_error = float(sys.argv[4])

    rho, rho_error = eggleton_density(q, q_error, period, period_error)
    print(f"Mean Density = {rho:.5f} +/- {rho_error:.5f} g/cm^3")
