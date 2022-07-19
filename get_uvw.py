
# Geometry equations:
# https://articles.adsabs.harvard.edu/pdf/1987AJ.....93..864J
# Johnson & Soderblom 1987 v93 4

# Solar LSR values:
# https://ui.adsabs.harvard.edu/abs/2010MNRAS.403.1829S/abstract

import numpy as np
from scipy.linalg import solve
from Coordinates import RA2Decimal,Dec2Decimal


def get_uvw(ra, dec, rv, rv_err, para, para_err, pmra, pmra_err, pmdec, pmdec_err):

#    print(ra, dec, rv, rv_err, para, para_err, pmra, pmra_err, pmdec, pmdec_err)

    ra = np.radians(ra)
    dec = np.radians(dec)

    pmra /= 1000. ; pmra_err /= 1000.
    pmdec /= 1000. ; pmdec_err /= 1000.
    para /= 1000. ; para_err /= 1000.

    ra_ngp = np.radians(192.25)
    dec_ngp = np.radians(27.4)
    theta = np.radians(123.)

    solar = [11.10, 12.24, 7.25]
    solar_err = [[0.75,1.0],[0.47,2.0],[0.37,0.5]]


    tmatrix = [[-np.cos(theta)*np.sin(dec_ngp)*np.cos(ra_ngp)-np.sin(theta)*np.sin(ra_ngp), -np.cos(theta)*np.sin(dec_ngp)*np.sin(ra_ngp)+np.sin(theta)*np.cos(ra_ngp), np.cos(theta)*np.cos(dec_ngp)],
               [-np.sin(theta)*np.sin(dec_ngp)*np.cos(ra_ngp)+np.cos(theta)*np.sin(ra_ngp), -np.sin(theta)*np.sin(dec_ngp)*np.sin(ra_ngp)-np.cos(theta)*np.cos(ra_ngp), np.sin(theta)*np.cos(dec_ngp)],
               [np.cos(dec_ngp)*np.cos(ra_ngp), np.cos(dec_ngp)*np.sin(ra_ngp), np.sin(dec_ngp)]]
    T = tmatrix

    amatrix = np.array([[np.cos(ra)*np.cos(dec), -np.sin(ra), -np.cos(ra)*np.sin(dec)],
                        [np.sin(ra)*np.cos(dec),  np.cos(ra), -np.sin(ra)*np.sin(dec)],
                        [np.sin(dec),                      0,             np.cos(dec)]])
    A = amatrix

    k = 4.74057
    G = np.array([rv, k*pmra/para, k*pmdec/para])

    # Do it manually...
    # B = np.array([
    #            [T[0][0]*A[0][0] + T[0][1]*A[1][0] + T[0][2]*A[2][0],
    #               T[0][0]*A[0][1] + T[0][1]*A[1][1] + T[0][2]*A[2][1],
    #                 T[0][0]*A[0][2] + T[0][1]*A[1][2] + T[0][2]*A[2][2]],
    #
    #            [T[1][0]*A[0][0] + T[1][1]*A[1][0] + T[1][2]*A[2][0],
    #               T[1][0]*A[0][1] + T[1][1]*A[1][1] + T[1][2]*A[2][1],
    #                 T[1][0]*A[0][2] + T[1][1]*A[1][2] + T[1][2]*A[2][2]],
    #
    #            [T[2][0]*A[0][0] + T[2][1]*A[1][0] + T[2][2]*A[2][0],
    #               T[2][0]*A[0][1] + T[2][1]*A[1][1] + T[2][2]*A[2][1],
    #                 T[2][0]*A[0][2] + T[2][1]*A[1][2] + T[2][2]*A[2][2]],
    #            ])
    #
    # u = B[0][0]*G[0] + B[0][1]*G[1] + B[0][2]*G[2] + solar[0]
    # v = B[1][0]*G[0] + B[1][1]*G[1] + B[1][2]*G[2] + solar[1]
    # w = B[2][0]*G[0] + B[2][1]*G[1] + B[2][2]*G[2] + solar[2]
    # print(u,v,w)

    B = np.dot(T,A)
    uvw = np.dot(B, G)
    uvw += solar

    # Quadrature vector.
    Q = [rv_err**2,
         ((k/para)**2)*(pmra_err**2 + (pmra*para_err/para)**2 ),
         ((k/para)**2)*(pmdec_err**2 + (pmdec*para_err/para)**2 )]

    # Error terms
    uvw_err = np.array([ np.sqrt((B[0][0]*rv_err)**2 + (B[0][1]*k*pmra_err/para)**2 + (B[0][2]*k*pmdec_err/para)**2 +(k*para_err/para/para*(B[0][1]*pmra+B[0][2]*pmdec))**2 + (np.sqrt(solar_err[0][0]**2+solar_err[0][1]**2))**2),
                         np.sqrt((B[1][0]*rv_err)**2 + (B[1][1]*k*pmra_err/para)**2 + (B[1][2]*k*pmdec_err/para)**2 +(k*para_err/para/para*(B[1][1]*pmra+B[1][2]*pmdec))**2 + (np.sqrt(solar_err[1][0]**2+solar_err[1][1]**2))**2),
                         np.sqrt((B[2][0]*rv_err)**2 + (B[2][1]*k*pmra_err/para)**2 + (B[2][2]*k*pmdec_err/para)**2 +(k*para_err/para/para*(B[2][1]*pmra+B[2][2]*pmdec))**2 + (np.sqrt(solar_err[2][0]**2+solar_err[2][1]**2))**2)])

    return(uvw, uvw_err)


if __name__ == "__main__":
    #
    # Test case for calculating UVW and UVW error
    #
    print('-----------')
    ra = float(RA2Decimal("01:35:00.856"))
    dec = float(Dec2Decimal("+23:59:460"))
    pmra = -6.428
    pmra_err = 0.300
    pmdec = -7.056
    pmdec_err = 0.225
    para = 1.1769
    para_err = 0.2884
    rv = -35.66
    rv_err = 6.18

    grav_rv = 2.43

    rv -= grav_rv

    print(f"U = {uvw[0]:>8.3f} +/- {np.sqrt(uvw_err[0]):>6.3f}")
    print(f"V = {uvw[1]:>8.3f} +/- {np.sqrt(uvw_err[1]):>6.3f}")
    print(f"W = {uvw[2]:>8.3f} +/- {np.sqrt(uvw_err[2]):>6.3f}")
