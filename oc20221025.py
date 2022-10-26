
#https://iopscience.iop.org/article/10.1088/2041-8205/740/2/L53#apjl406696fd7
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':['DejaVu Sans'],'size':18})
rc('text', usetex=True)

def est(m1,sigm1, m2,sigm2, p,sigp, t):
    sigt = 0
    pdot,sigpdot = get_pdot(m1,sigm1, m2,sigm2, p*24*3600,sigp*24*3600)

    print(f"Pdot = {pdot} +- {sigpdot}")
    val = -0.5*pdot/(p/365.25)*t*t
    val *= 86400*365.25 # Convert years to seconds (total offset in seconds)

    A = 0.5*86400*365.25
    pdothat = sigpdot * t*t / (p/365.25)
    phat = (sigp/365.25) * t*t * pdot
    that = sigt * 2*pdot / (p/365.25) * t

    err = A * np.sqrt( phat**2 + pdothat**2 + that**2)

    return([val,err])


def get_pdot(m1,sigm1, m2,sigm2, p,sigp):

    G = 6.67e-11
    c = 299792458
    msol = 2e30

    val = 96./5 * (G**(5./3)) / (c**5) * (m1*msol*m2*msol)/((m1*msol+m2*msol)**(1./3)) * ((2*np.pi/p)**(8./3)) * p
    A = 96./5 * (G**(5./3)) / (c**5) * (2*np.pi)**(8./3)

    m1hat = sigm1*msol * (p**(-5./3)) * (-1./3 * m1*msol*m2*msol / ((m1*msol+m2*msol)**(4./3)) + m1*msol/((m1*msol+m2*msol)**(1./3)))
    m2hat = sigm2*msol * (p**(-5./3)) * (-1./3 * m1*msol*m2*msol / ((m1*msol+m2*msol)**(4./3)) + m2*msol/((m1*msol+m2*msol)**(1./3)))
    phat = -5./3*sigp * p**(-8./3) * m1*msol * m2*msol / ((m1*msol + m2*msol)**(1./3))

    err =  A * np.sqrt(phat**2 + m1hat**2 + m2hat**2)

    return([val, err])



# Make sure all instances of t or t-t0 being passed into 'test' are in years, not days.


m1 = 0.306 ; sigm1 = 0.014
m2 = 0.524 ; sigm2 = 0.050
# p = 0.0281258426 ; sigp = 0 # Period from run03 using slightly diff sum of radii
# p = 0.028125839937778847 # lomb-scargle period, 25 terms stepsize/=240
# p = 0.028125846978240172 ; sigp = 0.0000000016 # lomb-scargle period
p = 0.0281258393 ; sigp = 0.0000000016 # jktebop period all
p = 0.0281258394 ; sigp = 0.0000000015 # jktebop period all
# p = 0.0281258436 # one of the recent overnight fits
nyears = (2460047.5000000 - 2458073.88809)/365.25
nyears = 10

# Define a 'package' to you can type '*package' instead of 'm1,sigm1,m2,sigm2,p,sigp' everytime
package = [m1, sigm1, m2, sigm2, p, sigp]

val,err = est(*package, nyears) # Print projected offset based on *package parameters nyears in the future

print(f"Timing offset in {nyears} years: {val} +- {err} seconds")
import sys; sys.exit()


t0 = 2457814.82095
t0 = 2458073.88809
t1 = 2458132.89610
t0m = t0-0.00002
t0p = t0+0.00002
# t3 = 2458554.75159
mins = [t0,t0m,t0p,t1]

x = [k-t0 for k in mins]
x_err = [0.00002,0,0,0.00003]

y = [est(*package, k/365.25)[0] for k in x]
y_err = [k*24*3600 for k in x_err]
yz_err = [0.00002*24*3600, 0.00003*24*3600]

y2 = [] # Determine expected eclipse time compared to observed.
for k in mins:
    q = (k-t0)/p - int((k-t0)/p)
    if k != t0m and k != t0p:
        if q*p*24*3600 >= p/2*24*3600:
            print(f'-{np.round(p*24*3600-q*p*24*3600,1)} seconds off expected eclipse timing')
        else:
            print(f'+{np.round(q*p*24*3600,1)} seconds off expected expected eclipse timing')
    if q == 0:
        z = 0
    elif q > 0.5:
        z = -(1-q)*p
    else:
        z = q*p
    y2.append(z*24*3600)


mod = np.linspace(t0,t0+365.25*nyears,1000)
est_val = est(*package, (mod-t0)/365.25)[0]
est_err = est(*package, (mod-t0)/365.25)[1]

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(9,8))

combined_error = np.sqrt(est_err**2+((t0-t0m)*24*3600)**2)
proj_err = (np.mean(y_err) + est_err)

z = (mod[np.where(est_val + combined_error + 3*np.mean(yz_err) <= -3*np.mean(yz_err[0]))[0][0]]-t0)/365.25
z2 = (mod[np.where(est_val + combined_error + 3*np.mean(yz_err) <= -0*np.mean(yz_err[0]))[0][0]]-t0)/365.25

print(f'Years til 3-sigma detection: {(mod[np.where(est_val + combined_error + 3*np.mean(yz_err) <= 0*np.mean(yz_err[0]))[0][0]]-t0)/365.25}')

ax.errorbar(x,y2, yerr=y_err, marker='.',color='black',markersize=0,capsize=2,elinewidth=1,linestyle='None', label="Observed $T_0$")
ax.axhline(y=0,xmin=0,xmax=1,color='black',linewidth=1,linestyle='--')
# ax.fill_between(mod-t0, y1=-x_err[0]*24*3600, y2=+x_err[0]*24*3600,alpha=0.3,color='darkgrey')   # horizontal grey shaded region
# ax.errorbar(mod-t0, est_val + combined_error, yerr=3*np.mean(yz_err),linestyle="None",alpha=0.2,marker='x',markersize=5)  # Blue data points
# ax.axhline(y=np.mean(y_err[0])*-1,xmin=0,xmax=1,color='black',linewidth=1,linestyle='--')
ax.plot(mod-t0, est(*package, (mod-t0)/365.25)[0], linestyle="--", color='crimson', linewidth=1,label='Projected offset')
ax.fill_between(mod-t0, y1=est_val-est_err, y2=est_val+est_err,alpha=0.3,color='crimson')
ax.fill_between(mod-t0, y1=est_val-est_err, y2=est_val+est_err,alpha=0.3,color='crimson')


ax.fill_between(mod-t0, y1=est_val - combined_error, y2=est_val + combined_error, alpha=0.3,color='darkgrey')


# ax.fill_between(mod-t0, y1=est_val-est_err-(t0-t0m)*24*3600, y2=est_val+est_err-(t0p-t0)*24*3600, alpha=0.3,color='darkgrey')
# ax.fill_between(mod-t0, y1=est_val-est_err+(t0-t0m)*24*3600, y2=est_val+est_err+(t0p-t0)*24*3600, alpha=0.3,color='darkgrey')
ax.set_xlabel("Date (YYYY)")
ax.set_ylabel("$O-C$\n(seconds)")
ax.legend(loc='lower left')
# ticks = [0,365.25*1,365.25*2,365.25*3,365.25*4,365.25*5,365.25*6,365.25*7,365.25*8,365.25*9,365.25*10] # Mark full years away from observation
ticks = np.array([0,365.25*1,365.25*2,365.25*3,365.25*4,365.25*5,365.25*6,365.25*7,365.25*8,365.25*9,365.25*10]) # Mark new years only: 2017.0, 2018.0, 2019.0, etc
ticks = np.array([k*365.25 for k in range(18)])
ticks += 46.5
ax.set_xticks(ticks)
ax.set_yticks([5,0,-5,-10,-15,-20,-25,-30,-35])
ax.set_yticks(np.arange(5,-95,-5))
# ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10])

from datetime import datetime, timedelta, date
# start = date.fromisoformat('2017-03-02')
start = date.fromisoformat('2017-11-16')
hold = [start + timedelta(days=k) for k in ticks]
xlabels = [f'{k.year}' for k in hold]
ax.set_xticklabels(xlabels,rotation=45)
ax.tick_params(bottom=True,labelbottom=True,top=True,labeltop=False,right=True,labelright=False,direction='inout')

d = start + timedelta(days=z2*365.25)
print(d)

plt.savefig('grav_offset3.png',dpi=500)
plt.show()
