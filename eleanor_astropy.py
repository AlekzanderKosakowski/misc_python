'''
Use eleanor to find data on target at (RA,Dec)
Uses astropy.timeseries.LombScargle to obtain peak frequency
Saves light-curve file as 3-column .txt
Plots LC, LS-Diagram, and Phase-folded LC on a single plot
Optional bootstrap error estimates for amplitude and frequency
Bootstrap function built for multiprocessing


Common errors:
---------------------------------------------------------------------------
First step: figure out which sector gave an error.
Next step: delete your /.eleanor/metadata/s00##/ directory ; ## = sector
Next step: run: eleanor.Update(sector=##) to redownload the files
Final step: try your code again
---------------------------------------------------------------------------
Problem:
OSError: /.eleanor/metadata/s0013/cadences_s0013.txt not found.
-------------------------------------
Solution:
Delete the s0013/ directory (for sector 13)
Run eleanor.Update(sector=13) to redownload the sector13 files.
---------------------------------------------------------------------------
Problem:
Singular Matrix
-------------------------------------
Solution:
Not sure how to fix this.
I believe this also requires eleanor.Update(sector=##).
Check your /.eleanor/metadata/s####/ directories. Make sure they each have
  the same number of files in them.
I notice this error after connection timeout during sector data downloads.
---------------------------------------------------------------------------
'''


# pip3 install eleanor
# pip3 install --user eleanor
import eleanor # https://adina.feinste.in/eleanor/index.html
import numpy as np
from Coordinates import RA2Decimal,Dec2Decimal # https://github.com/AlekzanderKosakowski/misc_python/blob/master/Coordinates.py
from astropy.timeseries import LombScargle # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html
from multiprocessing import Pool, cpu_count
import time
import os.path
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':['DejaVu Sans']})
rc('text', usetex=True)

def get_lc(ra,dec):
    # Collect data on target from each sector.
    # tc (bool, optional) – If True, use a TessCut cutout to produce
    #   postcards rather than downloading the eleanor postcard data products.
    # If tc=False (default) then sector 13 returns a singular matrix with
    #   do_pca=True.
    global targetname

    data = eleanor.multi_sectors('all',coords=(ra,dec),tc=True)


    # Extract LC data for each sector using PSF photometry
    mjd,flux,ferr = ([] for x in range(3))
    for k in data:
        d = eleanor.TargetData(k, height=15, width=15, bkg_size=31,
                               do_psf=True, do_pca=True, crowded_field=True)
        quality = d.quality  # Keep only quality = 0 data
        index = np.where(quality == 0)

        # Based on fits file headers, time is Barycentric Julian Date (-2457000 days)
        t = d.time[index]

        # corr_flux = Systematics-corrected version of lightcurve derived using aperture and tpf.
        f = d.corr_flux[index]
        # flux_err = Estimated uncertainty on raw_flux.
        e = d.flux_err[index]

        # 'Append' all data for each sector together into a 1-D array
        mjd  = np.concatenate( (mjd, t) )
        #flux = np.concatenate( (flux, f/np.median(f)) )
        flux = np.concatenate( (flux, f/np.median(f)) )
        ferr = np.concatenate( (ferr, e) )

        # LC's may be large. Clear space after finished.
        del(d,t,f,e)

    # Save LC to file for easy access later.
    lcname = str(targetname) + "_tess.txt"
    with open(lcname,'w') as ofile:
        for k in range(len(mjd)):
            ofile.write(str(mjd[k]) + "  " + str(flux[k]) + "  " + str(ferr[k]) + "\n")

    return(mjd,flux,ferr)


def get_ft(mjd,flux,ferr):
    # Use astropy.timeseries.LombScargle to obtain LS periodogram
    # Creates periodogram and model.


    # Perform three iterations of 3-sigma clipping
    for k in range(3):
      index = np.where( (flux < 3*np.std(flux)+np.median(flux)) & (flux > -3*np.std(flux) + np.median(flux)) )
      mjd  = mjd[index]
      flux = flux[index]
      ferr = ferr[index]

    # Define minimum and maximum period (in days) to search.
    # Define step size for frequency grid.
    minp = (60.)*(1./60.)*(1./24.)
    maxp = (1440.)*(1./60.)*(1./24.)
    stepsize = 0.0012342821877648902  # 1 minute stepsize
    stepsize /= 1.

    freq_grid = np.arange(1/maxp,1/minp,stepsize)

    t1 = np.min(mjd)
    t2 = np.max(mjd)
    npts = (t2-t1)/(24*60*6000)
    times = np.arange(t1,t2,npts)
    nterms = 1 # Number of sine-terms to use to find best-period.
    #start = time.time()
    power = LombScargle(mjd,flux,dy=ferr,center_data=True,nterms=nterms).power(freq_grid)
    peak_freq = freq_grid[np.where(power==np.max(power))][0]
    model = LombScargle(mjd,flux,dy=ferr,center_data=True,nterms=1).model(times,peak_freq)
    amp = abs(np.median(model)-np.max(model))
    #print(f"Peak Frequency ({nterms} terms): {np.round(peak_freq,8)} ; {np.round((time.time() - start)/60.,3)} minutes.")

    return(mjd,flux,ferr,freq_grid,power,peak_freq,times,model,amp)


def get_phase(mjd,peak_freq,times,model):
    # Create phase points for data and model.
    # Sort the model points so it appears as a smooth line with ax.plt()
    # Aside from eleanor finding/downloading/creating LC data, this
    #   is by far the slowest part of the code.

    # Assign phase to each flux point for phase-folded light curve.
    # Uses best-fit frequency from astropy.timeseries.LombScargle
    phase = []
    for j in mjd:
        phase.append( (j-mjd[0])*peak_freq - int( (j-mjd[0])*peak_freq   ) )
    phase = np.array(phase)
    mphase = []
    for j in times:
        mphase.append( (j-times[0])*peak_freq - int( (j-times[0])*peak_freq   ) )
    mphase = np.array(mphase)

    # Create phased model to overplot on phase data. Needs to be sorted for ax.plot()
    # There is probably a better way to sort 2D arrays, but I can't think of it.
    m = np.array([np.concatenate((mphase,mphase+1)),np.concatenate((model,model))]).T
    smodel  = np.array(sorted(m,key=lambda l:l[0])).T[1][0::int(len(times)/len(mjd))]
    smphase = np.array(sorted(m,key=lambda l:l[0])).T[0][0::int(len(times)/len(mjd))]

    return(phase,smphase,smodel)

def get_plot(mjd,flux,ferr,phase,freq_grid,power,peak_freq,mphase,model):
    # Create the 3-part plot.
    # Plots are:
    #   1: TESS flux plot
    #   2: LS Periodogram
    #   3: TESS phase folded LC. folded to highest peak from LS periodogram
    #      Includes best-fit model overplotted as well

    global targetname


    # Plot everything.
    fig,[ax1,ax2,ax3] = plt.subplots(nrows=3,ncols=1)
    plt.subplots_adjust(hspace=0.5)
    #ax1.errorbar(mjd, flux, yerr=ferr, color="black", alpha=0.25,linestyle="None",capsize=2,elinewidth=1,marker=".",markersize=1)
    ax1.scatter(mjd, flux, color="black", alpha=0.25,linestyle="None",marker=".",s=2)
    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlabel("BJD ($-$2457000 days)")

    label = str(np.round(peak_freq,5)) + " cycles/day"
    ax2.plot(freq_grid,power,color="black", alpha=1.0,linewidth=1,label=label)
    #ax2.axhline(y=4*np.mean(power),xmin=0,xmax=1,color="crimson",linestyle="--",linewidth=1)
    #ax2.axhline(y=5*np.mean(power),xmin=0,xmax=1,color="crimson",linestyle="-",linewidth=1)
    ax2.set_ylabel("LS Power")
    ax2.set_xlabel("Frequency (cycles/day)")
    if True:
        # Put a global flag here for optional inset plotting.

        # https://matplotlib.org/1.3.1/mpl_toolkits/axes_grid/users/overview.html#insetlocator
        # https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html
        # https://matplotlib.org/mpl_toolkits/axes_grid/api/inset_locator_api.html?highlight=mark_inset#mpl_toolkits.axes_grid1.inset_locator.mark_inset
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        axins = zoomed_inset_axes(ax2,zoom=30,bbox_to_anchor=(.25,0.5,.2,0.5),
                                  bbox_transform=ax2.transAxes)
        axins.plot(freq_grid,power,color="black", alpha=1.0,linewidth=1)
        axins.set_xlim(7.5,7.7)
        axins.set_ylim(0,0.005)
        mark_inset(ax2, axins, loc1=3, loc2=4, fc='none',linewidth=0.5)
        #axins.set_yticks([])
    ax2.plot(freq_grid,power,color="black", alpha=1.0,linewidth=1)
    ax2.legend()

    ax3.scatter(np.concatenate((phase,phase+1)), np.concatenate((flux,flux)),color='black',alpha=0.25,s=2)
    ax3.plot(mphase,model,color='red',alpha=1.0,linewidth=1)
    ax3.set_ylabel("Normalized Flux")
    ax3.set_xlabel("Phase")
    #ax3.set_ylim((0.995,1.005))
    #plt.suptitle(targetname)
    figname = str(targetname) + "_tess.pdf"
    plt.savefig(figname,dpi=500)

def get_boots(mjd,flux,ferr):
    # Create bootstrapped datasets for error estimation.
    # Use numpy.random.randint to randomly choose N points (with replacement)
    #   from the original data.
    # This is used because I don't know how to get a covariance matrix for
    #   uncertainties in amplitude and frequency from astropy...

    ind = np.random.randint(0,len(mjd),len(mjd))
    mjd  = np.array(mjd[ind])
    flux = np.array(flux[ind])
    ferr = np.array(ferr[ind])

    return(mjd,flux,ferr)

def run_apls(dataset):
    # Run AstroPy LombScargle
    # Takes in a dataset being handled by a process of multiprocessing
    # Each dataset was created in the function get_boots()

    mjd,flux,ferr = dataset
    mjd,flux,ferr,freq_grid,power,peak_freq,times,model,amp = get_ft(mjd,flux,ferr)

    return(peak_freq,amp)


if __name__ == '__main__':

    targetname = "0642m5605"
    ra0 = "06:42:07.99"
    dec0 = "-56:05:47.44"
    ra = float(RA2Decimal(ra0))
    dec = float(Dec2Decimal(dec0))
    print(ra,dec)

    if os.path.isfile(str(targetname) + "_tess.txt"):
        mjd0,flux0,ferr0 = np.loadtxt(str(targetname) + "_tess.txt",unpack=True,dtype=np.float64)
    else:
        mjd0,flux0,ferr0 = get_lc(ra,dec)

    bootstrap = False # Estimate amplitude/frequency errors using bootstrapping?
    if bootstrap:
        # Took like an hour to run 10000 bootstraps with 4 cores and 12k datapoints
        datasets = []
        for k in range(10000): # 10,000 bootstrapped datasets
            mjd  = np.copy(mjd0)
            flux = np.copy(flux0)
            ferr = np.copy(ferr0)

            mjd,flux,ferr = get_boots(mjd,flux,ferr)
            datasets.append([mjd,flux,ferr])

        pool = Pool(cpu_count()-1) # Use all but one of your CPU cores
        result = pool.map(run_apls, datasets)
        result = np.array(result)
        np.savetxt('results.txt',result) # Save the output in a 2-column file: [freq, amplitude]
        print(f"Amplitude: {amp} +- {np.std(result[:,0])}")
        print(f"Frequency: {peak_freq} +- {np.std(result[:,1])}")

    mjd,flux,ferr,freq_grid,power,peak_freq,times,model,amp = get_ft(mjd0,flux0,ferr0)
    phase,mphase,model = get_phase(mjd,peak_freq,times,model)
    get_plot(mjd,flux,ferr,phase,freq_grid,power,peak_freq,mphase,model)
'''
Use eleanor to find data on target at (RA,Dec)
Uses astropy.timeseries.LombScargle to obtain peak frequency
Saves light-curve file as 3-column .txt
Plots LC, LS-Diagram, and Phase-folded LC on a single plot
Optional bootstrap error estimates for amplitude and frequency
Bootstrap function built for multiprocessing


Common errors:
---------------------------------------------------------------------------
First step: figure out which sector gave an error.
Next step: delete your /.eleanor/metadata/s00##/ directory ; ## = sector
Next step: run: eleanor.Update(sector=##) to redownload the files
Final step: try your code again
---------------------------------------------------------------------------
Problem:
OSError: /.eleanor/metadata/s0013/cadences_s0013.txt not found.
-------------------------------------
Solution:
Delete the s0013/ directory (for sector 13)
Run eleanor.Update(sector=13) to redownload the sector13 files.
---------------------------------------------------------------------------
Problem:
Singular Matrix
-------------------------------------
Solution:
Not sure how to fix this.
I believe this also requires eleanor.Update(sector=##).
Check your /.eleanor/metadata/s####/ directories. Make sure they each have
  the same number of files in them.
I notice this error after connection timeout during sector data downloads.
---------------------------------------------------------------------------
'''


# pip3 install eleanor
# pip3 install --user eleanor
import eleanor # https://adina.feinste.in/eleanor/index.html
import numpy as np
from Coordinates import RA2Decimal,Dec2Decimal # https://github.com/AlekzanderKosakowski/misc_python/blob/master/Coordinates.py
from astropy.timeseries import LombScargle # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html
from multiprocessing import Pool, cpu_count
import time
import os.path
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':['DejaVu Sans']})
rc('text', usetex=True)

def get_lc(ra,dec):
    # Collect data on target from each sector.
    # tc (bool, optional) – If True, use a TessCut cutout to produce
    #   postcards rather than downloading the eleanor postcard data products.
    # If tc=False (default) then sector 13 returns a singular matrix with
    #   do_pca=True.
    global targetname

    data = eleanor.multi_sectors('all',coords=(ra,dec),tc=True)


    # Extract LC data for each sector using PSF photometry
    mjd,flux,ferr = ([] for x in range(3))
    for k in data:
        d = eleanor.TargetData(k, height=15, width=15, bkg_size=31,
                               do_psf=True, do_pca=True, crowded_field=True)
        quality = d.quality  # Keep only quality = 0 data
        index = np.where(quality == 0)

        # Based on fits file headers, time is Barycentric Julian Date (-2457000 days)
        t = d.time[index]

        # corr_flux = Systematics-corrected version of lightcurve derived using aperture and tpf.
        f = d.corr_flux[index]
        # flux_err = Estimated uncertainty on raw_flux.
        e = d.flux_err[index]

        # 'Append' all data for each sector together into a 1-D array
        mjd  = np.concatenate( (mjd, t) )
        #flux = np.concatenate( (flux, f/np.median(f)) )
        flux = np.concatenate( (flux, f/np.median(f)) )
        ferr = np.concatenate( (ferr, e) )

        # LC's may be large. Clear space after finished.
        del(d,t,f,e)

    # Save LC to file for easy access later.
    lcname = str(targetname) + "_tess.txt"
    with open(lcname,'w') as ofile:
        for k in range(len(mjd)):
            ofile.write(str(mjd[k]) + "  " + str(flux[k]) + "  " + str(ferr[k]) + "\n")

    return(mjd,flux,ferr)


def get_ft(mjd,flux,ferr):
    # Use astropy.timeseries.LombScargle to obtain LS periodogram
    # Creates periodogram and model.


    # Perform three iterations of 3-sigma clipping
    for k in range(3):
      index = np.where( (flux < 3*np.std(flux)+np.median(flux)) & (flux > -3*np.std(flux) + np.median(flux)) )
      mjd  = mjd[index]
      flux = flux[index]
      ferr = ferr[index]

    # Define minimum and maximum period (in days) to search.
    # Define step size for frequency grid.
    minp = (60.)*(1./60.)*(1./24.)
    maxp = (1440.)*(1./60.)*(1./24.)
    stepsize = 0.0012342821877648902  # 1 minute stepsize
    stepsize /= 1.

    freq_grid = np.arange(1/maxp,1/minp,stepsize)

    t1 = np.min(mjd)
    t2 = np.max(mjd)
    npts = (t2-t1)/(24*60*6000)
    times = np.arange(t1,t2,npts)
    nterms = 1 # Number of sine-terms to use to find best-period.
    #start = time.time()
    power = LombScargle(mjd,flux,dy=ferr,center_data=True,nterms=nterms).power(freq_grid)
    peak_freq = freq_grid[np.where(power==np.max(power))][0]
    model = LombScargle(mjd,flux,dy=ferr,center_data=True,nterms=1).model(times,peak_freq)
    amp = abs(np.median(model)-np.max(model))
    #print(f"Peak Frequency ({nterms} terms): {np.round(peak_freq,8)} ; {np.round((time.time() - start)/60.,3)} minutes.")

    return(mjd,flux,ferr,freq_grid,power,peak_freq,times,model,amp)


def get_phase(mjd,peak_freq,times,model):
    # Create phase points for data and model.
    # Sort the model points so it appears as a smooth line with ax.plt()
    # Aside from eleanor finding/downloading/creating LC data, this
    #   is by far the slowest part of the code.

    # Assign phase to each flux point for phase-folded light curve.
    # Uses best-fit frequency from astropy.timeseries.LombScargle
    phase = []
    for j in mjd:
        phase.append( (j-mjd[0])*peak_freq - int( (j-mjd[0])*peak_freq   ) )
    phase = np.array(phase)
    mphase = []
    for j in times:
        mphase.append( (j-times[0])*peak_freq - int( (j-times[0])*peak_freq   ) )
    mphase = np.array(mphase)

    # Create phased model to overplot on phase data. Needs to be sorted for ax.plot()
    # There is probably a better way to sort 2D arrays, but I can't think of it.
    m = np.array([np.concatenate((mphase,mphase+1)),np.concatenate((model,model))]).T
    smodel  = np.array(sorted(m,key=lambda l:l[0])).T[1][0::int(len(times)/len(mjd))]
    smphase = np.array(sorted(m,key=lambda l:l[0])).T[0][0::int(len(times)/len(mjd))]

    return(phase,smphase,smodel)

def get_plot(mjd,flux,ferr,phase,freq_grid,power,peak_freq,mphase,model):
    # Create the 3-part plot.
    # Plots are:
    #   1: TESS flux plot
    #   2: LS Periodogram
    #   3: TESS phase folded LC. folded to highest peak from LS periodogram
    #      Includes best-fit model overplotted as well

    global targetname


    # Plot everything.
    fig,[ax1,ax2,ax3] = plt.subplots(nrows=3,ncols=1)
    plt.subplots_adjust(hspace=0.5)
    #ax1.errorbar(mjd, flux, yerr=ferr, color="black", alpha=0.25,linestyle="None",capsize=2,elinewidth=1,marker=".",markersize=1)
    ax1.scatter(mjd, flux, color="black", alpha=0.25,linestyle="None",marker=".",s=2)
    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlabel("BJD ($-$2457000 days)")

    label = str(np.round(peak_freq,5)) + " cycles/day"
    ax2.plot(freq_grid,power,color="black", alpha=1.0,linewidth=1,label=label)
    #ax2.axhline(y=4*np.mean(power),xmin=0,xmax=1,color="crimson",linestyle="--",linewidth=1)
    #ax2.axhline(y=5*np.mean(power),xmin=0,xmax=1,color="crimson",linestyle="-",linewidth=1)
    ax2.set_ylabel("LS Power")
    ax2.set_xlabel("Frequency (cycles/day)")
    if True:
        # Put a global flag here for optional inset plotting.

        # https://matplotlib.org/1.3.1/mpl_toolkits/axes_grid/users/overview.html#insetlocator
        # https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html
        # https://matplotlib.org/mpl_toolkits/axes_grid/api/inset_locator_api.html?highlight=mark_inset#mpl_toolkits.axes_grid1.inset_locator.mark_inset
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        axins = zoomed_inset_axes(ax2,zoom=30,bbox_to_anchor=(.25,0.5,.2,0.5),
                                  bbox_transform=ax2.transAxes)
        axins.plot(freq_grid,power,color="black", alpha=1.0,linewidth=1)
        axins.set_xlim(7.5,7.7)
        axins.set_ylim(0,0.005)
        mark_inset(ax2, axins, loc1=3, loc2=4, fc='none',linewidth=0.5)
        #axins.set_yticks([])
    ax2.plot(freq_grid,power,color="black", alpha=1.0,linewidth=1)
    ax2.legend()

    ax3.scatter(np.concatenate((phase,phase+1)), np.concatenate((flux,flux)),color='black',alpha=0.25,s=2)
    ax3.plot(mphase,model,color='red',alpha=1.0,linewidth=1)
    ax3.set_ylabel("Normalized Flux")
    ax3.set_xlabel("Phase")
    #ax3.set_ylim((0.995,1.005))
    #plt.suptitle(targetname)
    figname = str(targetname) + "_tess.pdf"
    plt.savefig(figname,dpi=500)

def get_boots(mjd,flux,ferr):
    # Create bootstrapped datasets for error estimation.
    # Use numpy.random.randint to randomly choose N points (with replacement)
    #   from the original data.
    # This is used because I don't know how to get a covariance matrix for
    #   uncertainties in amplitude and frequency from astropy...

    ind = np.random.randint(0,len(mjd),len(mjd))
    mjd  = np.array(mjd[ind])
    flux = np.array(flux[ind])
    ferr = np.array(ferr[ind])

    return(mjd,flux,ferr)

def run_apls(dataset):
    # Run AstroPy LombScargle
    # Takes in a dataset being handled by a process of multiprocessing
    # Each dataset was created in the function get_boots()

    mjd,flux,ferr = dataset
    mjd,flux,ferr,freq_grid,power,peak_freq,times,model,amp = get_ft(mjd,flux,ferr)

    return(peak_freq,amp)


if __name__ == '__main__':

    targetname = "0642m5605"
    ra0 = "06:42:07.99"
    dec0 = "-56:05:47.44"
    ra = float(RA2Decimal(ra0))
    dec = float(Dec2Decimal(dec0))
    print(ra,dec)

    if os.path.isfile(str(targetname) + "_tess.txt"):
        mjd0,flux0,ferr0 = np.loadtxt(str(targetname) + "_tess.txt",unpack=True,dtype=np.float64)
    else:
        mjd0,flux0,ferr0 = get_lc(ra,dec)

    bootstrap = False # Estimate amplitude/frequency errors using bootstrapping?
    if bootstrap:
        # Took like an hour to run 10000 bootstraps with 4 cores and 12k datapoints
        datasets = []
        for k in range(10000): # 10,000 bootstrapped datasets
            mjd  = np.copy(mjd0)
            flux = np.copy(flux0)
            ferr = np.copy(ferr0)

            mjd,flux,ferr = get_boots(mjd,flux,ferr)
            datasets.append([mjd,flux,ferr])

        pool = Pool(cpu_count()-1) # Use all but one of your CPU cores
        result = pool.map(run_apls, datasets)
        result = np.array(result)
        np.savetxt('results.txt',result) # Save the output in a 2-column file: [freq, amplitude]
        print(f"Amplitude: {amp} +- {np.std(result[:,0])}")
        print(f"Frequency: {peak_freq} +- {np.std(result[:,1])}")

    mjd,flux,ferr,freq_grid,power,peak_freq,times,model,amp = get_ft(mjd0,flux0,ferr0)
    phase,mphase,model = get_phase(mjd,peak_freq,times,model)
    get_plot(mjd,flux,ferr,phase,freq_grid,power,peak_freq,mphase,model)
