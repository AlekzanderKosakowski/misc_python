import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
#import matplotlib ; matplotlib.use('TkAgg') # To allow interactive. Not needed for automated plot saving.
#from matplotlib import rc
#rc('font',**{'family':['DejaVu Sans']})
#rc('text', usetex=True)
import os
import pickle
import catsHTM
from Coordinates import RA2Decimal,Dec2Decimal,Decimal2RA,Decimal2Dec

def get_abs_mag(app_mag, app_mag_err, para, para_err):
  app_mag_err = 0
  abs_mag = app_mag + 5 + 5*np.log10(para/1000)
  abs_mag_err = np.sqrt( app_mag_err**2 + (5*para_err/para/np.log(10))**2 )
  return(abs_mag, abs_mag_err)

def fancy_plot(data, ra0, dec0, fgrid, peak_freq, power, amp, save_fig):
  ra = float(ra0)
  dec = float(dec0)
  min_jd = np.min(data[0])

  zg = np.where(data[:,3] == 1.)[0] ; gcolor = 'darkgreen'
  zr = np.where(data[:,3] == 2.)[0] ; rcolor = 'crimson'
  zi = np.where(data[:,3] == 3.)[0] ; icolor = 'orange'

  fig = pickle.load(open("gaia_cmd.pkl", 'rb'))
  ax = fig.axes

  canvas = FigureCanvas(fig)

  ax[0].errorbar(data[zg,0]-min_jd, data[zg,1], yerr=data[zg,2], linestyle='None', color=gcolor, alpha=0.4, marker='.', elinewidth=1, capsize=2, markersize=2, label='ZTF g')
  ax[0].errorbar(data[zr,0]-min_jd, data[zr,1], yerr=data[zr,2], linestyle='None', color=rcolor, alpha=0.4, marker='.', elinewidth=1, capsize=2, markersize=2, label='ZTF r')
  ax[0].errorbar(data[zi,0]-min_jd, data[zi,1], yerr=data[zi,2], linestyle='None', color=icolor, alpha=0.4, marker='.', elinewidth=1, capsize=2, markersize=2, label='ZTF i')
  ax[1].plot(fgrid, power, color='black', alpha=1., linewidth=0.3, label=f"Peak={peak_freq:>7.3f}")
  ax[2].errorbar(data[zg,4], data[zg,1], yerr=data[zg,2], linestyle='None', color=gcolor, alpha=0.4, marker='.', elinewidth=1, capsize=2, markersize=2, label='ZTF g')
  ax[2].errorbar(data[zr,4], data[zr,1], yerr=data[zr,2], linestyle='None', color=rcolor, alpha=0.4, marker='.', elinewidth=1, capsize=2, markersize=2, label='ZTF r')
  ax[2].errorbar(data[zi,4], data[zi,1], yerr=data[zi,2], linestyle='None', color=icolor, alpha=0.4, marker='.', elinewidth=1, capsize=2, markersize=2, label='ZTF i')

  ra_title = ':'.join(Decimal2RA(ra))
  dec_title = ':'.join(Decimal2Dec(dec))
  ax[0].set_xlabel(f"MHJD (days)")
  #ax[0].set_title(f"Amplitude $\approx{amp:>4.2f}~mag$.  Period $\approx{1440./peak_freq:6.2f}~min$.", loc='right')
  ax[0].set_title(r"Amplitude $\approx$ "+f"{amp:>4.2f} mag.  Period "+r"$\approx$ "+f"{1440./peak_freq:6.2f} min.", loc='right')
  ax[3].set_title(f"{ra_title} {dec_title}")
#  ax[3].set_xlim(-1,+5)
#  ax[3].set_ylim(0, 17.5)
  ax[0].invert_yaxis()
  ax[1].legend(loc='upper right')
  ax[2].invert_yaxis()
  ax[3].invert_yaxis()

  radius = 2.5
  cat_path = "/lustre/research/kupfer/catalogs"
  cat, colcell, colunits = catsHTM.cone_search('GAIAEDR3', np.radians(ra), np.radians(dec), radius, catalogs_dir=cat_path, verbose=False)
  if len(cat) != 0:
    nans = np.array([np.sum(np.isnan(k)) for k in np.array(cat)])
    object_guess = np.where(nans == np.min(nans))[0] # Sometimes there are >1 entries for 1 object in Gaia. Here we assume that the entry with the fewest NaNs is the correct object.
    if len(object_guess > 1): # If two entries have the same number of nans, then use the first entry.
      object_guess = object_guess[0]
    results = cat[object_guess]

    bprp = results[20]-results[22] if results[20]-results[22] > -5 else 0
    bprp_err = (results[21]**2 + results[23]**2)**0.5
    parallax = results[5] if results[5] > 0.1 else 0.1
    parallax_err = results[6] if results[7] > 0 else 0
    gmag = results[18] if results[18] > 0 else 0
    gmag_err = results[19] if results[19] > 0 else 0
    absg, absg_err = get_abs_mag(gmag, gmag_err, parallax, parallax_err)

    #print(bprp, bprp_err, parallax, parallax_err, gmag, gmag_err, absg, absg_err)

    #ax[3].errorbar(bprp, absg, xerr=bprp_err, yerr=absg_err, color='red', capsize=1, elinewidth=0.1, marker='*', markersize=5, label=r"$G_{BP}-G_{RP}=$"+f"{bprp:>4.1f} +/- {bprp_err:>4.1f}\n"+r"$M_G=$"+f"{absg:>4.1f} +/- {absg_err:4.1f}")
    ax[3].errorbar(bprp, absg, xerr=bprp_err, yerr=absg_err, color='red', capsize=1, elinewidth=0.1, marker='*', markersize=5, label=f"$G_B$$_P-G_R$$_P={bprp:>4.1f}\pm{bprp_err:>4.1f}$\n$M_G={absg:>4.1f}\pm{absg_err:4.1f}$")
#    ax[3].scatter(bprp, absg, color='red', marker='*', s=15)
    ax[3].legend(loc='upper right')

    ra_name = "".join(ra_title.split(':')[:2]) + str(int(float(ra_title.split(':')[-1]))).zfill(2)
    dec_name = "".join(dec_title.split(':')[:2]).replace('+','p').replace('-','m') + str(int(float(dec_title.split(':')[-1]))).zfill(2)
    if save_fig:
      figdir = "./figs_als/"
      if not os.path.isdir(figdir):
          os.system(f"mkdir {figdir}")
      #plt.savefig(f'{figdir}{ra_name}{dec_name}_als.jpg', dpi=200)
      #canvas.print_figure(f'{figdir}{ra_name}{dec_name}_als.jpg', dpi=200)
      canvas.print_figure(f'{figdir}{ra_name}{dec_name}_als.jpg')
  plt.close(fig)
#  plt.show()


if __name__ == "__main__":
  ra, dec = "08:22:39.54","+30:48:57.19"
  fancy_plot([[1,2,3],[1,1,1],[0.1,0.1,0.1]], ra, dec)
