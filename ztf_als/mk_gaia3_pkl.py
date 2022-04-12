import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_gaia_abs_mag(app_mag, app_mag_err, para, para_err):
  abs_mag = app_mag + 5 + 5*np.log10(para/1000)
  abs_mag_err = np.sqrt( app_mag_err**2 + (5*para_err/para/np.log(10))**2 )
  return(abs_mag, abs_mag_err)


def base_plot(data, object_name):
  #
  # Create the base for the final output plots.
  # Will have three empty light curve plots on the left and one big CMD plot on the right.
  # Here we populate only the CMD plot with Gaia eDR3 data, then save the entire figure object as a pickle file.
  # The pickle file is then read in later when we populate the three light curve plots.
  #
    gaia = np.loadtxt("gaia3_output.txt", unpack=True)
    gaia_absg = get_gaia_abs_mag(gaia[2],gaia[3],gaia[0],gaia[1])

    fig = plt.figure(figsize=(15,5))
    plt.subplots_adjust(hspace=0.5,wspace=1.5, left=0.05, right=0.95)
    ax1 = fig.add_subplot(3,16,(1,10)) # Grid of 3 Rows and 16 Columns. The first subplot fills entries 1 to 10
    ax2 = fig.add_subplot(3,16,(17,26))
    ax3 = fig.add_subplot(3,16,(33,42))
    ax4 = fig.add_subplot(3,16,(11,48)) # Fourth subplot fills entries 11 to 48
    ax1.set_ylabel("Magnitude")
    ax2.set_xlabel("Frequency (cycles/day)")
    ax2.set_ylabel("Power")
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Magnitude")
    ax4.set_xlabel(r"$G_{BP}-G_{RP}$")
    ax4.set_ylabel(r"$M_{G}$")
    # ax4.invert_yaxis()
    ax4.tick_params(direction="inout",left=True, right=True,bottom=True, top=True, labeltop=False, labelbottom=True, labelright=True, labelleft=False)
    ax4.scatter(gaia[4],gaia_absg[0],color='black',s=0.001,alpha=1)
    ofile = open("gaia_cmd.pkl", 'wb')
    pickle.dump(fig, ofile)
    # plt.show()

if __name__ == "__main__":

  test_hjd = np.linspace(1,1000,1000)
  amp = np.random.uniform(0,5)
  test_mag = amp*np.sin(2*np.pi/np.random.uniform(20,200)*test_hjd) + 18*np.random.normal(amp,0.1*amp)
  test_merr = np.abs(np.random.normal(0,1,len(test_hjd)))

  test_data = np.array([test_hjd, test_mag, test_merr])

  base_plot(test_data, 'test_object')

