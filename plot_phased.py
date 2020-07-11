import numpy as np
import matplotlib.pyplot as plt

datafile  = "data1.txt"   # File containing the data in 2 or 3 column format.
                          #   Phase running from 0 to 1
modelfile = "model1.txt"  # File containing model in 2 column format.
                          #   Phase running from 0 to 1

readdata  = np.loadtxt(datafile, unpack=True, dtype=np.float64)
readmodel = np.loadtxt(modelfile, unpack=True, dtype=np.float64)


cycles = 2  # How many times should the model repeat itself?

# Begin creating stacked data/model to plot multiple cycles in a row.
data  = [[] for x in range(len(readdata))]
model = [[] for x in range(len(readmodel))]
for k in range(cycles):
    for j in range(len(data)):
        data[j] = np.concatenate((data[j], k+readdata[j])) if j == 0 \
                  else np.concatenate((data[j], readdata[j]))
    for j in range(len(model)):
        model[j] = np.concatenate((model[j], k+readmodel[j])) if j == 0 \
                  else np.concatenate((model[j], readmodel[j]))
data, model = np.array(data), np.array(model)
data = np.array(sorted(data.T, key=lambda l:l[0])).T   # Sort final by phase.
model = np.array(sorted(model.T, key=lambda l:l[0])).T # Sort final by phase.



rows = 1    # How many rows (data+model sets) to plot in the image.
cols = 8    # Adjust this to adjust size of primary/secondary zoom-ins.
            #   The primary and secondary zoom-ins are worth 1 unit each.
            #   The full light curve is worth the remaining units.
alpha = 0.5 # How transparent the data points are from 0.0 to 1.0


fig = plt.figure(figsize=(13,5))
plt.subplots_adjust(wspace=0.05, hspace=0.5)

# ax1 = Data+Model
ax1 = fig.add_subplot(rows,cols,(1,1*cols-2))
ax1.invert_yaxis()
ax1.errorbar(data[0],data[1],yerr=data[2],marker=".",color="black",alpha=alpha,markersize=2,elinewidth=1,capsize=2,linestyle="None")
ax1.plot(model[0],model[1],color="crimson",linewidth=1.5)
ax1.set_ylabel("Relative Magnitude")
ax1.set_xlabel("Phase")
ax1.set_title("Data1 + Model1",loc='left')

# ax2 = Zoom-in of secondary eclipse
ax2 = fig.add_subplot(rows,cols,1*cols-1)
ax2.invert_yaxis()
ax2.errorbar(data[0],data[1],yerr=data[2],marker=".",color="black",alpha=alpha,markersize=2,elinewidth=1,capsize=2,linestyle="None")
ax2.plot(model[0],model[1],color="crimson",linewidth=1.5)
ax2.set_xlim(0.45,0.55)
ax2.set_xlabel("Phase")
ax2.set_xticks([0.47,0.5,0.53])
ax2.set_yticklabels([])

# ax3 = Zoom-in of primary eclipse
ax3 = fig.add_subplot(rows,cols,1*cols)
ax3.invert_yaxis()
ax3.errorbar(data[0],data[1],yerr=data[2],marker=".",color="black",alpha=alpha,markersize=2,elinewidth=1,capsize=2,linestyle="None")
ax3.plot(model[0],model[1],color="crimson",linewidth=1.5)
ax3.set_xlim(0.95,1.05)
ax3.set_xlabel("Phase")
ax3.set_xticks([0.97,1.0,1.03])
ax3.set_yticklabels([])

plt.show()
