import numpy as np
import matplotlib.pyplot as plt

datafile  = "data1.txt"
modelfile = "model1.txt"

data  = np.loadtxt(datafile,unpack=True,dtype=np.float64)
model = np.loadtxt(modelfile,unpack=True,dtype=np.float64)

fig = plt.figure(figsize=(18,9))
ax1 = fig.add_subplot(1,5,(1,4))
ax2 = fig.add_subplot(1,5,5,sharey=ax1)
ax1.invert_yaxis()
ax1.errorbar(data[0],data[1],yerr=data[2],marker=".",color="black",alpha=0.5,markersize=2,elinewidth=1,capsize=2,linestyle="None")
ax1.plot(model[0],model[1],color="crimson",linewidth=2)
ax1.set_ylabel("Relative Magnitude")
ax2.errorbar(data[0],data[1],yerr=data[2],marker=".",color="black",alpha=0.5,markersize=2,elinewidth=1,capsize=2,linestyle="None")
ax2.plot(model[0],model[1],color="crimson",linewidth=2)
ax2.set_xlim(0.95,1.05)
ax1.set_xlabel("Phase")
ax2.set_xlabel("Phase")
plt.show()
