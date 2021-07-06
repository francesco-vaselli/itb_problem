import numpy as np 
import matplotlib.pyplot as plt 

peaks = np.abs(np.load('peaks.npy'))
Masses = np.load('Masses.npy')
times = np.load("times.npy")
max = np.argmax(peaks)
print(Masses[max])
print(max)
n, bins, patches = plt.hist(times, 10, density=False, facecolor='g', alpha=0.75)
plt.xlabel('GPS Times')
plt.ylabel('#')
plt.title('Histogram of times for SNR peaks')
plt.show()