from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, kstest
from scipy.signal import correlate

def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


if __name__=='__main__':
    t0 = 1126257415
    t1 = 1126259462.4 
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    c = int(len(data)/100000)
    kstest(data[c*2:c*3], data[c:2*c])

    for k in np.array_split(data, 5):
        for j in np.array_split(data, 5):
            kstest(k, j)

    data1 = data

    print(np.mean(data[:c]))
    print(np.std(data[:c]))
    print(skew(data[:c]))
    print(np.mean(data[c:]))
    print(np.std(data[c:]))
    print(skew(data[c:]))
    acf_array = acf(data, fft=True, nlags=10000)
    plt.plot(acf_array)
    plt.show()

    corr = correlate(data, data1, method='fft')

    strain = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')
    specgram = strain1.spectrogram2(fftlength=4, overlap=2, window='hann') ** (1/2.)
    plot = specgram.plot()
    ax = plot.gca()
    ax.set_yscale('log')
    ax.set_ylim(10, 1400)
    ax.colorbar(
    clim=(1e-24, 1e-20),
    norm="log",
    label=r"Strain noise [$1/\sqrt{\mathrm{Hz}}$]",
    )
    center = int(t1)
    strain1 = strain.crop(center-16, center+(16))
    fig2 = strain1.asd(fftlength=8).plot()
    plt.show()
    fig1 = strain.plot()
    # plt.show()
    white_data = strain.whiten()
    bp_data = white_data.bandpass(30, 400)
    fig3 = bp_data.plot()
    plt.show()