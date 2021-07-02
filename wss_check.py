from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window
import matplotlib.mlab as mlab
from statsmodels.tsa.stattools import acf
import scipy


def plot_psd_comparison(data):

    fs = 4096
    NFFT = 4 * fs           # Use 4 seconds of data for each fourier transform
    NOVL = 1 * NFFT / 2     # The number of points of overlap between segments used in Welch averaging
    psd_window = scipy.signal.windows.tukey(NFFT, alpha=1./4)

    Pxx_H1, freqs = mlab.psd(data, Fs=fs, NFFT=NFFT,
                                 window=psd_window, noverlap=NOVL)
    # psd using a tukey window but no welch averaging
    tukey_Pxx_H1, tukey_freqs = mlab.psd(data, Fs=fs, NFFT=NFFT, window=psd_window)
    # psd with no window, no averaging
    nowin_Pxx_H1, nowin_freqs = mlab.psd(data, Fs=fs, NFFT=NFFT, 
                                            window=mlab.window_none)

    plt.figure(figsize=(8, 5))
    ax = plt.axes()
    # scale x and y axes
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)

    # plot nowindow, tukey, welch together 
    plt.plot(nowin_freqs, nowin_Pxx_H1,'red',label= 'No Window',
                     alpha=.8, linewidth=.7)
    # plt.plot(tukey_freqs, tukey_Pxx_H1 ,'green',label='Tukey Window',
                    #  alpha=.8, linewidth=.7)
    plt.plot(freqs, Pxx_H1,'blue',label='Tukey Window + Welch Average', alpha=.8,
                     linewidth=.7)

    # plot 1/f^2
    # give it the right starting scale to fit with the rest of the plots
    # don't include zero frequency
    inverse_square = np.array(list(map(lambda f: 1 / (f**2), 
                                    nowin_freqs[1:])))
    # inverse starts at 1 to take out 1/0
    scale_index = 500 # chosen by eye to fit the plot
    scale = nowin_Pxx_H1[scale_index]  / inverse_square[scale_index]
    plt.plot(nowin_freqs[1:], inverse_square * scale,'green',
                label= r'$1 / f^2$', alpha=.8, linewidth=1)

    plt.axis([20, 512, 1e-47, 1e-40])
    plt.ylabel(r'S$_n$(t)')
    plt.xlabel('Freq (Hz)')
    # ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', direction='out')
    # ax.minorticks_on()
    plt.legend(loc='best')
    plt.title('PSD for the full 4096 seconds')
    plt.show()

    acf_array = acf(Pxx_H1, fft=True, nlags=50)
    acf_nowin = acf(nowin_freqs, fft=True, nlags=50)
    plt.plot(acf_array, label='Windowed')
    plt.plot(acf_nowin, label='Leakage')
    plt.ylabel('Normalized ACF')
    plt.xlabel('Lag')
    plt.legend(loc='best')
    plt.title('Normalized ACF for the PSD')
    plt.show()
    return 


if __name__ == '__main__':
    t0 = 1126257415
    t1 = 1126259462.4 
    hdata = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    plot_psd_comparison(hdata)
