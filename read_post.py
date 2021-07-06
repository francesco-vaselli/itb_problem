import bilby
import numpy as np 
import matplotlib.pyplot as plt 


if __name__=='__main__':

    result_short = bilby.result.read_in_result(outdir='short', label="GW150914")
    # sample="unif", nlive=500, dlogz=3  # <- Arguments are used to make things fast - not recommended for general use

    Mc = result_short.posterior["chirp_mass"].values

    lower_bound = np.quantile(Mc, 0.05)
    upper_bound = np.quantile(Mc, 0.95)
    median = np.quantile(Mc, 0.5)
    print("Mc = {} with a 90% C.I = {} -> {}".format(median, lower_bound, upper_bound))
    '''
    fig, ax = plt.subplots()
    ax.hist(result_short.posterior["chirp_mass"], bins=50)
    ax.axvspan(lower_bound, upper_bound, color='C1', alpha=0.4)
    ax.axvline(median, color='C1')
    ax.set_xlabel("chirp mass")
    plt.show()
    '''

    result_short.plot_corner(parameters=["chirp_mass", "mass_ratio"], prior=True, save=False)
    # plt.show()

    parameters = dict(mass_1=36.2, mass_2=29.1)
    fig = result_short.plot_corner(parameters, save=False)

    # result_short.plot_walkers(parameters=["chirp_mass", "mass_ratio"], prior=True, save=False)
    plt.show()