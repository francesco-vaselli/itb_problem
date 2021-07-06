from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

import bilby
from bilby.gw.prior import UniformInComponentsChirpMass, UniformInComponentsMassRatio
from bilby.core.prior import Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters

from gwpy.timeseries import TimeSeries


if __name__=='__main__':
    t0 = 1126257415
    time_of_event = 1126259462.4 
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    strain = TimeSeries(data, t0=t0, sample_rate=4096, unit='strain')

    H1 = bilby.gw.detector.get_empty_interferometer("H1")

    # Definite times in relatation to the trigger time (time_of_event), duration and post_trigger_duration
    post_trigger_duration = 2
    duration = 4
    analysis_start = time_of_event + post_trigger_duration - duration

    H1_analysis_data = strain.crop(analysis_start , analysis_start+duration)

    # H1_analysis_data.plot()
    # plt.show()

    H1.set_strain_data_from_gwpy_timeseries(H1_analysis_data)

    psd_duration = duration * 32
    psd_start_time = analysis_start - psd_duration

    H1_psd_data = strain.crop(psd_start_time , psd_start_time+psd_duration)

    psd_alpha = 2 * H1.strain_data.roll_off / duration
    H1_psd = H1_psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")

    H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=H1_psd.frequencies.value, psd_array=H1_psd.value)

    H1.maximum_frequency = 1024

    prior = bilby.core.prior.PriorDict()
    prior['chirp_mass'] = UniformInComponentsChirpMass(name='chirp_mass', minimum=10.0,maximum=35.0)
    prior['mass_ratio'] = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.5, maximum=1)
    prior['phase'] = Uniform(name="phase", minimum=0, maximum=2*np.pi)
    prior['geocent_time'] = Uniform(name="geocent_time", minimum=time_of_event-0.1, maximum=time_of_event+0.1)
    prior['a_1'] =  0.0
    prior['a_2'] =  0.0
    prior['tilt_1'] =  0.0
    prior['tilt_2'] =  0.0
    prior['phi_12'] =  0.0
    prior['phi_jl'] =  0.0
    prior['dec'] =  -1.2232
    prior['ra'] =  2.19432
    prior['theta_jn'] =  1.89694
    prior['psi'] =  0.532268
    prior['luminosity_distance'] = 412.066

    # First, put our "data" created above into a list of intererometers (only H1)
    interferometers = [H1]

    # Next create a dictionary of arguments which we pass into the LALSimulation waveform - we specify the waveform approximant here
    waveform_arguments = dict(
        waveform_approximant='TaylorF2', reference_frequency=100., catch_waveform_errors=True)

    # Next, create a waveform_generator object. This wraps up some of the jobs of converting between parameters etc
    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters)

    # Finally, create our likelihood, passing in what is needed to get going
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers, waveform_generator, priors=prior,
        time_marginalization=True, phase_marginalization=True, distance_marginalization=False)
    
    result_short = bilby.run_sampler(
        likelihood, prior, sampler='dynesty', outdir='shortL', label="GW150914",
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        n_effective = 5000, dlogz=3
        )
    
    # result_short = bilby.result.read_in_result(outdir='short', label="GW150914")
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
    plt.show()