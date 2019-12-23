# note that for this module to be found, stglib\__init__.py needs to include puv_lomb in from .core import
import scipy.signal as spsig
from astropy.timeseries import LombScargle
import numpy as np
from stglib.core.waves import qkfs
from stglib.core.waves import plot_spectra
from stglib.core.waves import test_variances
import xarray as xr
import matplotlib.pyplot as plt

def puv_lomb(time, pressure, u, v, depth, height_of_pressure, height_of_velocity, sampling_frequency,
              rho=1025., first_frequency_cutoff=1 / 50, infra_gravity_cutoff=0.05,
              last_frequency_cutoff=1 / 5, frequency_resolution=0.001,
              show_diagnostic_plot=False, check_variances=False, variance_error=0.0,
              overlap_length='default'):
    """
    Determine wave heights from pressure, east_velocity, v velocity data

    Parameters
    ----------
    time : array_like
        sample time stamps in seconds
    pressure : array_like
        pressure (dbar)
    u :  array_like
        u velocities (m/s)
    v :  array_like
        v velocities (m/s)
    depth : float
        Average water depth (m, positive number)
    height_of_pressure : float
        Height of pressure sensor off bottom (m, positive number)
    height_of_velocity : float
        Height of velocity sensor off bottom (m, positive number)
    sampling_frequency : float
        Hz
    rho : float
        Water density (kg/m^3)
    first_frequency_cutoff : float
        Low-frequency cutoff for wave motions
    infra_gravity_cutoff : float
        Infra-gravity wave frequency cutoff
    last_frequency_cutoff : float
        High-frequency cutoff for wave motions
    frequency_resolution : float
        resolution desired, Hz, for precision of Lomb Scargle computation
    show_diagnostic_plot : bool
        print plots and other checks
    check_variances : bool
        test to see if variance is preserved in calculations
    variance_error : float
        tolerance for variance preservation, in percent
    overlap_length : str "default" or int length, default will result in fft_length / 2

    Returns
    -------
    dict::
        'Hrmsp': Hrms (=Hmo) from pressure
        'Hrmsu': Hrms from u,v
        'ubr': Representative orbital velocity amplitude in freq. band
            ( first_frequency_cutoff <= f <= last_frequency_cutoff ) (m/s)
        'omegar': Representative orbital velocity (radian frequency)
        'Tr': Representative orbital velocity period (s)
        'Tpp': Peak period from pressure (s)
        'Tpu': Peak period from velocity (s)
        'phir': Representative orbital velocity direction (angles from x-axis, positive ccw)
        'azr': Representative orb. velocity direction (deg; geographic azimuth; ambiguous =/- 180 degrees)
        'ublo': ubr in freq. band (f <= first_frequency_cutoff) (m/s)
        'ubhi': ubr in freq. band (f >= last_frequency_cutoff) (m/s)
        'ubig': ubr in infra-gravity freq. band (first_frequency_cutoff f <= 1/20) (m/s)
        'figure': figure handle
        'axis': axis handle
        'variance_test_passed': True if passing
        'freq': frequencies
        'Gpp': pressure power spectrum
        'Guu': East velocity power spectrum
        'Gvv': North velocity power spectrum

    References
    ----------
    Madsen (1994) Coastal Engineering 1994, Proc., 24th, Intl. Conf., Coastal Eng. Res. Council / ASCE. pp.384-398.
        (esp. p. 395)
    Thorton & Guza

    Acknowledgements
    ----------------
    converted to python and updated by Marinna Martini from Chris Sherwood's puvq.m.
    puvq.m also had contributions from Laura Landerman and Patrick Dickudt
    """

    gravity = 9.81  # m/s^2

    # pressure = spsig.detrend(pressure)
    # u = spsig.detrend(u)
    # v = spsig.detrend(v)

    # compute wave height from velocities

    # with Lomb Scargle, we should use the entire time series in one block.
    # fft_length is needed for later calculations, not for Lomb Scargle periodogram
    fft_length = len(time)

    # we want to provide a probably set of frequencies to examine.
    # http: // jakevdp.github.io / lombscargle / lombscargle / quickstart.html
    # "Note that the fastest Lomb-Scargle implementation requires regularly-spaced frequencies;
    # if frequencies are irregularly-spaced, a slower method will be used instead."
    number_of_frequencies = (last_frequency_cutoff-first_frequency_cutoff)/frequency_resolution
    frequencies = np.linspace(first_frequency_cutoff, last_frequency_cutoff, number_of_frequencies)

    # Determine velocity spectra for u and v
    # [frequencies, Gpp] = spsig.welch(rho * gravity * pressure, fs=sampling_frequency, window=fft_window_type,
    #                                nperseg=fft_length, noverlap=overlap_length)
    # frequencies, Gpp = LombScargle(time, rho * gravity * pressure).autopower()
    Gpp = LombScargle(time, rho * gravity * pressure).power(frequencies)

    # df = frequencies[2] - frequencies[1]
    df = frequency_resolution
    # [_, Guu] = spsig.welch(u, fs=sampling_frequency, window=fft_window_type, nperseg=fft_length,
    #                       noverlap=overlap_length)
    # [_, Gvv] = spsig.welch(v, fs=sampling_frequency, window=fft_window_type, nperseg=fft_length,
    #                       noverlap=overlap_length)
    # _, Guu = LombScargle(time, u).autopower()
    Guu = LombScargle(time, u).power(frequencies)
    # _, Gvv = LombScargle(time, v).autopower()
    Gvv = LombScargle(time, v).power(frequencies)

    # determine wave number
    # omega = angular wave frequency
    omega = np.array([2 * np.pi * x for x in frequencies])  # omega must be numpy array for qkfs
    # catch numpy errors
    np.seterr(divide='ignore', invalid='ignore')
    k = qkfs(omega, float(depth))  # make sure it is float, or qkfs will bomb
    np.seterr(divide=None, invalid=None)

    # compute linear wave transfer function
    kh = k * depth
    kzp = k * height_of_pressure
    kzuv = k * height_of_velocity
    nf = len(omega)
    Hp = np.ones(nf)
    Huv = np.ones(nf)

    # change wavenumber at 0 Hz to 1 to avoid divide by zero
    # i = np.array(range(nf))  # this is an index, thus needs to start at first element, in this case 0
    # for some reason in the MATLAB version CRS tests omega for nans instead of k.
    # Here we test k also because that's where the nans show up
    if np.isnan(omega[0]) or np.isnan(k[0]) or (omega[0] <= 0):  # 0 Hz is the first element
        # i = i[1:]
        i = range(1, nf)
        Hp[0] = 1
        Huv[0] = 1
    else:
        i = range(nf)

    # compute height from pressure and orbital
    # I was getting Hmo far too large.
    # from my doNDwavesLomb.m, Sandwich mooring 10811
    # in development for the CWTM paper, I have this note:
    #   % confirmed: Chris' desired transfer function ratio is instrument depth / site depth
    #   % thus when the pressure spectra are divided by the transfer
    #   % function, energy should be increased (e.g. Hp < 1).
    #   % Gpp is normally the output of p_welch.m, now living in advlib
    #   % where p is p*rho*g but we don't do that with Lomb so omit rho*g here
    #   % this should be OK because in Welch the energy will be
    #   % squared, and in the denominator below, it is also squared
    #   % Hp(i,1) = rho*g .* ( cosh(kzp(i,1)) ./ cosh(kh(i,1)) );
    #   % confirmed: rho*g not needed here, we are using pressure data

    Hp[i] = rho * gravity * (np.cosh(kzp[i]) / np.cosh(kh[i]))
    # Hp = np.cosh(kzp) / np.cosh(kh)
    # Hp[i] = np.cosh(kzp[i]) / np.cosh(kh[i])
    # Huv[i] = omega[i] * (np.cosh(kzuv[i]) / np.sinh(kh[i]))
    # Huv = omega * (np.cosh(kzuv) / np.sinh(kh))
    Huv[i] = omega[i] * (np.cosh(kzuv[i]) / np.sinh(kh[i]))

    # combine horizontal velocity spectra
    Guv = Guu + Gvv

    # with Lomb Scargle power(), we defined the cutoff and frequency domain from the start
    # create cut off frequency, so noise is not magnified
    # at least in first testing, subtracting 1 here got closer to the intended freq. cutoff value
    # ff = np.argmax(frequencies > first_frequency_cutoff) - 1
    ff = 0
    # lf = np.argmax(frequencies > last_frequency_cutoff)
    lf = len(frequencies)-1
    # print(f'ff = {ff}')
    # print(f'lf = {lf}')

    # Determine wave height for velocity spectra
    Snp = Gpp[ff:lf] / (Hp[ff:lf] ** 2)
    Snu = Guv[ff:lf] / (Huv[ff:lf] ** 2)
    fclip = frequencies[ff:lf]

    # Determine rms wave height (multiply by another sqrt(2) for Hs)
    # Thornton and Guza say Hrms = sqrt(8 mo)
    Hrmsu = 2 * np.sqrt(2 * np.sum(Snu * df))
    Hrmsp = 2 * np.sqrt(2 * np.sum(Snp * df))

    # These are representative orbital velocities for w-c calculations,
    # according to Madsen (1994) Coastal Engineering 1994, Proc., 24th
    # Intl. Conf., Coastal Eng. Res. Council / ASCE. pp.384-398.
    # (esp. p. 395)
    ubr = np.sqrt(2 * np.sum(Guv[ff:lf] * df))
    ubr_check = np.sqrt(2 * np.var(u) + 2 * np.var(v))
    omegar = np.sum(omega[ff:lf] * Guv[ff:lf] * df) / np.sum(Guv[ff:lf] * df)
    Tr = 2 * np.pi / omegar

    if len(np.where(np.isnan(Snp))) > 0 | len(np.where(Snp == 0)) > 0:
        Tpp = np.nan
    else:
        jpeak = np.argmax(Snp)  # index location of the maximum value
        Tpp = 1 / fclip[jpeak]

    if len(np.where(np.isnan(Snu))) > 0 | len(np.where(Snu == 0)) > 0:
        Tpu = np.nan
    else:
        jpeak = np.argmax(Snu)
        Tpu = 1 / fclip[jpeak]

    # phi is angle wrt to x axis; this assumes Guu is in x direction
    # phir = atan2( sum(Guu(ff:lf)*df), sum(Gvv(ff:lf)*df) );

    # this is the line changed on 6/24/03 - I still think it is wrong (CRS)
    # phir = atan2( sum(Gvv(ff:lf)*df), sum(Guu(ff:lf)*df) );

    # This is Jessie's replacement for direction
    # 12/08 Jessie notes that Madsen uses velocity and suggests
    # Suu = sqrt(Guu);
    # Svv = sqrt(Gvv);
    # Suv = sqrt(Guv);
    # but I (CRS) think eqn. 24 is based on u^2, so following is ok:
    rr = np.corrcoef(u, v)
    ortest = np.sign(rr[1][0])
    phir = np.arctan2(ortest * np.sum(Gvv[ff:lf] * df), np.sum(Guu[ff:lf] * df))

    # convert to degrees; convert to geographic azimuth (0-360, 0=north)
    azr = 90 - (180 / np.pi) * phir

    # Freq. bands for variance contributions
    ig = np.max(np.where(frequencies <= infra_gravity_cutoff))
    # low freq, infra-gravity, high-freq
    if 1 < ff:
        ublo = np.sqrt(2 * np.sum(Guv[1:ff] * df))
    else:
        ublo = 0
    if ig > ff:
        ubig = np.sqrt(2 * np.sum(Guv[ff:ig] * df))
    else:
        ubig = 0
    if lf < fft_length:
        ubhi = np.sqrt(2 * np.sum(Guv[lf:] * df))
    else:
        ubhi = 0

    ws = {
        'Hrmsp': Hrmsp,
        'Hrmsu': Hrmsu,
        'ubr': ubr,
        'ubr_check': ubr_check,
        'omegar': omegar,
        'Tr': Tr,
        'Tpp': Tpp,
        'Tpu': Tpu,
        'phir': phir,
        'azr': azr,
        'ublo': ublo,
        'ubhi': ubhi,
        'ubig': ubig,
        'freq': frequencies,
        'Gpp': Gpp,
        'Guu': Guu,
        'Gvv': Gvv
    }

    if check_variances:
        variance_preserved = test_variances(u, v, pressure, Gpp, Guu, Gvv, df, allowable_error=variance_error)
        ws['variance_test_passed'] = variance_preserved

    if show_diagnostic_plot:
        fig, ax = plot_spectra(Guu, Gvv, Guv, Gpp, frequencies, first_frequency_cutoff, ff, last_frequency_cutoff, lf,
                               infra_gravity_cutoff, ig)
        ws['figure'] = fig
        ws['axis'] = ax

    return ws