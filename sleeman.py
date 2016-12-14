#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Function calculating electronic noise of 3 traces recording strictly the \
    same signal. This function uses sleeman's method (doi: 10.1785/0120050032).

:author:
    Maxime Bès de Berc (mbesdeberc@unistra.fr)

:copyright:
    Maxime Bès de Berc (mbesdeberc@unistra.fr)

:license:
    The Beerware License
    (https://tldrlegal.com/license/beerware-license)
"""
import numpy as np
from matplotlib import mlab
from obspy.signal.util import prev_pow_2
from obspy.signal.invsim import cosine_taper

def detrend_func(data):
    data = mlab.detrend_mean(data)
    data = mlab.detrend_linear(data)
    return data

def sleeman(stream):
    """
    Sleeman method calculating the noise of each trace of input stream.
    :type stream: Stream object from module obspy.core.stream
    :param stream: Stream containing 3 traces with the same signal recorded.
    """

    #Copy original stream
    m = stream.copy()

    #Verify stream
    if m[0].stats.sampling_rate != m[1].stats.sampling_rate \
    or m[0].stats.sampling_rate != m[2].stats.sampling_rate \
    or m[1].stats.sampling_rate != m[2].stats.sampling_rate:
        print("[pyColocSensors.sleeman]: Sampling rates are not identical \
        between traces")

    if m[0].stats.npts != m[1].stats.npts \
    or m[0].stats.npts != m[2].stats.npts \
    or m[1].stats.npts != m[2].stats.npts:
        print("[pyColocSensors.sleeman]: Traces does not have the same length")

    if m[0].stats.starttime-m[1].stats.starttime  >= m[0].stats.sampling_rate/2\
    or m[1].stats.starttime-m[2].stats.starttime  >= m[1].stats.sampling_rate/2\
    or m[0].stats.starttime-m[2].stats.starttime  >= m[0].stats.sampling_rate/2:\
        print("[pyColocSensors.sleeman]: Traces does not have the same start time")

    #Set psd and csd parameters. Set as McNamara recommands in his paper \
    #(doi: 10.1785/012003001)
    n_fft = 1024
    n_overlap = int(n_fft*0.75)
    fs = m[0].stats.sampling_rate

    #Calculate psd and csd
    (P00,f) = mlab.psd(m[0].data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,\
    detrend=detrend_func,window=cosine_taper(n_fft,p=0.2),scale_by_freq=True)

    (P11,f) = mlab.psd(m[1].data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,\
    detrend=detrend_func,window=cosine_taper(n_fft,p=0.2),scale_by_freq=True)

    (P22,f) = mlab.psd(m[2].data,Fs =fs,NFFT=n_fft,noverlap=n_overlap,\
    detrend=detrend_func,window=cosine_taper(n_fft,p=0.2),scale_by_freq=True)

    (P01,f) = mlab.csd(m[0].data,m[1].data, Fs=fs,NFFT=n_fft,noverlap=n_overlap,\
    detrend=detrend_func,window=cosine_taper(n_fft,p=0.2),scale_by_freq=True)

    (P02,f) = mlab.csd(m[0].data,m[2].data, Fs=fs,NFFT=n_fft,noverlap=n_overlap,\
    detrend=detrend_func,window=cosine_taper(n_fft,p=0.2),scale_by_freq=True)

    P10 = np.conj(P01)

    (P12,f) = mlab.csd(m[1].data,m[2].data, Fs=fs,NFFT=n_fft,noverlap=n_overlap,\
    detrend=detrend_func,window=cosine_taper(n_fft,p=0.2),scale_by_freq=True)

    P20 = np.conj(P02)

    P21 = np.conj(P12)

    #Apply corrections as McNamara recommends
    for spectra in [P00,P11,P22,P01,P02,P10,P12,P20,P21]:
        spectra = spectra*1.142857

    #Calculate electronic noises acoording to Sleeman's method
    N0 = P00-P10*P02/P12
    N1 = P11-P21*P10/P20
    N2 = P22-P02*P21/P01

    #Remove first samples (3%) in order to avoid poor resolution problems.
    #Samples remaining should be enough to cover one decade in frequency.
    N0 = N0[int(n_fft*0.03):]
    N1 = N1[int(n_fft*0.03):]
    N2 = N2[int(n_fft*0.03):]
    f = f[int(n_fft*0.03):]

    return (N0,N1,N2,f)
