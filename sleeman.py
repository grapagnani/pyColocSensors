#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
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

def sleeman(stream, **kwargs):

	m=stream.copy()

	if 'nfft' in kwargs:
		n_fft=kwargs['nfft']
	else:
		n_fft=1024

	if 'npad' in kwargs:
		n_pad=kwargs['npad']
	else:
		n_pad=n_fft*4

	if 'noverlap' in kwargs:
		n_overlap=kwargs['noverlap']
	else:
		n_overlap=int(n_fft*0.90)

	fs=m[0].stats.sampling_rate

	(P00,f)=mlab.psd(m[0].data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	(P11,f)=mlab.psd(m[1].data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	(P22,f)=mlab.psd(m[2].data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	(P01,f)=mlab.csd(m[0].data,m[1].data, Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	(P02,f)=mlab.csd(m[0].data,m[2].data, Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	P10=np.conj(P01)
	(P12,f)=mlab.csd(m[1].data,m[2].data, Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	P20=np.conj(P02)
	P21=np.conj(P12)

	N0=P00-P10*P02/P12
	N1=P11-P21*P10/P20
	N2=P22-P02*P21/P01
	
	return (N0,N1,N2,f)
 