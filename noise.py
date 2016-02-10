#!/usr/bin/python
# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from obspy.arclink import Client
import numpy as np
from matplotlib import mlab
from matplotlib import pyplot

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

if __name__=='__main__':
		
	client=Client(host='gavrinis.u-strasbg.fr',port=18001,user='mbesdeberc@unistra.fr')
	t1=UTCDateTime("2016-01-29T06:57:00") #bruit blanc injecté dans 3 ADCs
	duration=10*60

	data=Client(host='gavrinis',port=18001,user='mbesdeberc@unistra.fr')	#ouvre un client Arclink
		
	res=data.getWaveform('XX','GP000','00','CH?',t1,t1+duration,route=False)
	res.sort()
	res.write('res.mseed',format='MSEED')
	for tr in res:
		tr.data=tr.data*2.384e-6	

	(NE,NN,NZ,f)=sleeman(res,nfft=4096)
	pyplot.semilogx(f,10*np.log10(NE))
	pyplot.semilogx(f,10*np.log10(NN))
	pyplot.semilogx(f,10*np.log10(NZ))
	
	pyplot.title(res[0].stats.network+'.'+res[0].stats.station+'.'+res[0].stats.location)
	pyplot.grid()
	pyplot.xlabel("Frequency (Hz)")
	pyplot.ylabel("Electronic Noise (dB rel to 1 V**2/Hz)")
	pyplot.legend()
	pyplot.show()



	

