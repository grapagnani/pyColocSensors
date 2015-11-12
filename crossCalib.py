#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Module de calcul de la fonction de transfert d'un sismomètre large-bande par comparaison spectrale de capteurs colocalisés.
crossCalib: Renvoie un tuple de array, définissant la fonction de transfert et les fréquences associées. 

Paramètres:
monitor_trace: Objet trace du module obspy.core.trace correspondant au signal de bruit blanc injecté dans le sismomètre, ou du signal enregistré sur la chaine colocalisée de référence
response_trace: Objet trace du module obspy.core.trace correspondant au signal du sismomètre (réponse de la bobine au bruit blanc)
paz: Dictionnaire poles et zéros (modèle obspy) utilisé pour la déconvolution
plotting: Booléen activant le plot de la fonction de transfert calculée

Executé directement, le fichier propose un exemple de calcul de fonction de transfert d'un T120, par un signal de bruit blanc généré par un Q330

Maxime Bès de Berc
12/11/2013
"""

from obspy.core import UTCDateTime,Stream
from obspy.arclink import Client
from HParam import HParam
from matplotlib import mlab
import numpy as np
import scipy.signal as sp


def crossCalib(monitor_trace, response_trace, **kwargs):
	
	m_trace=monitor_trace.copy()
	r_trace=response_trace.copy()

	if 'demean' in kwargs and kwargs['demean']:
		m_trace.detrend('demean')
		r_trace.detrend('demean')

	if 'taper' in kwargs and kwargs['taper']:
		m_trace.taper(0.05)
		r_trace.taper(0.05)

	#Paramètres des PSD
	if 'nfft' in kwargs:
		n_fft=kwargs['nfft']
	else:
		n_fft=4096

	if 'npad' in kwargs:
		n_pad=kwargs['npad']
	else:
		n_pad=16384

	if 'noverlap' in kwargs:
		n_overlap=kwargs['noverlap']
	else:
		n_overlap=int(n_fft*0.90)

	#paz par défaut: STR station de référence
	if 'paz' in kwargs:
		paz=kwargs['paz']
	else:
		paz=dict()
		paz['zeros']=np.array([0,0,-1.515e1,-3.186e2+4.012e2j,-3.186e2-4.012e2j])
		paz['poles']=np.array([-3.7e-2-3.7e-2j,-3.7e-2+3.7e-2j,-1.599e1,-4.171e2,-1.009e2+4.019e2j,-1.009e2-4.019e2j,-7.454e3-7.142e3j,-7.454e3+7.142e3j,-1.87239e2])
		paz['gain']=np.absolute(np.prod(2j*np.pi-paz['poles'])/np.prod(2j*np.pi-paz['zeros']))
		paz['seismometer_gain']=1500
		paz['datalogger_gain']=1677721
		paz['sensitivity']=paz['seismometer_gain']*paz['datalogger_gain']*paz['gain']

	
	fs=m_trace.stats.sampling_rate
	(P00,f)=mlab.psd(m_trace.data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	(P01,f)=mlab.csd(m_trace.data,r_trace.data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	(C,f)=mlab.cohere(m_trace.data,r_trace.data,Fs=fs,NFFT=n_fft,noverlap=n_overlap,pad_to=n_pad,detrend=mlab.detrend_mean,window=mlab.window_hanning)
	
	(b,a)=sp.zpk2tf(paz['zeros'],paz['poles'],paz['sensitivity'])
	(_w,H0)=sp.freqs(b,a,f*2*np.pi)

	H1=P01*H0/P00
	H1=H1[1:]
	C=C[1:]
	f=f[1:]

	return (H1,C,f)

if __name__=='__main__':
	client=Client(host='gavrinis.u-strasbg.fr',port=18001,user='mbesdeberc@unistra.fr')
	t0=UTCDateTime("2015-11-11T00:00:00")
	duration=4*3600
	t1=t0+duration
	st=Stream()
	st.append(client.getWaveform('FR','STR','00','BHZ',t0,t1)[0])
	st.append(client.getWaveform('XX','GPIL','00','BHZ',t0,t1)[0])
	(H,C,f)=crossCalib(st[0],st[1],demean=False,taper=False)
	(per,dam,amp)=HParam(H,C,f,plotting=True)
	print(per,dam,amp*1.0/419430)
