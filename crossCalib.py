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

:author:
    Maxime Bès de Berc (mbesdeberc@unistra.fr)

:copyright:
    Maxime Bès de Berc (mbesdeberc@unistra.fr)

:license:
    The Beerware License
    (https://tldrlegal.com/license/beerware-license)
"""

from obspy.core import UTCDateTime,Stream
from obspy.clients.arclink import Client
from HParam import HParam
from matplotlib import mlab
import numpy as np
import scipy.signal as sp
import shutil as sh


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
		n_fft=1024

	if 'npad' in kwargs:
		n_pad=kwargs['npad']
	else:
		n_pad=n_fft*4

	if 'noverlap' in kwargs:
		n_overlap=kwargs['noverlap']
	else:
		n_overlap=int(n_fft*0.90)

	#paz par défaut: chaine générique
	if 'paz' in kwargs:
		paz=kwargs['paz']
	else:
		paz=dict()
		paz['zeros']=np.array([])
		paz['poles']=np.array([])
		paz['gain']=1
		paz['seismometer_gain']=1
		paz['datalogger_gain']=1
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
