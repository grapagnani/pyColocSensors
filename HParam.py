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
from _model import modelAmp,modelPhase
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def HParam(H, C, f, **kwargs):
	
	#Periode propre determinée avec le fit de la phase de la fonction de transfert avec la phase du modèle
	#Donne un résultat bien plus fiable
	#On travaille dans les fréquences inférieures à la fréquence de normalisation (souvent fn=1Hz pour les LB)
	#et à des fréquences supérieures à fmin. Celle-ci est à determiner au cas par cas en fonction des premiers résultats.
	
	#Détermination de l'indice de fnorm
	if 'fnorm' in kwargs:
		inorm=int(np.absolute(f-kwargs['fnorm']).argmin())
	else:
		inorm=int(np.absolute(f-1).argmin())

	#Détermination de l'indice de fmin
	if 'fmin' in kwargs:
		imin=int(np.absolute(f-kwargs['fmin']).argmin())
	else:
		imin=int(np.absolute(f-0.001).argmin())

	#Determination de la période de coupure, de l'amortissement et de l'amplitude de la fonction de transfert mesurée
	#Determination grossière
	i0=imin+int(np.absolute(np.angle(H[imin:inorm],deg=True)-90).argmin())
	popt=list()
	popt.append(1./f[i0])
	popt.append(0.707)
	popt.append(np.absolute(H[inorm]))

	#Determination fine
	(popt, _pcov)=sp.optimize.curve_fit(modelPhase, f[imin:inorm], np.angle(H[imin:inorm],deg=True), p0=(popt[0],popt[1],1),sigma=1./C[imin:inorm])
	(_popt, _pcov)=sp.optimize.curve_fit(modelAmp, f[imin:inorm], np.absolute(H[imin:inorm]), p0=(popt[0],popt[1],popt[2]),sigma=1./C[imin:inorm])
	popt[2]=_popt[2]

	#Fonction de transfert en amplitude normalisée et phase (mesure et modèle)
	Ar=np.absolute(H)
	Am=modelAmp(f, popt[0], popt[1], popt[2])
	Pr=np.angle(H,deg=True)
	Pm=modelPhase(f, popt[0], popt[1], popt[2])
	
	if 'plotting' in kwargs and kwargs['plotting']:
		plt.subplot(311)
		plt.semilogx(f, 20*np.log10(Ar), 'r-', label='measurement')
		plt.semilogx(f, 20*np.log10(Am), 'g-', label='best fit')
		plt.xlim([f[imin], np.amax(f)])
		plt.grid()
		plt.ylabel("Amplitude (dB)")
		plt.subplot(312)
		plt.semilogx(f, Pr, 'r-')
		plt.semilogx(f, Pm, 'g-')
		plt.xlim([f[imin], np.amax(f)])
		plt.grid()
		plt.ylabel("Phase (deg)")
		plt.subplot(313)
		plt.semilogx(f, C, 'r-')
		plt.xlim([f[imin], np.amax(f)])
		plt.grid()
		plt.ylabel("Coherence")
		plt.xlabel("Frequence (Hz)")
		plt.suptitle("Cross calibration result")
		plt.show()
	
	return (popt[0], popt[1], popt[2])
