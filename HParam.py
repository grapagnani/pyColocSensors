#!/usr/bin/python
# -*- coding: utf-8 -*-

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
		_fig=plt.figure(figsize=(8.267,3.898),dpi=150)	
		_fig.suptitle("crossCalib")
		_fig.add_subplot(311)
		_ax1=_fig.gca()
		_fig.add_subplot(312)
		_ax2=_fig.gca()
		_fig.add_subplot(313)
		_ax3=_fig.gca()
		_ax1.semilogx(f, 20*np.log10(Ar), 'r-')
		_ax1.semilogx(f, 20*np.log10(Am), 'g-')
		_ax1.set_xlim([f[imin], np.amax(f)])
		_ax1.grid()
		_ax1.set_ylabel("Amplitude (dB)")
		_ax2.semilogx(f, Pr, 'r-')
		_ax2.semilogx(f, Pm, 'g-')
		_ax2.set_xlim([f[imin], np.amax(f)])
		_ax2.grid()
		_ax2.set_ylabel("Phase (deg)")
		_ax2.set_xlabel("Frequence (Hz)")
		_ax3.semilogx(f, C, 'r-')
		_ax3.grid()
		_ax3.set_ylabel("Coherence")
		_ax3.set_xlabel("Frequence (Hz)")
		_fig.savefig("HParam_output.png",dpi=150)
		plt.close(_fig)	
	
	return (popt[0], popt[1], popt[2])
