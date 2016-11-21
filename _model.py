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
import scipy as sp

def _model(f, p, h, a):
	w0=2*np.pi/p
	zeroes=np.array([0,0])
	poles=np.array([w0*(-h+np.sqrt(1-h**2)*1j), w0*(-h-np.sqrt(1-h**2)*1j)])
	(n, d)=sp.signal.zpk2tf(zeroes, poles, a)
	(_w, h)=sp.signal.freqs(n, d, worN=2*np.pi*f)
	return h

def modelPhase(f, p, h, a):
	return np.angle(_model(f, p, h, a),deg=True)

def modelAmp(f, p, h, a):
	return np.absolute(_model(f, p, h, a))
