#!/usr/bin/python
# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from obspy.clients.arclink import Client
import numpy as np
    
def transMatrix(res_stream, mon_stream, fmin=0.1, fmax=10):

    #Selection des traces
    mon_stream.detrend('demean')
    mon_stream.detrend('linear')
    mon_stream.taper(0.05)
    mon_stream.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True, corners=2)

    res_stream.detrend('demean')
    res_stream.detrend('linear')
    res_stream.taper(0.05)
    res_stream.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True, corners=2)

    coeff_matrix=np.matrix([mon_stream[0].data,mon_stream[1].data,mon_stream[2].data]).transpose()

    matrix=np.array([])
    for i in range(3):
        matrix=np.append(matrix,np.linalg.lstsq(coeff_matrix,res_stream[i].data)[0])
    matrix=np.reshape(matrix,(3,3))

    return matrix
