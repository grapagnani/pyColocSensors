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


def transMatrix(res_stream, mon_stream, fmin=0.1, fmax=10):

    m_stream = mon_stream.copy()
    r_stream = res_stream.copy()

    for m in (m_stream, r_stream):

        # Verify each stream
        if m[0].stats.sampling_rate != m[1].stats.sampling_rate \
           or m[0].stats.sampling_rate != m[2].stats.sampling_rate \
           or m[1].stats.sampling_rate != m[2].stats.sampling_rate:
            print("[pyColocSensors.transMatrix]: Sampling rates are not identical \
                  between traces")

        if m[0].stats.npts != m[1].stats.npts \
           or m[0].stats.npts != m[2].stats.npts \
           or m[1].stats.npts != m[2].stats.npts:
            print("[pyColocSensors.transMatrix]: Traces does not have the same \
                  length")

        if m[0].stats.starttime-m[1].stats.starttime >= \
           m[0].stats.sampling_rate/2 \
           or m[1].stats.starttime-m[2].stats.starttime >= \
           m[1].stats.sampling_rate/2\
           or m[0].stats.starttime-m[2].stats.starttime >= \
           m[0].stats.sampling_rate/2:
            print("[pyColocSensors.transMatrix]: Traces does not have the same start\
                  time")

        # Selection des traces
        m.detrend('demean')
        m.detrend('linear')
        m.taper(0.1)
        m.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True,
                 corners=8)

    # Verify streams between them
    if m_stream[0].stats.sampling_rate != r_stream[0].stats.sampling_rate:
        print("[pyColocSensors.transMatrix]: Sampling rates are not identical \
              between traces")

    if m_stream[0].stats.npts != r_stream[0].stats.npts:
        print("[pyColocSensors.transMatrix]: Traces does not have the same length")

    if m_stream[0].stats.starttime-r_stream[0].stats.starttime >= \
       m_stream[0].stats.sampling_rate/2:
        print("[pyColocSensors.transMatrix]: Traces does not have the same start\
              time")

    coeff_matrix = np.matrix([m_stream[0].data, m_stream[1].data,
                             m_stream[2].data]).transpose()

    matrix = np.array([])
    for i in range(3):
        matrix = np.append(matrix, np.linalg.lstsq(coeff_matrix,
                                                   r_stream[i].data)[0])
    matrix = np.reshape(matrix, (3, 3))

    return matrix
