#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Function calculating a rotation matrix between a stream of 3 orthogonal \
    traces (ouput of unknown seismometer) and another stream of 3 orthogonal \
    reference traces (output of known seismometer) recording the same signal. \
    For each trace of unknown stream, it uses the linear method to express it \
    as a combination of the reference stream. Therefore, the result is 3 \
    coeff per unknown channel, ie a 3x3 matrix:
        |X|   |a00 a10 a20|   |Xref|
        |Y| = |a01 a11 a21| x |Yref|
        |Z|   |a02 a12 a22|   |Zref|
    From this matrix, one can calculate the orientation error between \
    seismometers:
        alpha=arctan(a01/a00).
    the calculate gains:
        Gx=a00/cos(alpha)
        Gy=a11/cos(alpha)
        Gz=a22
    More, apply this method between a stream and itself gives a direct result \
    of diaphony/orthogonility errors. In fact, the matrix becomes ideally:
        |X|   |1 0 0|   |X|
        |Y| = |0 1 0| . |Y|
        |Z|   |0 0 1|   |Z|
    Values close to 0 give addition of the orthogonality/diaphony errors of \
    the two streams.
    Finally, that supposes strong coherence between signals. It is therefore \
    necessary to filter the signals over an appropriate range (ie. \
    micro-seismic peak).

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
    """
    Rotation matrix calculation function.
    :type res_stream: Stream object from module obspy.core.stream
    :param res_stream: Stream containing 3 traces with the signal from the \
    unknown seismometer.
    :type mon_stream: Stream object from module obspy.core.stream
    :param mon_stream: Stream containing 3 traces with the signal from the \
    known seismometer.
    :type fmin: float
    :param fmin: Minimal frequency of the bandpass filter applied.
    :type fmax: float
    :param fmax: Maximal frequency of the bandpass filter applied.
    """
    m_stream = mon_stream.copy()
    r_stream = res_stream.copy()

    for m in (m_stream, r_stream):

        if len(m) != 3:
            print("[pyColocSensors.transMatrix]: Stream must strictly have 3 \
traces")
            raise SystemExit

        # Verify each trace of stream: same sampling rate?
        if m[0].stats.sampling_rate != m[1].stats.sampling_rate \
           or m[0].stats.sampling_rate != m[2].stats.sampling_rate \
           or m[1].stats.sampling_rate != m[2].stats.sampling_rate:
            print("[pyColocSensors.transMatrix]: Sampling rates are not identical \
                  between traces")
            raise SystemExit

        # Verify each trace of stream: same length?
        if m[0].stats.npts != m[1].stats.npts \
           or m[0].stats.npts != m[2].stats.npts \
           or m[1].stats.npts != m[2].stats.npts:
            print("[pyColocSensors.transMatrix]: Traces does not have the same \
length")
            raise SystemExit

        # Verify each trace of stream: same start time?
        if m[0].stats.starttime-m[1].stats.starttime >= \
           m[0].stats.sampling_rate/2 \
           or m[1].stats.starttime-m[2].stats.starttime >= \
           m[1].stats.sampling_rate/2\
           or m[0].stats.starttime-m[2].stats.starttime >= \
           m[0].stats.sampling_rate/2:
            print("[pyColocSensors.transMatrix]: Traces does not have the same start\
                  time")
            raise SystemExit

        # Detrend, taper and filter stream
        m.detrend('demean')
        m.detrend('linear')
        m.taper(0.2)
        m.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True,
                 corners=8)

    # Verify streams between them: same sampling rate?
    if m_stream[0].stats.sampling_rate != r_stream[0].stats.sampling_rate:
        print("[pyColocSensors.transMatrix]: Sampling rates are not identical \
              between traces")
        raise SystemExit

    # Verify streams between them: same length?
    if m_stream[0].stats.npts != r_stream[0].stats.npts:
        print("[pyColocSensors.transMatrix]: Traces does not have the same \
              length")
        raise SystemExit

    # Verify streams between them: same start time?
    if m_stream[0].stats.starttime-r_stream[0].stats.starttime >= \
       m_stream[0].stats.sampling_rate/2:
        print("[pyColocSensors.transMatrix]: Traces does not have the same \
              start time")
        raise SystemExit

    # Create matrix of data with shape 3 x npts
    coeff_matrix = np.matrix([m_stream[0].data, m_stream[1].data,
                             m_stream[2].data]).transpose()

    # Create empty matrix
    matrix = np.array([])
    # Feed it with 9 coefficients calculated with linear regression
    for i in range(3):
        matrix = np.append(matrix, np.linalg.lstsq(coeff_matrix,
                                                   r_stream[i].data)[0])
    # Reshape correctly the final matrix
    matrix = np.reshape(matrix, (3, 3))

    return matrix
