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

if __name__=='__main__':	
	client=Client(host='10.0.0.15',port=18001,user='mbesdeberc@unistra.fr')
	t1=UTCDateTime("2016-02-25T14:00:00")
	t2=t1+10*60
	st_ref=client.get_waveforms('FR','STR','00','HH?',t1,t2,route=False)
	st_ref.sort()
	for t in st_ref:
		t.data=t.data*1.0/(1677721*1201)
	st_test=client.get_waveforms('XX','GP000','00','HH?',t1,t2,route=False)
	st_test.sort()
	for t in st_test:
		t.data=t.data*1./(419430)
	t_mat=transMatrix(st_test,st_ref)
	alpha=np.arctan(t_mat[1][0]/t_mat[0][0])
	print("Orientation error: "+str(alpha*180/np.pi)+" degrees")
	print(st_test[0].id+ " gain: "+str(t_mat[0][0]/np.cos(alpha)))
	print(st_test[1].id+ " gain: "+str(t_mat[1][1]/np.cos(alpha)))
	print(st_test[2].id+ " gain: "+str(t_mat[2][2]/np.cos(alpha)))
	


	

