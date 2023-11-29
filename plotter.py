#===========================================================================
#*** Copyright Notice ***

#SALSA_python (SALSython) Copyright (c) 2023, The Regents
#of the University of California, through Lawrence Berkeley National
#Laboratory (subject to receipt of any required approvals from the U.S.
#Dept. of Energy). All rights reserved.

#If you have questions about your rights to use or distribute this software,
#please contact Berkeley Lab's Intellectual Property Office at
#IPO@lbl.gov.

#NOTICE.  This Software was developed under funding from the U.S. 
#Department of Energy and the U.S. Government consequently retains 
#certain rights.  As such, the U.S. Government has been granted for itself 
#and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#worldwide license in the Software to reproduce, distribute copies to the 
#public, prepare derivative works, and perform publicly and display publicly, 
#and to permit others to do so.

#===========================================================================
#*** License Agreement ***

#GPL v3 License
#SALSA_python (SALSython) Copyright (c) 2023, The Regents
#of the University of California, through Lawrence Berkeley National
#Laboratory (subject to receipt of any required approvals from the U.S.
#Dept. of Energy). All rights reserved.
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#===========================================================================
import numpy as np
import os
import salsa2 as salsa # SALSA library
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import glob

def purger(file_name):
	ret_val=0
	for f in glob.glob(file_name):
		try:
			os.remove(f)
		except Exception as e:
			print(f'Unable to trash file:')#' {e}')
			print(f'Delete unwanted files manually')
			ret_val=1
	return ret_val		


def contourplot(xar,yar,zar,iout,dir_name):
	#plt.rcParams["savefig.directory"]=os.chdir(os.path.dirname(dir_name))
	fig, ax1 = plt.subplots(nrows=1)
	ax1.tricontour(xar, yar, zar, \
	levels=14, linewidths=0.5, colors='k')
	cntr2 = ax1.tricontourf(xar, yar, zar,\
	levels=14, cmap="RdBu_r")
	fig.colorbar(cntr2, ax=ax1)
	plt.subplots_adjust(hspace=0.5)
	nme=dir_name+str(iout)+'.pdf'
	plt.savefig(nme,dpi=50)
	plt.close()

def time_series_plot(xar,yar,t0,dt,iobs,xobs,yobs,dir_name):#yout is
	Nt=np.shape(yar)[0]
	time=np.linspace(t0,t0+((Nt-1)*dt),Nt) 	
	#fig, ax1 = plt.subplots(nrows=1)
	lgd=[str(tm)+' day' for tm in time]
	for i in range(Nt):
		plt.plot(yar[i],xar,'-')
	plt.ylabel('z from bottom (m)')
	plt.xlabel('Head (m)')
	plt.legend(lgd,frameon=False)
	nme=dir_name+str(iobs)+'.pdf'
	plt.title('x='+str(xobs)+' m, y='+str(yobs)+' m')
	plt.savefig(nme,dpi=50)
	plt.close()