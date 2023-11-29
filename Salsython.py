#===========================================================================
# Wrapper developed by Pramod Bhuvankar
# f77 library created using SALSA.f by A Cihan
# 11/13/2023
# Energy Geosciences Division
# Lawrence Berkeley National Lab, Berkeley CA
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
import salsa2 as salsa # SALSA library
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import plotter as ppt
import os
import shutil
#------------Declaring the input dimensions-----------------------------------#
n_aq=2	          #Number of Aquifers    		 #mnaq=60 
n_at=3		  #Number of Aquitards    		 #mnat=61
n_lw=1            #Number of leaky wells                 #mnlw=100  
n_iw =1           #Number of active wells                #mniw=50
n_xyz=28000                                       #nxyz          #4
n_per=2           #Number of injection periods           #mnp=100
c_visualize_flag=1  #Create contour plots? 1: Yes    0: No
visualize_flag=1  #Create time series plots? 1: Yes    0: No
#-----------Output parameters-------------------------------------------------#
n_t  =10     #Number of output times                #mnt=1000
n_aqp=2      #Number of aquifer layers where contour plot is requested
n_b  =1      #Number of locations for calculation of buildup in aquifers
n_ab =1      #Number of observation points where vertical head profiles are computed
n_abv=10     #Number of points along vertical direction within each aquifer for head
             #profile
#-----------------------------------------------------------------------------#
n_eq=n_lw*n_aq    #Number of equations = n_lkw x n_aqf   #mneq=5000  

direc = dict(mtype=0, bl=0, tl=0 , naq=1, \
	baq=np.zeros((n_aq)), hconx=np.zeros((n_aq)),\
	ansr=0., ss=np.zeros((n_aq)),rho=np.zeros((n_aq)),hi=np.zeros((n_aq)),gamma=np.zeros((n_aq)),\
	naqt=1, baqt=np.zeros((n_at)), \
	hconp=np.zeros((n_at)), ssp=np.zeros((n_at)), gamma0=np.zeros((n_at)),gammat=np.zeros((n_at)), \
	botb=0, topb=0, hbot=0., htop=0.,niw=1, nper=n_per, \
	dt1= np.zeros((n_per)), x1=np.zeros((n_iw)), y1=np.zeros((n_iw)), nap=1, \
	kk=np.zeros((n_aq,n_iw)), riw=np.zeros((n_aq,n_iw)), q=np.zeros((n_aq,n_iw,n_per)), nlw=2,\
	x2=np.zeros((n_lw)),y2=np.zeros((n_lw)),rw=np.zeros((n_lw,n_aq)),kww=np.zeros((n_aq)),\
	rwa=np.zeros((n_lw,n_at)), kw=np.zeros((n_at)), stata=np.zeros((n_lw,n_aq)),\
	staca=np.zeros((n_lw,n_aq+1)), slw=np.zeros(n_lw), lwid=np.zeros((n_lw,n_aq)),\
	qlw=np.zeros((n_lw,n_per)), nint=1, nt=0, dt=5, dt3=np.zeros((n_t)), outa=[0,0,0],\
	flag_mesh=1, xcent=0., ycent=0., xmax=0., ymax=0., exp_plot=1.,ni=1, nj=1, ncirc=1,\
	rmax=1., expr=1., naqp= 1, aql=np.zeros(n_aqp), nb=n_b, xba=np.zeros(n_b),\
	yba=np.zeros(n_b), nab= n_ab, nabv=1, xaq=np.zeros(n_ab), yaq=np.zeros(n_ab), nxyz=n_xyz, \
	x=np.zeros((n_xyz)), y=np.zeros((n_xyz)),\
#-----Output parameters------------------------------------------------------#
    haqt=np.zeros((n_t,n_abv+1,n_at,n_ab)),saqf=np.zeros((n_t,n_ab,n_aq)),\
    faqt=np.zeros((n_t,n_aq*2)), cfaqt=np.zeros((n_t,n_aq*2)), qctvt=np.zeros((n_t,n_aq)),\
    cfct=np.zeros((n_t,n_aq*2)), fcte_ind=np.zeros((n_t,n_lw,n_aq)),\
    chead=np.zeros((n_t,n_aqp,n_xyz)),cxout=np.zeros((n_xyz)), cyout=np.zeros((n_xyz)))

#-----------Defining the input dictionary-------------------------------------#
direc['mtype']      =  3 #Fixed in test code, don't change!
direc['bl'] =  0 # element 1: Bottom layer, element 2:Top layer
direc['tl'] =  0 # element 1: Bottom layer, element 2:Top layer
# Aquifers-------------------------------------#
direc['naq']       =  n_aq
direc['baq'][0:n_aq] 	   =  [50.00, 50.00]#, 60.]
direc['hconx'][0:n_aq]     =  [0.02983924, 0.02983924]#, 0.31144]
direc['ansr']      =  1.#[1.,1.,1.]
direc['ss'][0:n_aq]		   =  [2.11258e-6,2.11258e-6]#[1.69e-6, 1.703e-6, 1.72e-6]
direc['rho'][0:n_aq]	   =  [1000.00,1000.00]#[1092.,1000.,1000.]
direc['hi'][0:n_aq] 	   =  [70.6334,36.4788]#[706.,394.,394.]
direc['gamma'][0:n_aq] 	   =  [0.,0.]#[0.,0.,0.]

# Aquitards-------------------------------------#
direc['naqt'] 	   =  direc['naq'] + 1 - direc['bl']-direc['tl']
direc['baqt'][0:n_at]	   =  [186.00,186.00,186.00]
direc['hconp'][0:n_at]     =  [8.75166e-7, 8.75166e-7, 8.75166e-7]
direc['ssp'][0:n_at]	   =  [5.38798e-6, 5.38798e-6, 5.38798e-6]
direc['gamma0'][0:n_at]	   =  [0., 0., 0.]
direc['gammat'][0:n_at]	   =  [0., 0., 0.]

# Domain BCs-------------------------------------#
direc['botb']      =  1
direc['topb']	   =  0
direc['hbot']      =  0.
direc['htop']      =  0.

# Active wells-------------------------------------#
direc['niw']	   =  n_iw
direc['nper']	   =  n_per
direc['dt1'][0:n_per]	   =  [18250.00, 36500.00*100.00]

# Well 1
direc['x1'][0]     =  0.00
direc['y1'][0]     =  0.00

direc['kk'][0][0]  =  1    #order: [aquifer][injection well]
direc['kk'][1][0]  =  3    #order: [aquifer][injection well]

direc['riw'][0][0] =  0.1    #order: [aquifer][injection well]
direc['riw'][1][0] =  0.1    #order: [aquifer][injection well]

direc['q'][0][0][0] = 835.00#3343.  #order: [aquifer][injection well][period]
direc['q'][0][0][1] = 0.0  #order: [aquifer][injection well][period]
direc['q'][1][0][0] = 0.0#3343.  #order: [aquifer][injection well][period]
direc['q'][1][0][1] = 0.0 #order: [aquifer][injection well][period]

#Leaky wells-----------------------------------------#
direc['nlw']	   =  n_lw#2

# leaky well 1
direc['x2'][0]     =  2000.0
direc['y2'][0]     =  0.0

direc['rw'][0][0]     =  0.15 # order: [leaky well][aquifer] #IMPORTANT!
direc['rw'][0][1]     =  0.15 # order: [leaky well][aquifer] #IMPORTANT!

direc['kww'][0]    = 99464.13382# order: [aquifer] (in this version
direc['kww'][1]    = 99464.13382#         assume all wells have same kww)

direc['rwa'][0][0]  = 0.15  #order: [leaky well][aquitard]
direc['rwa'][0][1]  = 0.15  #order: [leaky well][aquitard]
direc['rwa'][0][2]  = 0.15  #order: [leaky well][aquitard]

direc['kw'][0]    = 99464.13382# order: [aquitard] (in this version
direc['kw'][1]    = 99464.13382#         assume all wells have same kw)
direc['kw'][2]    = 99464.13382#

direc['stata'][0][0] = 1 #order: [leaky well][aquifer]

direc['staca'][0][0] = 1 #order: [leaky well][aquifer]
direc['staca'][0][1] = 1 #order: [leaky well][aquifer]
direc['staca'][0][2] = 0 #order: [leaky well][aquifer]

direc['slw'][0]      = 1

direc['lwid'][0][0] = 0. #order: [leaky well][aquifer]
direc['lwid'][0][1] = 0. #order: [leaky well][aquifer]
direc['qlw'][0][0] = 0. #order: [leaky well][aquifer]
direc['qlw'][0][1] = 0. #order: [leaky well][aquifer]

#--Solution control & time step-----------------------#
direc['nint']   =  8
direc['nt']     = n_t

direc['dt3'][0] = 3650.00
for i in range(1,direc['nt']):
	direc['dt3'][i] = 3650.00*(i+1.0)

#Number of points where transient head in aquifer reqd.
direc['nb']	   = n_b

direc['xba'][0] = 1990.00 # x location of observation point
direc['yba'][0] = 0.0   # y location of observation point

#Number of points where vertical head profile is reqd.
direc['nab']   = n_ab#1
direc['nabv']      = n_abv#10
direc['xaq'][0]    = direc['xba'][0] #observation pt coordinates..
direc['yaq'][0]    = direc['yba'][0] #..for vertical profile

#----------For generating output mesh-----------------#
direc['flag_mesh']  = 1 #  Same as 'flag' in main code
direc['xcent'] = 0.0
direc['ycent'] = 0.0
direc['xmax']  = 3000.0
direc['ymax']  = 3000.0
direc['exp_plot'] = 1.00
direc['ni']    = 40
direc['nj']    = 40
direc['ncirc'] = 6
direc['rmax']  = 50.00
direc['expr']  = 1.20
direc['nxyz']  = n_xyz
#-----------------------------------------------------#

direc['naqp']  = n_aqp

direc['aql'][0]= 1
direc['aql'][1]= 2

#-----------------------------------------------------#
saq = np.zeros((n_t,n_ab,n_aq))
fcte = np.zeros((n_t,n_eq))
chd = np.zeros((n_t,n_aqp,n_xyz))

def salsa_wrap(dcy,sq,fc):
	print("Calling SALSA...")
	[dcy['haqt'],dcy['saqf'],dcy['faqt'],dcy['cfaqt'],dcy['qctvt'],\
	dcy['cfct'],dcy['fcte_ind'],dcy['chead'],dcy['cxout'],dcy['cyout']]=\
	salsa.sals2(\
#------------All variables-----------------------------#		
		dcy['mtype'],dcy['bl'],dcy['tl'],dcy['baq'],\
		dcy['hconx'],dcy['ansr'],dcy['ss'],dcy['rho'],dcy['hi'],\
		dcy['gamma'],dcy['baqt'],dcy['hconp'],dcy['ssp'],dcy['gamma0'],\
		dcy['gammat'],dcy['botb'],dcy['topb'],dcy['hbot'],dcy['htop'],\
		dcy['dt1'],dcy['x1'],dcy['y1'],\
		dcy['riw'],dcy['q'],dcy['x2'],dcy['y2'],dcy['rw'],\
		dcy['kww'],dcy['rwa'],dcy['kw'],dcy['stata'],dcy['staca'],\
		dcy['slw'],dcy['qlw'],dcy['nint'],dcy['dt3'],\
		dcy['xba'],dcy['yba'],dcy['xaq'],\
		dcy['yaq'],dcy['flag_mesh'],dcy['xcent'],dcy['ycent'],\
		dcy['xmax'],dcy['ymax'],dcy['exp_plot'],dcy['ni'],\
		dcy['nj'],dcy['ncirc'],dcy['rmax'],dcy['expr'],dcy['aql'],\
#-------------------output parameters-------------------#		
		dcy['haqt'],dcy['saqf'],dcy['faqt'],dcy['cfaqt'],dcy['qctvt'],\
		dcy['cfct'],\
		dcy['fcte_ind'],dcy['chead'],dcy['cxout'],dcy['cyout'],sq,fc,\
# haqt: aquitard head buildup;
#---------------Array sizes-(In proper order)-------------#
        dcy['naq'],dcy['naqt'],dcy['nper'],dcy['nt'],dcy['niw'],\
        dcy['nlw'],dcy['nb'],dcy['nab'],dcy['nabv'],dcy['naqp'],\
        dcy['nxyz'])
#---------------------------------------------------------------#        
	
salsa_wrap(direc,saq,fcte)
print("SALSA computations done")
#--------------Trashing existing folders------------------------#
for filename in os.listdir(os.getcwd()):
	direct=os.getcwd()+'/'+filename
	if os.path.isdir(os.path.join(os.getcwd(),filename)):
		try:			
			shutil.rmtree(direct)
		except Exception as e:
			print(f'Failed to delete directory: {e}')
			print(f'Please delete folders of format Aq*, Head*, Leakage before running')
#---------------------------------------------------------------#
#========Postprocessing arrays==================================#
#=======Contour plots-------------------------------------------#
if(c_visualize_flag==1): print("Creating contour plots...")
for j in range(direc['naq']):
	if(c_visualize_flag==1): direct=os.getcwd()+'/Aq'+str(j)+'/'
	if(c_visualize_flag==1): os.mkdir(direct)

	for i in range(direc['nt']):
		if(c_visualize_flag==1): head=direc['chead'][i][j]
		if(c_visualize_flag==1): ppt.contourplot(direc['cxout'], direc['cyout'],\
			head,i,direct)
#=======Aquitard heads------------------------------------------#
#-------Z-array for plots---------------------------
sub_ind=1
if(direc['bl']==1):
	sub_ind=0
zar=np.linspace(0,direc['baqt'][0],direc['nabv']+1)
if(direc['bl']==1):
	zar=np.concatenate(([0],zar+direc['baq'][0]))
endz=zar[np.size(zar)-1]
for j in range(direc['naqt']):
	if(j>0):
		begz=endz+direc['baq'][j-sub_ind]
		endz=begz+direc['baqt'][j]
		zar = np.concatenate((zar,np.linspace(begz,endz,direc['nabv']+1)))
if(direc['tl']==1):
	zar=np.concatenate((zar,[zar[np.size(zar)-1]+(direc['baq'][direc['naq']-1])]))
#=======Aquitard heads------------------------------------------#
if(visualize_flag==1):
	print("Creating time series plots...")
	direct=os.getcwd()+'/Head/'
	os.mkdir(direct)

head_aqt=np.zeros((direc['nab'],direc['nt'],np.size(zar)))
for j in range(direc['nab']):
	har=np.zeros((direc['nt'],np.size(zar)))
	
	for i in range(direc['nt']):
		ct=0
		if(direc['bl']==1):
			har[i][0]=direc['haqt'][i][0][0][j]
			ct=1		
		for k in range(direc['naqt']):
			for l in range(direc['nabv']+1):				
				har[i][ct]=direc['haqt'][i][l][k][j]
				ct=ct+1
		if(direc['tl']==1):
			har[i][ct]=har[i][ct-1]		
	
	head_aqt[j]=har	
	if(visualize_flag==1): 
		ppt.time_series_plot(zar,head_aqt[j],direc['dt3'][0],\
			direc['dt3'][1]-direc['dt3'][0],\
			j,direc['xaq'][j],direc['yaq'][j],direct+'Vertical_Head_Profile')
	
#=======Aquifer heads------------------------------------------#
if(visualize_flag==1): lg=['A'+str(k) for k in range(direc['naq'])]

head_aqf=np.zeros((direc['nb'],direc['naq'],direc['nt']))

for j in range(direc['nb']):
	for i in range(direc['naq']):
		har=np.zeros((direc['nt']))
		for k in range(direc['nt']):
			har[k]=direc['saqf'][k][j][i]
		head_aqf[j][i]=har
		if(visualize_flag==1): plt.plot(direc['dt3'],head_aqf[j][i])#har)	

	if(visualize_flag==1):
		plt.legend(lg,frameon=False)
		plt.title('x='+str(direc['xba'][j])\
			+' m, y='+str(direc['yba'][j])+' m')
		plt.xlabel('Time (day)')
		plt.ylabel('Head (m)')
		plt.savefig(direct+'Aquifers'+str(j)+'.pdf',dpi=50)
		plt.close()

#======Leakage into aquifers & aquitards-------------------------#
#fcte_ind=np.zeros((n_t,n_lw,n_aq))
#cfct=np.zeros((n_t,n_aq*2)) qctvt=np.zeros((n_t,n_aq))
if(visualize_flag==1):
	direct=os.getcwd()+'/Leakage/'
	os.mkdir(direct)
	lg=['Aqf# '+str(k) for k in range(direc['naq'])]
	locleak=np.zeros((direc['nt']))
	for j in range(direc['nlw']):
		for i in range(direc['naq']):
			for k in range(direc['nt']):
				locleak[k]=direc['fcte_ind'][k][j][i]
			plt.plot(direc['dt3'],locleak)
		plt.legend(lg,frameon=False)
		plt.title('Leaky well# '+str(j))
		plt.xlabel('Time (day)')
		plt.ylabel('Leakage rate (m^3/day)')
		plt.savefig(direct+'LeakyWell'+str(j)+'.pdf',dpi=50)
		plt.close()
	locleak=np.zeros((direc['nt']))
	for j in range(direc['naq']):
		for k in range(direc['nt']):
			locleak[k]=direc['qctvt'][k][j]
		plt.plot(direc['dt3'],locleak)
	plt.legend(lg,frameon=False)
	plt.title('Aquifer leakage rate')
	plt.xlabel('Time (day)')
	plt.ylabel('Leakage rate (m^3/day)')
	plt.savefig(direct+'Aquifer_LeakRate.pdf',dpi=50)
	plt.close()

	for j in range(direc['naq']):
		for k in range(direc['nt']):
			locleak[k]=direc['cfct'][k][j]
		plt.plot(direc['dt3'],locleak)
	plt.legend(lg,frameon=False)
	plt.title('Aquifer leakage rate')
	plt.xlabel('Time (day)')
	plt.ylabel('Leakage (m^3)')
	plt.savefig(direct+'Aquifer_Cumulative.pdf',dpi=50)
	plt.close()

#-------Trashing files if visualize--------------------------------
if(visualize_flag==1):
	ret_val=ppt.purger("CONTOUR*")
	if(ret_val==0):
		ret_val=ppt.purger("FLOW_*")
		ret_val=ppt.purger("MESH*")
		ret_val=ppt.purger("Active*")
		ret_val=ppt.purger("list*")
		ret_val=ppt.purger("Head_*")		
#===================================================================
# Output arrays:
# Variable            Dimensions
# 
		
		
		


