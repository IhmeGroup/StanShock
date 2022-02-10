#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
    Copyright 2017 Kevin Grogan
    
    This file is part of StanShock.
    
    StanShock is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License.
    
    StanShock is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    
    You should have received a copy of the GNU Lesser General Public License
    along with StanShock.  If not, see <https://www.gnu.org/licenses/>.
'''
import sys; sys.path.append('../')
from stanshock import StanShock
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from ignDelay import  flameSpeed
import time
import cantera as ct

#user parameters
TU=300.0
p = 1e5
mech = "Hong.xml"
dt=1e-3
estFlameThickness=1e-2
ntFlowThrough=.1
eps=0.05
fontsize=12
f = 0.1 #factor to reduce Cantera domain


#find the initial state of the fluids
gas = ct.Solution(mech)
unburnedState = TU,p,"H2:2,O2:1,N2:3.76"
gas.TPX = unburnedState

#get the flame thickness
flameSpeed, flame = flameSpeed(gas,estFlameThickness,returnFlame=True)
TU, TB = flame.T[0] ,flame.T[-1]
flameThickness=(TB-TU)/max(np.gradient(flame.T,flame.grid))

#get flame parameters
gasUnburned = ct.Solution(mech) 
gasUnburned.TPY = flame.T[0], flame.P, flame.Y[:,0]
uUnburned = flame.u[0]
unburnedState = gasUnburned, uUnburned
gasBurned = ct.Solution(mech) 
gasBurned.TPY = flame.T[-1], flame.P, flame.Y[:,-1]
uBurned = flame.u[-1]
burnedState = gasBurned, uBurned


#set up grid
nX = flame.grid.shape[0]
xCenter = flame.grid[np.argmax(np.gradient(flame.T,flame.grid))]
L = flame.grid[-1] - flame.grid[0]
xUpper, xLower = xCenter +L*f, xCenter-L*f

geometry=(nX,xLower,xUpper,(xUpper+xLower)/2.0)
boundaryConditions = (gasUnburned.density,uUnburned,None,gasUnburned.Y),(None,None,gasBurned.P,None)
ss = StanShock(gas, initializeRiemannProblem=(unburnedState, burnedState, geometry),
               boundaryConditions=boundaryConditions,
               cfl=.9,
               reacting=True,
               includeDiffusion=True,
               outputEvery=10)

#interpolate flame solution
ss.r = np.interp(ss.x,flame.grid,flame.density)
ss.u = np.interp(ss.x,flame.grid,flame.u)
ss.p[:] = flame.P
for iSp in range(gas.n_species): 
    ss.Y[:,iSp] = np.interp(ss.x,flame.grid,flame.Y[iSp,:])
T = ss.thermoTable.getTemperature(ss.r,ss.p,ss.Y)
ss.gamma = ss.thermoTable.get_gamma(T, ss.Y)
#calculate the final time
tFinal = ntFlowThrough*(xUpper-xLower)/(uUnburned+uBurned)*2.0

#Solve
t0 = time.clock()
ss.advance_simulation(tFinal)
t1 = time.clock()
print("The process took ", t1-t0)

#plot setup
plt.close("all")
font = {'family':'serif', 'serif': ['computer modern roman']}
plt.rc('font',**font)
mpl.rcParams['font.size']=fontsize
plt.rc('text',usetex=True)
#plot
plt.plot((flame.grid-xCenter)/flameThickness,flame.T/flame.T[-1],'r',label="$T/T_\mathrm{F}$")
T = ss.thermoTable.getTemperature(ss.r,ss.p,ss.Y)
plt.plot((ss.x-xCenter)/flameThickness,T/flame.T[-1],'r--s')
iOH = gas.species_index("OH")
plt.plot((flame.grid-xCenter)/flameThickness,flame.Y[iOH,:]*10,'k',label="$Y_\mathrm{OH}\\times 10$")
plt.plot((ss.x-xCenter)/flameThickness,ss.Y[:,iOH]*10,'k--s')
iO2 = gas.species_index("O2")
plt.plot((flame.grid-xCenter)/flameThickness,flame.Y[iO2,:],'g',label="$Y_\mathrm{O_2}$")
plt.plot((ss.x-xCenter)/flameThickness,ss.Y[:,iO2],'g--s')
iH2 = gas.species_index("H2")
plt.plot((flame.grid-xCenter)/flameThickness,flame.Y[iH2,:],'b',label="$Y_\mathrm{H_2}$")
plt.plot((ss.x-xCenter)/flameThickness,ss.Y[:,iH2],'b--s')
plt.xlabel("$x/\delta_\mathrm{F}$")
plt.legend(loc="best")
