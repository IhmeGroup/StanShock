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
import os
from typing import Optional
import time

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cantera as ct
import imageio

from StanShock.stanShock import stanShock


#=============================================================================
def getPressureData(fileName):
    '''
    function getPressureData
    ==========================================================================
    This function returns the formatted pressure vs time data
        Inputs:
            fileName = name of csv data
        Outputs: 
             t = time [s]
             p = pressure [Pa]
    '''   
    #parameters
    tLower, tUpper = 0.0, 0.05 #in [s]
    pLower, pUpper = 0.0, 1.032e5*5.0 #in [Pa]
    
    #read image file
    imageData = imageio.imread(fileName)
    imageData=imageData[-1::-1,:]
    imageData[imageData<128]=1.0
    imageData[imageData>=128]=0.0
    imageData=imageData[:,np.sum(imageData,axis=0)!=0]
    nP, nT = imageData.shape
    
    #extract pressure and time data
    p = np.linspace(pLower,pUpper,nP).reshape((nP,1))
    t = np.linspace(tLower,tUpper,nT).reshape((nT,1))
    p = imageData*p
    p = np.sum(p,axis=0)/np.sum(imageData,axis=0)
    return (t,p)


def main(data_filename: str = "data/validation/case4.png",
         mech_filename: str = "data/mechanisms/N2O2HeAr.xml",
         show_results: bool = True,
         results_location: Optional[str] = None) -> None:
    #=============================================================================
    #provided condtions for Case4
    T1 = T4 = 292.05
    p1 = 390.0*133.322
    p4 = 82.0*6894.76*0.9
    tFinal=60e-3

    #plotting parameters
    fontsize=12

    #provided geometry
    DDriven = 4.5*0.0254
    DDriver = DDriven
    LDriver = 142.0*0.0254
    LDriven = 9.73

    #Set up gasses and determine the initial pressures
    u1 = 0.0;
    u4 = 0.0; #initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1; #assumed
    gas1.TPX = T1,p1,"O2:0.21,AR:0.79"
    gas4.TPX = T4,p4,"HE:0.25,N2:0.75"

    #set up geometry
    nX = 1000 #mesh resolution
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    geometry=(nX,xLower,xUpper,xShock)
    #arrays from HTGL
    xInterp = -0.0254*np.array([142,140,130,120,110,100,90,80,70,60,50,40,36,37,30,20,10,0])
    dInterp = 0.0254*np.array([3.25,3.21,3.01,2.81,2.61,2.41,2.21,2.01,1.81,1.61,1.41,1.21,1.13,0.00,0.00,0.00,0.00,0.00])
    dDInterpdxInterp = (dInterp[1:]-dInterp[:-1])/(xInterp[1:]-xInterp[:-1])
    def DOuter(x):
        nX = x.shape[0]
        return DDriven*np.ones(nX)
    def DInner(x):
        diameter = np.interp(x,xInterp,dInterp)
        return diameter
    def dDOuterdx(x): return np.zeros(nX)
    def dDInnerdx(x):
        dDiameterdx = np.interp(x,xInterp[:-1],dDInterpdxInterp)
        return dDiameterdx
    A = lambda x: np.pi/4.0*(DOuter(x)**2.0-DInner(x)**2.0)
    dAdx = lambda x: np.pi/2.0*(DOuter(x)*dDOuterdx(x)-DInner(x)*dDInnerdx(x))
    dlnAdx = lambda x,t: dAdx(x)/A(x)

    #solve with boundary layer model
    boundaryConditions=['reflecting','reflecting']
    state1 = (gas1,u1)
    state4 = (gas4,u4)
    ssbl = stanShock(gas1,initializeRiemannProblem=(state4,state1,geometry),
                     boundaryConditions=boundaryConditions,
                     cfl=.9,
                     outputEvery=100,
                     includeBoundaryLayerTerms=True,
                     Tw=T1, #assume wall temperature is in thermal eq. with gas
                     DInner= DInner,
                     DOuter= DOuter,
                     dlnAdx=dlnAdx)
    ssbl.addProbe(max(ssbl.x)) #end wall probe
    ssbl.addXTDiagram("p")
    ssbl.addXTDiagram("T")

    #adjust for partial filling strategy
    XN2Lower = 0.80 #assume smearing during fill
    XN2Upper = 1.5-XN2Lower
    dx = ssbl.x[1]-ssbl.x[0]
    dV = A(ssbl.x)*dx
    VDriver = np.sum(dV[ssbl.x<xShock])
    V = np.cumsum(dV)
    V -= V[0]/2.0 #center
    VNorms = V/VDriver
    #get gas properties
    iHE, iN2 = gas4.species_index("HE"), gas4.species_index("N2")
    for iX, VNorm in enumerate(VNorms):
        if VNorm<=1.0:
            #nitrogen and helium
            X = np.zeros(gas4.n_species)
            XN2 = XN2Lower+(XN2Upper-XN2Lower)*VNorm
            XHE = 1.0-XN2
            X[[iHE,iN2]] = XHE, XN2
            gas4.TPX = T4,p4, X
            ssbl.r[iX] = gas4.density
            ssbl.Y[iX,:] = gas4.Y
            ssbl.gamma[iX] = gas4.cp/gas4.cv

    #Solve
    t0 = time.perf_counter()
    ssbl.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)

    #Solve without boundayr layer model
    boundaryConditions=['reflecting','reflecting']
    gas1.TP = T1,p1
    gas4.TP = T4,p4
    ssnbl = stanShock(gas1,initializeRiemannProblem=(state4,state1,geometry),
                      boundaryConditions=boundaryConditions,
                      cfl=.9,
                      outputEvery=100,
                      includeBoundaryLayerTerms=False,
                      Tw=T1, #assume wall temperature is in thermal eq. with gas
                      DInner= DInner,
                      DOuter= DOuter,
                      dlnAdx=dlnAdx)
    ssnbl.addProbe(max(ssnbl.x)) #end wall probe
    ssnbl.addXTDiagram("p")
    ssnbl.addXTDiagram("T")

    #adjust for partial filling strategy
    XN2Lower = 0.80 #assume smearing during fill
    XN2Upper = 1.5-XN2Lower
    dx = ssnbl.x[1]-ssnbl.x[0]
    dV = A(ssnbl.x)*dx
    VDriver = np.sum(dV[ssnbl.x<xShock])
    V = np.cumsum(dV)
    V -= V[0]/2.0 #center
    VNorms = V/VDriver
    #get gas properties
    iHE, iN2 = gas4.species_index("HE"), gas4.species_index("N2")
    for iX, VNorm in enumerate(VNorms):
        if VNorm<=1.0:
            #nitrogen and helium
            X = np.zeros(gas4.n_species)
            XN2 = XN2Lower+(XN2Upper-XN2Lower)*VNorm
            XHE = 1.0-XN2
            X[[iHE,iN2]] = XHE, XN2
            gas4.TPX = T4,p4, X
            ssnbl.r[iX] = gas4.density
            ssnbl.Y[iX,:] = gas4.Y
            ssnbl.gamma[iX] = gas4.cp/gas4.cv

    #Solve
    t0 = time.perf_counter()
    ssnbl.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)

    #import shock tube data
    tExp, pExp = getPressureData(data_filename)
    timeDifference = (18.6-6.40)/1000.0 #difference between the test data and simulation times
    tExp+=timeDifference

    #make plots of probe and XT diagrams
    plt.close("all")
    mpl.rcParams['font.size']=fontsize
    plt.rc('text',usetex=True)
    plt.figure(figsize=(4,4))
    plt.plot(np.array(ssnbl.probes[0].t)*1000.0,np.array(ssnbl.probes[0].p)/1.0e5,'k',label="$\mathrm{Without\ BL\ Model}$",linewidth=2.0)
    plt.plot(np.array(ssbl.probes[0].t)*1000.0,np.array(ssbl.probes[0].p)/1.0e5,'r',label="$\mathrm{With\ BL\ Model}$",linewidth=2.0)
    plt.plot(tExp*1000.0,pExp/1.0e5,label="$\mathrm{Experiment}$",alpha=0.7)
    plt.axis([0,60,0,5])
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$p\ [\mathrm{bar}]$")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if show_results:
        plt.show()

    if results_location is not None:
        np.savez(
            os.path.join(results_location, "case4.npz"),
            pressure_with_boundary_layer=ssbl.probes[0].p,
            pressure_without_boundary_layer=ssnbl.probes[0].p,
            time_with_boundary_layer=ssbl.probes[0].t,
            time_without_boundary_layer=ssnbl.probes[0].t
        )
        plt.savefig(os.path.join(results_location, "case4.png"))


if __name__ == "__main__":
    main()
