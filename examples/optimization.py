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
from scipy.optimize import newton

from StanShock.stanShock import stanShock, smoothingFunction, dSFdx


def main(mech_filename: str = "data/mechanisms/HeliumArgon.xml",
         show_results: bool = True,
         results_location: Optional[str] = None) -> None:
    #parameters
    fontsize = 12
    tFinal = 7.5e-3
    p5, p1 = 18*ct.one_atm, 0.48e5
    T5 = 1698.0
    g4 = g1 = 5.0/3.0 #monatomic gas in driver and driven sections
    W4, W1 = 4.002602, 39.948 #Helium and argon
    MachReduction = 0.985 #account for shock wave attenuation
    nXCoarse, nXFine = 200, 1000 #mesh resolution
    LDriver, LDriven = 3.0, 5.0
    DDriver, DDriven = 7.5e-2, 5.0e-2
    plt.close("all")
    mpl.rcParams['font.size']=fontsize
    plt.rc('text',usetex=True)

    #set up geometry
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    Delta = 10*(xUpper-xLower)/float(nXFine)
    geometry=(nXCoarse,xLower,xUpper,xShock)
    DInner = lambda x: np.zeros_like(x)
    dDInnerdx = lambda x: np.zeros_like(x)
    def DOuter(x): return smoothingFunction(x,xShock,Delta,DDriver,DDriven)
    def dDOuterdx(x): return dSFdx(x,xShock,Delta,DDriver,DDriven)
    A = lambda x: np.pi/4.0*(DOuter(x)**2.0-DInner(x)**2.0)
    dAdx = lambda x: np.pi/2.0*(DOuter(x)*dDOuterdx(x)-DInner(x)*dDInnerdx(x))
    dlnAdx = lambda x,t: dAdx(x)/A(x)

    #compute the gas dynamics
    def res(Ms1):
        return p5/p1-((2.0*g1*Ms1**2.0-(g1-1.0))/(g1+1.0)) \
               *((-2.0*(g1-1.0)+Ms1**2.0*(3.0*g1-1.0))/(2.0+Ms1**2.0*(g1-1.0)))
    Ms1 = newton(res,2.0)
    Ms1*= MachReduction
    T5oT1 = (2.0*(g1-1.0)*Ms1**2.0+3.0-g1) \
            *((3.0*g1-1.0)*Ms1**2.0-2.0*(g1-1.0)) \
            /((g1+1.0)**2.0*Ms1**2.0)
    T1 = T5/T5oT1
    a1oa4 = np.sqrt(W4/W1)
    p4op1 = (1.0+2.0*g1/(g1+1.0)*(Ms1**2.0-1.0)) \
            *(1.0-(g4-1.0)/(g4+1.0)*a1oa4*(Ms1-1.0/Ms1))**(-2.0*g4/(g4-1.0))
    p4 = p1*p4op1

    #set up the gasses
    u1 = 0.0;
    u4 = 0.0; #initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1; #assumed
    gas1.TPX = T1,p1,"AR:1"
    gas4.TPX = T4,p4,"HE:1"

    #set up solver parameters
    boundaryConditions=['reflecting','reflecting']
    state1 = (gas1,u1)
    state4 = (gas4,u4)
    ss = stanShock(gas1,initializeRiemannProblem=(state4,state1,geometry),
                   boundaryConditions=boundaryConditions,
                   cfl=.9,
                   outputEvery=100,
                   includeBoundaryLayerTerms=True,
                   Tw=T1, #assume wall temperature is in thermal eq. with gas
                   DOuter= DOuter,
                   dlnAdx=dlnAdx)

    #Solve
    t0 = time.perf_counter()
    tTest = 2e-3
    tradeoffParam=1.0
    eps = 0.01**2.0+tradeoffParam*0.01**2.0
    ss.optimizeDriverInsert(tFinal,p5=p5,tTest=tTest,tradeoffParam=tradeoffParam,eps=eps)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)

    #recalculate at higher resolution with the insert
    geometry=(nXFine,xLower,xUpper,xShock)
    gas1.TPX = T1,p1,"AR:1"
    gas4.TPX = T4,p4,"HE:1"
    ss = stanShock(gas1,initializeRiemannProblem=(state4,state1,geometry),
                   boundaryConditions=boundaryConditions,
                   cfl=.9,
                   outputEvery=100,
                   includeBoundaryLayerTerms=True,
                   Tw=T1, #assume wall temperature is in thermal eq. with gas
                   DOuter= DOuter,
                   DInner= ss.DInner,
                   dlnAdx=ss.dlnAdx)
    ss.addXTDiagram("p")
    ss.addXTDiagram("T")
    ss.addProbe(max(ss.x)) #end wall probe
    t0 = time.perf_counter()
    ss.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)
    pInsert = np.array(ss.probes[0].p)
    tInsert = np.array(ss.probes[0].t)
    ss.plotXTDiagram(ss.XTDiagrams["t"],limits=[200.0,1800.0])
    ss.plotXTDiagram(ss.XTDiagrams["p"],limits=[0.5,25])
    xInsert = ss.x
    DOuterInsert = ss.DOuter(ss.x)
    DInnerInsert = ss.DInner(ss.x)

    #recalculate at higher resolution without the insert
    gas1.TPX = T1,p1,"AR:1"
    gas4.TPX = T4,p4,"HE:1"
    ss = stanShock(gas1,initializeRiemannProblem=(state4,state1,geometry),
                   boundaryConditions=boundaryConditions,
                   cfl=.9,
                   outputEvery=100,
                   includeBoundaryLayerTerms=True,
                   Tw=T1, #assume wall temperature is in thermal eq. with gas
                   DOuter= DOuter,
                   dlnAdx= dlnAdx)
    ss.addXTDiagram("p")
    ss.addXTDiagram("T")
    ss.addProbe(max(ss.x)) #end wall probe
    t0 = time.perf_counter()
    ss.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)
    pNoInsert = np.array(ss.probes[0].p)
    tNoInsert = np.array(ss.probes[0].t)
    ss.plotXTDiagram(ss.XTDiagrams["t"],limits=[200.0,1800.0])
    ss.plotXTDiagram(ss.XTDiagrams["p"],limits=[0.5,25])

    #plot
    plt.figure()
    plt.plot(tNoInsert/1e-3,pNoInsert/1e5,'k',label="$\mathrm{No\ Insert}$")
    plt.plot(tInsert/1e-3,pInsert/1e5,'r',label="$\mathrm{Optimized\ Insert}$")
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$p\ [\mathrm{bar}]$")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.figure()
    plt.plot(xInsert,DOuterInsert,'k',label="$D_\mathrm{o}$")
    plt.plot(xInsert,DInnerInsert,'r',label="$D_\mathrm{i}$")
    plt.xlabel("$x\ [\mathrm{m}]$")
    plt.ylabel("$D\ [\mathrm{m}]$")
    plt.legend(loc="best")
    plt.tight_layout()
    if show_results:
        plt.show()

    if results_location is not None:
        np.savez(
            os.path.join(results_location, "optimization.npz"),
            pressure_with_insert=pInsert,
            pressure_without_insert=pNoInsert,
            insert_diameter=DInnerInsert,
            shock_tube_diameter=DOuterInsert,
            position=xInsert,
            time_with_insert=tInsert,
            time_without_insert=tNoInsert
        )
        plt.savefig(os.path.join(results_location, "optimization.png"))


if __name__ == "__main__":
    main()
