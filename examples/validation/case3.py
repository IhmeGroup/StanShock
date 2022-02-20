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

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import cantera as ct

from StanShock.stanShock import stanShock, smoothingFunction, dSFdx
from StanShock.utils import getPressureData


def main(data_filename: str = "data/validation/case3.csv",
         mech_filename: str = "data/mechanisms/Nitrogen.xml",
         show_results: bool = True,
         results_location: Optional[str] = None) -> None:
    #=============================================================================
    #provided condtions for case 3
    Ms = 2.409616
    T1 = 292.25
    p1 = 1999.83552
    p2 = 13267.880629
    tFinal=60e-3

    #plotting parameters
    fontsize=12

    #provided geometry
    DDriven = 4.5*0.0254
    DDriver = 4.5*0.0254
    LDriver = 142.0*0.0254
    LDriven = 9.73
    DOuterInsertBack = 3.375*0.0254
    DOuterInsertFront = 1.25*0.0254
    LOuterInsert = 102.0*0.0254
    DInnerInsert = 0.625*0.0254
    LInnerInsert = 117.0*0.0254

    #Set up gasses and determine the initial pressures
    u1 = 0.0;
    u4 = 0.0; #initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1; #assumed
    gas1.TP = T1,p1
    gas4.TP = T4,p1 #use p1 as a place holder
    g1 = gas1.cp/gas1.cv
    g4 = gas4.cp/gas4.cv
    a4oa1 = np.sqrt(g4/g1*T4/T1*gas1.mean_molecular_weight/gas4.mean_molecular_weight)
    p4=p2*(1.0-(g4-1.0)/(g1+1.0)/a4oa1*(Ms-1.0/Ms))**(-2.0*g4/(g4-1.0)) #from handbook of shock waves
    p4*=1.04
    gas4.TP = T4,p4

    #set up geometry
    nX = 1000 #mesh resolution
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    geometry=(nX,xLower,xUpper,xShock)
    DeltaD = DDriven-DDriver
    dDOuterInsertdx=(DOuterInsertFront-DOuterInsertBack)/LOuterInsert
    DeltaSmoothingFunction = (xUpper-xLower)/float(nX)*10.0
    def DOuter(x): return DDriven*np.ones(nX)
    def DInner(x):
        diameter = np.zeros(nX)
        diameter+= smoothingFunction(x,xLower+LInnerInsert,DeltaSmoothingFunction,DInnerInsert,0.0)
        diameter+= smoothingFunction(x,xLower+LOuterInsert,DeltaSmoothingFunction,DOuterInsertFront-DInnerInsert,0.0)
        diameter+= smoothingFunction(x,xLower+LOuterInsert/2.0,LOuterInsert,DOuterInsertBack-DOuterInsertFront,0.0)
        return diameter
    def dDOuterdx(x): return np.zeros(nX)
    def dDInnerdx(x):
        dDiameterdx = np.zeros(nX)
        dDiameterdx+= dSFdx(x,xLower+LInnerInsert,DeltaSmoothingFunction,DInnerInsert,0.0)
        dDiameterdx+= dSFdx(x,xLower+LOuterInsert,DeltaSmoothingFunction,DOuterInsertFront-DInnerInsert,0.0)
        dDiameterdx+= dSFdx(x,xLower+LOuterInsert/2.0,LOuterInsert,DOuterInsertBack-DOuterInsertFront,0.0)
        return dDiameterdx
    A = lambda x: np.pi/4.0*(DOuter(x)**2.0-DInner(x)**2.0)
    dAdx = lambda x: np.pi/2.0*(DOuter(x)*dDOuterdx(x)-DInner(x)*dDInnerdx(x))
    dlnAdx = lambda x,t: dAdx(x)/A(x)

    #set up solver parameters
    print("Solving with boundary layer terms")
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

    #Solve
    t0 = time.perf_counter()
    ssbl.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)

    #without  boundary layer model
    print("Solving without boundary layer model")
    boundaryConditions=['reflecting','reflecting']
    gas1.TP = T1,p1
    gas4.TP = T4,p4
    ssnbl = stanShock(gas1,initializeRiemannProblem=(state4,state1,geometry),
                      boundaryConditions=boundaryConditions,
                      cfl=.9,
                      outputEvery=100,
                      includeBoundaryLayerTerms=False,
                      DInner= DInner,
                      DOuter= DOuter,
                      dlnAdx=dlnAdx)
    ssnbl.addProbe(max(ssnbl.x)) #end wall probe

    #Solve
    t0 = time.perf_counter()
    ssnbl.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1-t0)

    #import shock tube data
    tExp, pExp = getPressureData(data_filename)
    timeDifference = (12.211-8.10)/1000.0 #difference between the test data and simulation times
    tExp+=timeDifference

    #make plots of probe and XT diagrams
    plt.close("all")
    mpl.rcParams['font.size']=fontsize
    plt.rc('text',usetex=True)
    plt.figure(figsize=(4,4))
    plt.plot(np.array(ssnbl.probes[0].t)*1000.0,np.array(ssnbl.probes[0].p)/1.0e5,'k',label="$\mathrm{Without\ BL\ Model}$",linewidth=2.0)
    plt.plot(np.array(ssbl.probes[0].t)*1000.0,np.array(ssbl.probes[0].p)/1.0e5,'r',label="$\mathrm{With\ BL\ Model}$",linewidth=2.0)
    plt.plot(tExp*1000.0,pExp/1.0e5,label="$\mathrm{Experiment}$",alpha=0.7)
    plt.axis([0,60,-.5,2])
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$p\ [\mathrm{bar}]$")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if show_results:
        plt.show()

    if results_location is not None:
        np.savez(
            os.path.join(results_location, "case3.npz"),
            pressure_with_boundary_layer=ssbl.probes[0].p,
            pressure_without_boundary_layer=ssnbl.probes[0].p,
            time_with_boundary_layer=ssbl.probes[0].t,
            time_without_boundary_layer=ssnbl.probes[0].t
        )
        plt.savefig(os.path.join(results_location, "case3.png"))



if __name__ == "__main__":
    main()
