#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:52:48 2016
This script tests the validation test cases
@author: kgrogan
"""
import sys; sys.path.append('../../')
from stanShock import stanShock
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import cantera as ct

def isFloat(value):
    '''
    function isfloat
    ==========================================================================
    hacky python way to detect floats
    ''' 
    try:
        float(value)
        return True
    except ValueError:
        return False

def loadData(fileName):
    '''
    function loadData
    ==========================================================================
    This function loads the raw data contained in the csv file and initiates
    a list of dictionaries containg the data
        fileName: file name of csv data
        Return: list of dictionaries for each example
    '''
    import csv
    rawData=[]
    with open(fileName) as csvFile:
        reader = csv.DictReader(csvFile)
        for dictRow in reader:
            cleanDictRow = {key:(float(dictRow[key]) if isFloat(dictRow[key]) else dictRow[key]) for key in dictRow}
            rawData.append(cleanDictRow)
    return rawData
def getPressureData(fileName):
    '''
    function getPressureData
    ==========================================================================
    This function returns the formatted pressure vs time data
        Inputs:
            fileNamefile name of csv data
        Outputs: 
             t = time [s]
             p = pressure [Pa]
    '''   
    rawData = loadData(fileName)
    t = np.array([example["Time (s)"] for example in rawData])
    p = np.array([example["Pressure (atm)"] for example in rawData])
    p *= ct.one_atm
    return (t,p)

#=============================================================================
#provided condtions for Case 1
fileName = "case1.csv"
Ms = 2.4
T1 = 292.05
p1 = 2026.499994
p2 = 13340.21567 
tFinal=60e-3
delta = 0.5 #distance to smear the initial conditions; models incomplete initial formation of shock.

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
mech="Nitrogen.xml"
gas1 = ct.Solution(mech)
gas4 = ct.Solution(mech)
T4 = T1; #assumed
gas1.TP = T1,p1
gas4.TP = T4,p1 #use p1 as a place holder 
g1 = gas1.cp/gas1.cv
g4 = gas4.cp/gas4.cv
a4oa1 = np.sqrt(g4/g1*T4/T1*gas1.mean_molecular_weight/gas4.mean_molecular_weight)
p4=p2*(1.0-(g4-1.0)/(g1+1.0)/a4oa1*(Ms-1.0/Ms))**(-2.0*g4/(g4-1.0)) #from handbook of shock waves
p4*=1.05 #account for diaphragm
gas4.TP = T4,p4 

#set up geometry
nX = 1000 #mesh resolution
xLower = -LDriver
xUpper = LDriven
xShock = 0.0
geometry=(nX,xLower,xUpper,xShock)
DeltaD = DDriven-DDriver
DeltaX = (xUpper-xLower)/float(nX)*10 #diffuse area change for numerical stability
def D(x):
    diameter = DDriven+(DeltaD/DeltaX)*(x-xShock)
    diameter[x<(xShock-DeltaX)]=DDriver
    diameter[x>xShock]=DDriven
    return diameter
def dDdx(x):
    dDiameterdx = np.ones(len(x))*(DeltaD/DeltaX)
    dDiameterdx[x<(xShock-DeltaX)]=0.0
    dDiameterdx[x>xShock]=0.0
    return dDiameterdx
A = lambda x: np.pi/4.0*D(x)**2.0
dAdx = lambda x: np.pi/2.0*D(x)*dDdx(x)
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
                   DOuter=D,
                   Tw=T1, #assume wall temperature is in thermal eq. with gas
                   dlnAdx=dlnAdx)
ssbl.addProbe(max(ssbl.x)) #end wall probe

#Solve
t0 = time.clock()
ssbl.advanceSimulation(tFinal)
t1 = time.clock()
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
                   DOuter=D,
                   dlnAdx=dlnAdx)
ssnbl.addProbe(max(ssnbl.x)) #end wall probe

#Solve
t0 = time.clock()
ssnbl.advanceSimulation(tFinal)
t1 = time.clock()
print("The process took ", t1-t0)

#import shock tube data
tExp, pExp = getPressureData(fileName)
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