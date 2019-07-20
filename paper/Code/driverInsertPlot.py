#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:02:55 2018
This script makes the plot for the driver geometry
@author: kgrogan
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#user parameters

#case 3
LDriver = 142.0*0.0254
LDriven = 9.73
DOuterInsertBack3 = 3.375*0.0254
DOuterInsertFront3 = 1.25*0.0254
LOuterInsert3 = 102.0*0.0254
DInnerInsert3 = 0.625*0.0254
LInnerInsert3 = 117.0*0.0254

x3 = -np.array([-LDriver,
               -LDriver+LOuterInsert3,
               -LDriver+LOuterInsert3,
               -LDriver+LInnerInsert3,
               -LDriver+LInnerInsert3,
               0.0])
d3 = np.array([DOuterInsertBack3,
               DOuterInsertFront3,
               DInnerInsert3,
               DInnerInsert3,
               0.0,
               0.0])

#case 4
x4 = 0.0254*np.array([142,140,130,120,110,100,90,80,70,60,50,40,36,36,30,20,10,0])
d4 = 0.0254*np.array([3.25,3.21,3.01,2.81,2.61,2.41,2.21,2.01,1.81,1.61,1.41,1.21,1.13,0.00,0.00,0.00,0.00,0.00])


plt.close("all")
mpl.rcParams['font.size']=12
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.plot(x3,d3*100,'r',label='Case 3')
plt.plot(x4,d4*100,'k',label='Case 4')
plt.xlabel('Distance from Diaphragm [m]')
plt.ylabel('Insert Diameter [cm]')
plt.legend(loc='best')