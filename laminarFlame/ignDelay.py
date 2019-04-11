# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:16:41 2016
These functions are used for the determination of the physical timescale and 
lengthscales
@author: kgrogan
"""

import cantera as ct
import numpy as np

def chemicalPower(gas,reactor):
    '''
    Function chemicalPower
    ======================================================================
    This function returns the chemical power output of the gas at the given state
    for a given reactor
    gas: cantera phase object
    reactor: "hp" or "uv", denotes enthalpy release or internal energy release
    return: qdotppp [W/m^3]
    '''
    Wi = gas.molecular_weights
    wdoti = gas.net_production_rates*Wi #kg/m^3
    if reactor.lower()=='hp': energies = (ct.gas_constant*gas.T*gas.standard_enthalpies_RT)/Wi
    elif reactor.lower()=='uv': energies = (ct.gas_constant*gas.T*gas.standard_int_energies_RT)/Wi
    else: raise Exception("Unknown reactor type") 
    return -np.sum(wdoti*energies)

def estimateFlameSpeed(gas, tMax, returnFlameThickness=False):
    '''
    Function estimateFlameSpeed
    ======================================================================
    This function estimates the flameSpeed assuming a linear increase in the 
    temperature across the flame thickness. The formulation is given in Turns 
    (p. 264-267)
    gas:    cantera phase object
    tMax:   the maximum time to integrate to; this makes use of the UV ignition
            computation
    return: flamespeed Sl
    '''
    def averageChemicalPower(gas,tMax):
        '''
        Function: averageChemicalPower
        ======================================================================
        This function returns the average chemical Power of a HP reaction with 
        respect to the temperature
        '''
        #find the equilibrium value
        (T0,p0,X0)=gas.TPX;
        gas.equilibrate('HP');
        try:
            iH2O = gas.species_index('H2O')
            H2O='H2O';
        except:
            iH2O = gas.species_index('h2o') 
            H2O='h2o'; 
        H2Oeq=gas.Y[iH2O]
        gas.TPX=(T0,p0,X0);
        alpha=0.98; #arbitrary percentage
        #initiate the reactor network
        r = ct.IdealGasConstPressureReactor(gas);
        sim = ct.ReactorNet([r]);
        time = 0.0;
        qDotMean=0.0
        T = gas.T
        dt=tMax/100.0
        #advance simulation with guessed timestep
        while r.thermo[H2O].Y<(alpha*H2Oeq):
            time += dt
            sim.advance(time)
            qDotMean+=(gas.T-T)*chemicalPower(gas,'hp')
            T=gas.T
        return qDotMean/(gas.T-T0)
        
    Tu,rhou,Xu = gas.TDX
    p = gas.P
    qDotMean = averageChemicalPower(gas,tMax)
    Tb, rhob, Xb=gas.TDX
    XMean, TMean = (Xu+Xb)/2.0, (Tb+Tu)/2.0
    gas.TPX = TMean,p,XMean
    k = gas.thermal_conductivity
    cp = gas.cp_mass
    alpha = k/(rhou*cp)
    DeltaH=rhou*cp*(Tb-Tu)
    Sl = (2.0*alpha*qDotMean/DeltaH)**0.5
    delta = 2.0*alpha/Sl
    if returnFlameThickness: return Sl, delta
    return Sl
def estimateFlameThickness(gas, tMax):
    '''
    Function estimateFlameThickness
    ======================================================================
    This function estimates the thickness assuming a linear increase in the 
    temperature across the flame thickness. The formulation is given in Turns 
    (p. 264-267)
    gas:    cantera phase object
    tMax:   the maximum time to integrate to; this makes use of the UV ignition
            computation
    return: flame thickness
    '''
    Sl, delta = estimateFlameSpeed(gas, tMax,returnFlameThickness=True)
    print("The estimated flame speed is {0:.2f}".format(delta))
    return delta
    
def flameSpeed(gas,flameThickness,returnFlame=False):
    '''
    Function flameSpeed
    ======================================================================
    This function returns the flame speed for a gas. The cantera implementation
    is quite unstable. Therefore, this function is not very useful
        gas: cantera phase object at the desired state
        flameThickness: a guess on the flame thickness
        return: Sl
    '''
    #solution parameters
    width = 5.0*flameThickness  # m
    loglevel = 1  # amount of diagnostic output (0 to 8)
    # Flame object
    try:
        f = ct.FreeFlame(gas, width=width)
    except:
        f = ct.FreeFlame(gas)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    f.show_solution()
    f.transport_model = 'Mix'
    try:
        f.solve(loglevel=loglevel, auto=True)
    except:
        f.solve(loglevel=loglevel)
    f.show_solution()
    print('mixture-averaged flamespeed = {0:7f} m/s'.format(f.u[0]))
    #f.show_solution()
    if returnFlame:
        return f.u[0], f
    else: 
        return f.u[0]
#==============================================================================
# 
# def ignUV(gas,tmax,N):
#     '''
#     Function ignUV
#     ======================================================================
#     This function returns the ignition delay for a gas
#         gas: cantera phase object at the desired state
#         tmax: initial guess for endtime of the simulation
#         N: the number of steps in determining dt
#         return: tign
#     '''
#     (T,p,X)=gas.TPX;
#     gas.equilibrate('UV');
#     try:
#         iH2O = gas.species_index('H2O')
#         H2O='H2O'
#     except:
#         iH2O = gas.species_index('h2o') 
#         H2O='h2o'
#     H2Oeq=gas.Y[iH2O]
#     gas.TPX=(T,p,X);
#     dt = tmax/float(N);
#     alpha=0.95; #arbitrary percentage
#     #initiate the reactor network
#     T0,p0,Y0 = gas.TPY
#     #advance simulation with guessed timestep
#     it, itMin, itMax=0, 64, 1028
#     while True:
#         if it==0:
#             gas.TPY=T0, p0, Y0
#             r = ct.IdealGasReactor(gas);
#             sim = ct.ReactorNet([r]);
#             t0 = 0.0
#             time = t0
#             it+=1
#             print(dt,gas.TPY,time)
#         elif (r.thermo[H2O].Y>=(alpha*H2Oeq)) and it>=itMin:
#             break
#         elif (r.thermo[H2O].Y>=(alpha*H2Oeq)) and it<itMin:
#             #not enough data points; resolve more
#             it=0
#             dt/=2.0
#         elif it>itMax:
#             #too resolved; increase time step
#             it=0
#             dt*=2.0
#         else:
#             time += dt
#             sim.advance(time)
#             it+=1
#         return time
#==============================================================================
def ignUVTimeScales(gas,dt):
    '''
    Function ignUVAll
    ======================================================================
    This function returns the ignition delay and excitation time for the 
    reactor
        gas: cantera phase object
        dt: initial guess for the timestep
        return: (tign,texcite)
    '''
    def findTimescales(t,qdot):
        '''
        Function findTimescales
        ======================================================================
        This function determines the ignition delay and the excitation time 
        (using 5% of qdotMax).
        t: time 
        qdot: volumetric chemical energy release
        return: (tIgn,tExcite,resolution): resolution is given in number
                of timesteps between tIgn and the start of tExcitement
        '''
        qdotMax=np.max(qdot)
        k = len(qdot)-1
        kMax=0
        kExcite = 0
        while k>=0:
            if qdot[k]==qdotMax: kMax=k
            if kMax!=0 and qdot[k]<=qdotMax*0.05:
                kExcite=k
                break
            k-=1
        if kMax==0: Exception("Failed to find the excitation time")
        tExcite = t[kMax]-t[kExcite]
        tIgn=t[kMax]
        return tIgn,tExcite,kMax-kExcite, kExcite
        
    #find the equilibrium
    (T,p,X)=gas.TPX;
    gas.equilibrate('UV');
    try:
        iH2O = gas.species_index('H2O')
        H2O='H2O';
    except:
        iH2O = gas.species_index('h2o') 
        H2O='h2o'; 
    H2Oeq=gas.Y[iH2O]
    gas.TPX=(T,p,X);
    alpha=0.95; #arbitrary percentage
    #initiate the reactor network
    T0,p0,Y0 = gas.TPY
    qdot0 = chemicalPower(gas,'uv')
    #advance simulation with guessed timestep
    it, itMin, itMax=0, 64, 1028
    while True:
        if it==0:
            gas.TPY=T0, p0, Y0
            r = ct.IdealGasReactor(gas);
            sim = ct.ReactorNet([r]);
            t0 = 0.0
            time = t0
            t=[t0]
            T=[T0]
            p=[p0]
            Y=[Y0]
            qdot=[qdot0]
            it+=1
        elif (r.thermo[H2O].Y>=(alpha*H2Oeq)) and it>=itMin:
            break
        elif (r.thermo[H2O].Y>=(alpha*H2Oeq)) and it<itMin:
            #not enough data points; resolve more
            it=0
            dt/=2.0
        elif it>itMax:
            #too resolved; increase time step
            it=0
            dt*=2.0
        else:
            time += dt
            sim.advance(time)
            t.append(time)
            T.append(gas.T)
            p.append(gas.P)
            Y.append(gas.Y)
            qdot.append(chemicalPower(gas,'uv'))
            it+=1
        
    #find the ignition delay and the excitement time
    tIgn,tExcite,resolution, exciteIndex =  findTimescales(t,qdot)
    N=50
    while resolution < 50: #zoom in on excitement time
        N*=2 #enforced resolution
        kStart = max(exciteIndex-resolution,0)
        gas.TPY = T[kStart],p[kStart],Y[kStart]
        r = ct.IdealGasReactor(gas);
        sim = ct.ReactorNet([r]);
        time=0.0
        tInitial = t[kStart]
        tFinal = tIgn+2*tExcite #factor of two to make this symmetrical
        dt=(tFinal-tInitial)/float(N)
        t=[tInitial]
        T=[gas.T]
        p=[gas.P]
        Y=[gas.Y]
        qdot=[chemicalPower(gas,'uv')]
        while time<(tFinal-tInitial):
            time += dt
            sim.advance(time)
            t.append(time+tInitial)
            T.append(gas.T)
            p.append(gas.P)
            Y.append(gas.Y)
            qdot.append(chemicalPower(gas,'uv'))
        tIgn,tExcite,resolution, exciteIndex =  findTimescales(t,qdot)
    return tIgn,tExcite
