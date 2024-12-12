#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
    Copyright 2017 Kevin Grogan
    Copyright 2024 Matthew Bonanni

    This file is part of StanScram.

    StanScram is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    StanScram is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with StanScram.  If not, see <https://www.gnu.org/licenses/>.
'''

#necessary modules
import numpy as np
from numba import double, njit, int64
import cantera as ct
import matplotlib.pyplot as plt

from StanScram.jet_in_crossflow import JICModel

#Global variables (paramters) used by the solver
mt=3 #number of ghost nodes
mn=3 #number of 1D Euler equations

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


@njit(double3D(double1D, double1D, double1D, double2D, double1D))
def WENO5(r,u,p,Y,gamma):
    '''
    Method: WENO5
    ------------------------------------------------------------.----------
    This method implements the fifth-order WENO interpolation. This method 
    follows that of Houim and Kuo (JCP2011)
        inputs:
            r=density
            u=velocity
            p=pressure
            Y=species mass fraction matrix [x,species]
            gamma=specific heat ratio
        outputs:
            ULR=a matrix of the primitive variables [LR,]
    '''
    nLR=2
    nCells = len(r)-2*mt
    nFaces = nCells+1 
    nSp= len(Y[0])
    nVar = mn+nSp-1 #excluding density
    nStencil=2*mt
    epWENO=1.0E-06
    #Cell weight (WL(i,j,k); i=left(1) or right(2) j=stencil#,k=weight#)
    W=np.empty((2,3,3))
    W[0,0,0]=0.333333333333333
    W[0,0,1]=0.833333333333333
    W[0,0,2]=-0.166666666666667
 
    W[0,1,0]=-0.166666666666667
    W[0,1,1]=0.833333333333333
    W[0,1,2]=0.333333333333333

    W[0,2,0]=0.333333333333333
    W[0,2,1]=-1.166666666666667
    W[0,2,2]=1.833333333333333
 
    W[1,0,0]=W[0,2,2] 
    W[1,0,1]=W[0,2,1]
    W[1,0,2]=W[0,2,0]
 
    W[1,1,0]=W[0,1,2]
    W[1,1,1]=W[0,1,1]
    W[1,1,2]=W[0,1,0]
 
    W[1,2,0]=W[0,0,2]
    W[1,2,1]=W[0,0,1]
    W[1,2,2]=W[0,0,0]
	#Stencil Weight (i=left(1) or right(2) j=stencil#)
    D=np.empty((2,3))
    D[0,0]=0.3
    D[0,1]=0.6
    D[0,2]=0.1

    D[1,0]=D[0,2]
    D[1,1]=D[0,1]
    D[1,2]=D[0,0]
    #Weights for smoothness parameter
    B1 = 1.083333333333333
    B2 = 0.25 
                    
    B=np.empty(mt)
    PLR=np.empty((nLR,nFaces,nVar+1)) #return array of primitives
    YAverage = np.empty(nSp)
    U = np.empty(nVar) #vector of the conservative at a face
    R=np.zeros((nVar,nVar))
    L=np.zeros((nVar,nVar))
    CStencil = np.empty((nStencil,nVar)) #all the characteristic values in the stencil
    for iFace in range(nFaces): #iterate through each cell right edge              
        iCell=iFace+2 #face is on the right side of the cell
        #find the average at the face (use ordering of Houim and Kuo, JCP2011)
        rAverage=0.5*(r[iCell]+r[iCell+1])
        uAverage=0.5*(u[iCell]+u[iCell+1])
        pAverage=0.5*(p[iCell]+p[iCell+1])
        gammaAverage=0.5*(gamma[iCell]+gamma[iCell+1])
        for kSp in range(nSp): YAverage[kSp]=0.5*(Y[iCell,kSp]+Y[iCell+1,kSp])
        eAverage=pAverage/(rAverage*(gammaAverage-1.0))+0.5*uAverage**2.0
        hAverage=eAverage+pAverage/rAverage
        cAverage=np.sqrt(gammaAverage*pAverage/rAverage)
        #compute the eigenvector matrices using the face average
        #right matrix
        for i in range(nSp):
            R[i,0]=YAverage[i]
            R[i,-1]=YAverage[i]
            R[i,i+1]=1.0
            R[-2,i+1]=uAverage
            R[-1,i+1]=0.5*uAverage**2.0
        R[-2,0]=uAverage-cAverage
        R[-1,0]=hAverage-uAverage*cAverage
        R[-2,-1]=uAverage+cAverage
        R[-1,-1]=hAverage+uAverage*cAverage
        #left matrix
        gammaHat=gammaAverage-1.0
        phi=0.5*gammaHat*uAverage**2.0
        firstRowConstant=0.5*(phi+uAverage*cAverage)
        lastRowConstant=0.5*(phi-uAverage*cAverage)
        for i in range(nSp):
            for j in range(nSp): 
                L[i+1,j]=-YAverage[i]*phi
            L[i+1,i]=L[i+1,i]+cAverage**2.0
            L[0,i]=firstRowConstant
            L[-1,i]=lastRowConstant
            L[i+1,-2]=YAverage[i]*gammaHat*uAverage
            L[i+1,-1]=-YAverage[i]*gammaHat
        L[0,-2]=-0.5*(gammaHat*uAverage+cAverage)
        L[-1,-2]=-0.5*(gammaHat*uAverage-cAverage)
        L[0,-1]=gammaHat/2.0
        L[-1,-1]=gammaHat/2.0 
        L/=cAverage**2.0
        for iVar in range(nVar):
            for iStencil in range(nStencil):
                 iCellStencil=iStencil-2+iCell
                 #compute the conservative variables
                 for kSp in range(nSp): U[kSp]=r[iCellStencil]*Y[iCellStencil,kSp]
                 U[-2]=r[iCellStencil]*u[iCellStencil]
                 U[-1]=p[iCellStencil]/(gammaAverage-1.0)+0.5*r[iCellStencil]*u[iCellStencil]**2.0
                 #compute the characteristic variables in the stencil
                 CStencil[iStencil,iVar]=0.0
                 for jVar in range(nVar): 
                     CStencil[iStencil,iVar]+=L[iVar,jVar]*U[jVar]
        #perform the WENO interpolation in the characteristic variables
        for N in range(nLR): #!left edge and right edge
            for iVar in range(nVar): U[iVar]=0.0
            for iVar in range(nVar):
                NO =N+2 #!offset index
                #Find smoothness parameters 	
                B[0]=B1*(CStencil[0+NO,iVar]-2.0*CStencil[1+NO,iVar]+CStencil[2+NO,iVar])**2.0+B2*(3.0*CStencil[0+NO,iVar]-4.0*CStencil[1+NO,iVar]+CStencil[2+NO,iVar])**2
                B[1]=B1*(CStencil[-1+NO,iVar]-2.0*CStencil[0+NO,iVar]+CStencil[1+NO,iVar])**2.0+B2*(CStencil[-1+NO,iVar]-CStencil[1+NO,iVar])**2
                B[2]=B1*(CStencil[-2+NO,iVar]-2.0*CStencil[-1+NO,iVar]+CStencil[0+NO,iVar])**2.0+B2*(CStencil[-2+NO,iVar]-4.0*CStencil[-1+NO,iVar]+3.0*CStencil[0+NO,iVar])**2
                #Find the interpolated values at the cell edges
                ATOT = 0.0
                CW=0.0
                for iStencil in range(mt): #iterate through each stencil
                    iStencilO=NO-iStencil #offset iStencil index		
                    CINT=W[N,iStencil,0]*CStencil[0+iStencilO,iVar]+W[N,iStencil,1]*CStencil[1+iStencilO,iVar]+W[N,iStencil,2]*CStencil[2+iStencilO,iVar]
                    A=D[N,iStencil]/((epWENO+B[iStencil])**2)
                    ATOT+=A
                    CW+=CINT*A
                CiVar=CW/ATOT
                #compute the conservative vector using the eigenvector matrix
                for jVar in range(nVar): U[jVar]+=R[jVar,iVar]*CiVar
            rLR=0.0
            for kSp in range(nSp): rLR+=U[kSp]
            uLR=U[-2]/rLR
            eLR=U[-1]/rLR
            pLR=rLR*(gammaAverage-1.0)*(eLR-0.5*uLR**2.0)
            #fill primitive matrix in the following order (r,u,p,Y)
            PLR[N,iFace,0]=rLR
            PLR[N,iFace,1]=uLR
            PLR[N,iFace,2]=pLR
            for kSp in range(nSp): PLR[N,iFace,kSp+mn]=U[kSp]/rLR
    #apply first order interpolation at boundaries
    for N in range(nLR):
        for iFace in range(mt):
            iCell=iFace+2
            PLR[N,iFace,0]=r[iCell+N]
            PLR[N,iFace,1]=u[iCell+N]
            PLR[N,iFace,2]=p[iCell+N]
            for kSp in range(nSp): PLR[N,iFace,kSp+mn]=Y[iCell+N,kSp]
        for iFace in range(nFaces-mt,nFaces):
            iCell=iFace+2
            PLR[N,iFace,0]=r[iCell+N]
            PLR[N,iFace,1]=u[iCell+N]
            PLR[N,iFace,2]=p[iCell+N]
            for kSp in range(nSp): PLR[N,iFace,kSp+mn]=Y[iCell+N,kSp]
    #create primitive matrix
    P = np.zeros((nCells+2*mt,nVar+1))
    P[:,0] = r[:]
    P[:,1] = u[:]
    P[:,2] = p[:]
    P[:,mn:] = Y[:,:]
    #apply limiter
    alpha=2.0
    threshold=1e-6
    epsilon=1.0e-15
    for N in range(nLR):
        for iFace in range(nFaces):
            for iVar in range(nVar+1):
                iCell=iFace+2+N
                iCellm1 = iCell-1+2*N
                iCellp1 = iCell+1-2*N
                iCellm2 = iCell-2+4*N
                iCellp2 = iCell+2-4*N
                #check the error threshold for smooth regions
                error=abs((-P[iCellm2,iVar]+4.0*P[iCellm1,iVar]+4.0*P[iCellp1,iVar]-P[iCellp2,iVar]+epsilon)/(6.0*P[iCell,iVar]+epsilon)-1.0)
                if error < threshold: continue
                #compute limiter
                if P[iCell,iVar] != P[iCellm1,iVar]:
                    phi=min(alpha,alpha*(P[iCellp1,iVar]-P[iCell,iVar])/(P[iCell,iVar]-P[iCellm1,iVar]))
                    phi=min(phi,2.0*(PLR[N,iFace,iVar]-P[iCell,iVar])/(P[iCell,iVar]-P[iCellm1,iVar]))
                    phi=max(0.0,phi)
                else: phi=alpha
                #apply limiter
                PLR[N,iFace,iVar]=P[iCell,iVar]+0.5*phi*(P[iCell,iVar]-P[iCellm1,iVar])
    return PLR


@njit(double2D(double2D, double2D, double2D, double3D, double1D))
def LF(rLR,uLR,pLR,YLR,gamma):
    '''
    Method: LF
    ------------------------------------------------------------.----------
    This method computes the flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right species mass fraction states
                [nLR,nFaces,nSp]
            gamma= array containing the specific heat [nFaces]
        return:
            F=modeled Euler fluxes [nFaces,mn+nSp]
    '''
    nLR=len(rLR)
    nFaces = len(rLR[0])
    nSp=YLR[0].shape[1]
    nDim=mn+nSp
    
    #find the maximum wave speed
    lambdaMax=0.0
    for iFace in range(nFaces):
        a=max(np.sqrt(gamma[iFace]*pLR[0,iFace]/rLR[0,iFace]),np.sqrt(gamma[iFace]*pLR[1,iFace]/rLR[1,iFace]))
        u=max(abs(uLR[0,iFace]),abs(uLR[1,iFace]))
        lambdaMax=max(lambdaMax,u+a)
    lambdaMax*=0.9
    #find the regular flux
    FLR=np.empty((2,nFaces,nDim))
    for K in range(nLR):
        for iFace in range(nFaces):
            FLR[K,iFace,0]=rLR[K,iFace]*uLR[K,iFace]
            FLR[K,iFace,1]=rLR[K,iFace]*uLR[K,iFace]**2.0+pLR[K,iFace]
            FLR[K,iFace,2]=uLR[K,iFace]*(gamma[iFace]/(gamma[iFace]-1)*pLR[K,iFace]+0.5*rLR[K,iFace]*uLR[K,iFace]**2.0)
            for kSp in range(nSp): FLR[K,iFace,mn+kSp]=rLR[K,iFace]*uLR[K,iFace]*YLR[K,iFace,kSp]
        
    #compute the modeled flux
    F=np.empty((nFaces,mn+nSp))
    U=np.empty((nLR,mn+nSp))
    for iFace in range(nFaces):
        for K in range(nLR):
            U[K,0]=rLR[K,iFace]
            U[K,1]=rLR[K,iFace]*uLR[K,iFace]
            U[K,2]=pLR[K,iFace]/(gamma[iFace]-1.0)+0.5*rLR[K,iFace]*uLR[K,iFace]**2.0
            for kSp in range(nSp): U[K,mn+kSp]=rLR[K,iFace]*YLR[K,iFace,kSp]
        for iDim in range(nDim): 
            FBar=0.5*(FLR[0,iFace,iDim]+FLR[1,iFace,iDim])
            F[iFace,iDim]=FBar-0.5*lambdaMax*(U[1,iDim]-U[0,iDim])
    return F


@njit(double2D(double2D, double2D, double2D, double3D, double1D))
def HLLC(rLR,uLR,pLR,YLR,gamma):
    '''
    Method: HLLC
    ------------------------------------------------------------.----------
    This method computes the flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right species mass fraction states
                [nLR,nFaces,nSp]
            gamma= array containing the specific heat [nFaces]
        return:
            F=modeled Euler fluxes [nFaces,mn+nSp]
    '''
    nLR=len(rLR)
    nFaces = len(rLR[0])
    nSp=YLR[0].shape[1]
    nDim=mn+nSp
    
    #compute the wave speeds
    aLR=np.empty((2,nFaces))
    qLR=np.empty((2,nFaces))
    SLR=np.empty((2,nFaces))
    SStar=np.empty(nFaces)
    for iFace in range(nFaces):
        aLR[0,iFace]= np.sqrt(gamma[iFace]*pLR[0,iFace]/rLR[0,iFace])
        aLR[1,iFace]= np.sqrt(gamma[iFace]*pLR[1,iFace]/rLR[1,iFace])
        aBar=0.5*(aLR[0,iFace]+aLR[1,iFace])  
        pBar=0.5*(pLR[0,iFace]+pLR[1,iFace])
        rBar=0.5*(rLR[0,iFace]+rLR[1,iFace])
        pPVRS=pBar-0.5*(uLR[1,iFace]-uLR[0,iFace])*rBar*aBar
        pStar=max(0.0,pPVRS)
        qLR[0,iFace] = np.sqrt(1.0+(gamma[iFace]+1.0)/(2.0*gamma[iFace])*(pStar/pLR[0,iFace]-1.0)) if pStar>pLR[0,iFace] else 1.0 
        qLR[1,iFace] = np.sqrt(1.0+(gamma[iFace]+1.0)/(2.0*gamma[iFace])*(pStar/pLR[1,iFace]-1.0)) if pStar>pLR[1,iFace] else 1.0
        SLR[0,iFace] = uLR[0,iFace]-aLR[0,iFace]*qLR[0,iFace]
        SLR[1,iFace] = uLR[1,iFace]+aLR[1,iFace]*qLR[1,iFace]
        SStar[iFace] = pLR[1,iFace]-pLR[0,iFace]
        SStar[iFace]+= rLR[0,iFace]*uLR[0,iFace]*(SLR[0,iFace]-uLR[0,iFace])
        SStar[iFace]-= rLR[1,iFace]*uLR[1,iFace]*(SLR[1,iFace]-uLR[1,iFace])
        SStar[iFace]/= rLR[0,iFace]*(SLR[0,iFace]-uLR[0,iFace])-rLR[1,iFace]*(SLR[1,iFace]-uLR[1,iFace])
    
    #find the regular flux
    FLR=np.empty((2,nFaces,nDim))
    for K in range(nLR):
        for iFace in range(nFaces):
            FLR[K,iFace,0]=rLR[K,iFace]*uLR[K,iFace]
            FLR[K,iFace,1]=rLR[K,iFace]*uLR[K,iFace]**2.0+pLR[K,iFace]
            FLR[K,iFace,2]=uLR[K,iFace]*(gamma[iFace]/(gamma[iFace]-1)*pLR[K,iFace]+0.5*rLR[K,iFace]*uLR[K,iFace]**2.0)
            for kSp in range(nSp): FLR[K,iFace,3+kSp]=rLR[K,iFace]*uLR[K,iFace]*YLR[K,iFace,kSp]
        
    #compute the modeled flux
    F=np.empty((nFaces,mn+nSp))
    U=np.empty(mn+nSp)
    UStar=np.empty(mn+nSp)
    YFace=np.empty(nSp)
    for iFace in range(nFaces):
        if 0.0<=SLR[0,iFace]:
            for iDim in range(nDim): F[iFace,iDim]=FLR[0,iFace,iDim]
        elif 0.0>=SLR[1,iFace]:
            for iDim in range(nDim): F[iFace,iDim]=FLR[1,iFace,iDim]
        else:
            SStarFace=SStar[iFace]
            K=0 if 0.0<=SStarFace else 1
            rFace=rLR[K,iFace]
            uFace=uLR[K,iFace]
            pFace=pLR[K,iFace]
            for kSp in range(nSp): YFace[kSp]=YLR[K,iFace,kSp]
            gammaFace=gamma[iFace]
            SFace=SLR[K,iFace]
            #conservative variable vector
            U[0]=rFace
            U[1]=rFace*uFace
            U[2]=pFace/(gammaFace-1.0)+0.5*rFace*uFace**2.0
            for kSp in range(nSp): U[mn+kSp]=rFace*YFace[kSp]
            #star conservative variable vector
            prefactor=rFace*(SFace-uFace)/(SFace-SStarFace)
            UStar[0]=prefactor
            UStar[1]=prefactor*SStarFace
            UStar[2]=prefactor*(U[2]/rFace+(SStarFace-uFace)*(SStarFace+pFace/(rFace*(SFace-uFace))))
            for iSp in range(nSp): UStar[mn+iSp]=prefactor*YFace[iSp]
            #flux update
            for iDim in range(nDim): F[iFace,iDim]=FLR[K,iFace,iDim]+SFace*(UStar[iDim]-U[iDim])
    
    return F


@njit(double1D(double2D, double1D))
def getR(Y,molecularWeights):
    '''
    function: getR_python
    --------------------------------------------------------------------------
    Function used by the thermoTable class to find the gas constant. This 
    function is compiled for speed-up.
        inputs: 
            Y: species mass fraction [nX,nSp]
            molecularWeights: species molecular weights [nSp]
        output:
            R: gas constants [nX]
    '''
    #find dimensions
    nX = len(Y[:,0])
    nSp = len(Y[0,:])
    #determine R
    R = np.zeros(nX)
    for iX in range(nX):
        molecularWeight=0.0
        for iSp in range(nSp): molecularWeight+=Y[iX,iSp]/molecularWeights[iSp]
        molecularWeight=1.0/molecularWeight
        R[iX] = ct.gas_constant/molecularWeight
    return R


@njit(double1D(double1D, double2D, double1D, double2D, double2D))
def getCp(T,Y,TTable,a,b):
    '''
    function: getCp_python
    --------------------------------------------------------------------------
    Function used by the thermoTable class to find the constant pressure 
    specific heats. This function is compiled for speed-up.
        inputs:
            T: Temperatures [nX]
            Y: species mass fraction [nX,nSp]
            TTable: table of temperatures [nT]
            a: first order coefficient for cp [nT]
            b: zeroth order coefficient for cp [nT]
        output:
            cp: constant pressure specific heat ratios [nX]
    '''
    #find dimensions
    nX = len(Y[:,0])
    nSp = len(Y[0,:])
    #find table extremes
    TMin = TTable[0];
    dT = TTable[1]-TTable[0] #assume constant steps in table
    TMax = TTable[-1]+dT
    #determine the indices
    indices = np.zeros(nX,dtype=np.int64)
    for iX in range(nX): indices[iX] = int((T[iX]-TMin)/dT)
    #determine cp
    cp = np.zeros(nX)
    for iX in range(nX):
        if (T[iX]<TMin) or (T[iX]>TMax):
            print("Temperature out of bounds:", T[iX], "not in range [", TMin, ",", TMax, "]")
            raise ValueError("Temperature out of bounds")
        index = indices[iX]
        bbar=0.0
        for iSp in range(nSp):
            bbar += Y[iX,iSp]*(a[index,iSp]/2.0*(T[iX]+TTable[index])+b[index,iSp])
        cp[iX]=bbar
    return cp


class thermoTable(object):
    '''
    Class: thermoTable
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the temperature table with the 
    relevant methods
    '''
    def __init__(self,gas):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat 
        coefficients. The coefficients are selected to retain the exact 
        enthalpies at the table points.
        '''
        nSp = gas.n_species
        self.TMin=50.0
        self.dT=100.0
        self.TMax=9950.0
        self.T = np.arange(self.TMin,self.TMax,self.dT) #vector of temperatures assuming thermal equilibrium between species
        nT = len(self.T)
        self.h = np.zeros((nT,nSp)) #matrix of species enthalpies per temperature
        #cpk = ak*T+bk for T in [Tk,Tk+1], k in {0,1,2,...,nT-1}
        self.a = np.zeros((nT,nSp)) #matrix of species first order coefficients
        self.b = np.zeros((nT,nSp)) #matrix of species zeroth order coefficients
        self.molecularWeights = gas.molecular_weights
        #determine the coefficients
        for kSp, species in enumerate(gas.species()):
            #initialize with actual cp
            cpk = species.thermo.cp(self.T[0])/self.molecularWeights[kSp] 
            hk = species.thermo.h(self.T[0])/self.molecularWeights[kSp] 
            for kT, Tk in enumerate(self.T):
                #compute next
                Tkp1 = Tk+self.dT
                hkp1=species.thermo.h(Tkp1)/self.molecularWeights[kSp]
                dh = hkp1-hk
                #store
                self.h[kT,kSp]=hk
                self.a[kT,kSp]=2.0/self.dT*(dh/self.dT-cpk)
                self.b[kT,kSp]=cpk-self.a[kT,kSp]*Tk
                #update
                cpk = self.a[kT,kSp]*(Tkp1)+self.b[kT,kSp]
                hk = hkp1
##############################################################################
    def getR(self,Y):
        '''
        Method: getR 
        --------------------------------------------------------------------------
        This method computes the mixture-specific gas constat
            inputs:
                Y: matrix of mass fractions [n,nSp]
            outputs:
                R: vector of mixture-specific gas constants [n]
        '''
        return getR(Y,self.molecularWeights)
##############################################################################
    def getCp(self,T,Y):
        '''
        Method: getCp 
        --------------------------------------------------------------------------
        This method computes the constant pressure specific heat as determined
        by Billet and Abgrall (2003) for the double flux method.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                cp: vector of constant pressure specific heats
        '''
        return getCp(T,Y,self.T,self.a,self.b)
##############################################################################
    def getH0(self,T,Y):
        '''
        Method: getH0 
        --------------------------------------------------------------------------
        This method computes the enthalpy according to Billet and Abgrall (2003).
        This is the enthalpy that is frozen over the time step
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                cp: vector of constant pressure specific heats
        '''
        if any(np.logical_or(T<self.TMin,T>self.TMax)): raise Exception("Temperature not within table") 
        nT = len(T)
        indices = [int((Tk-self.TMin)/self.dT) for Tk in T]
        h0 = np.zeros(nT)
        for k, index in enumerate(indices):
            bbar = self.a[index,:]/2.0*(T[k]+self.T[index])+self.b[index,:]
            h0[k]=np.dot(Y[k,:],self.h[index]-bbar*self.T[index])
        return h0
 ##############################################################################
    def getGamma(self,T,Y):
        '''
        Method: getGamma 
        --------------------------------------------------------------------------
        This method computes the specific heat ratio, gamma.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                gamma: vector of specific heat ratios
        '''
        cp = self.getCp(T,Y)
        R = self.getR(Y)
        gamma = cp/(cp-R)
        return gamma
 ##############################################################################
    def getTemperature(self,r,p,Y):
        '''
        Method: getTemperature 
        --------------------------------------------------------------------------
        This method applies the ideal gas law to compute the temperature
            inputs:
                r: vector of densities [n]
                p: vector of pressures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                T: vector of temperatures
        '''
        R = self.getR(Y)
        return p/(r*R)     
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def smoothingFunction(x,xShock,Delta,phiLeft,phiRight):
            '''
            Function: smoothingFunction
            ----------------------------------------------------------------------
            This helper function returns the function of the variable smoothed
            over the interface
                inputs:
                    x = numpy array of cell centers
                    phiLeft = the value of the variable on the left side
                    phiRight = the value of the variable on the right side
                    xShock = the mean of the shock location
            '''
            dphidx = (phiRight-phiLeft)/Delta
            phi = (phiLeft+phiRight)/2.0+dphidx*(x-xShock)
            phi[x<(xShock-Delta/2.0)]=phiLeft
            phi[x>(xShock+Delta/2.0)]=phiRight
            return phi
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dSFdx(x,xShock,Delta,phiLeft,phiRight):
            '''
            Function: dSFdx
            ----------------------------------------------------------------------
            This helper function returns the derivative of the smoothing function
                inputs:
                    x = numpy array of cell centers
                    phiLeft = the value of the variable on the left side
                    phiRight = the value of the variable on the right side
                    xShock = the mean of the shock location
            '''
            dphidx = (phiRight-phiLeft)/Delta
            dphidx = np.ones(len(x))*dphidx
            dphidx[x<(xShock-Delta/2.0)]=0.0
            dphidx[x>(xShock+Delta/2.0)]=0.0
            return dphidx
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class stanScram(object):
    '''
    Class: stanScram
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver stanScram.
    '''
##############################################################################
    def __init__(self,gas,**kwargs):
        '''
        Method: __init__
        ----------------------------------------------------------------------
        initialization of the object with default values. The keyword arguments
        allow the user to initialize the state
        '''
        #######################################################################
        def initializeConstant(self,state,x):
            '''
            Method: initializeConstant
            ----------------------------------------------------------------------
            This helper function initializes a constant state
                inputs:
                    state = a tuple containing the Cantera solution object at the
                            the desired thermodynamic state and the velocity:
                            (canteraSolution,u)  
                    x = the grid for the problem
            '''
            # Initialize grid
            self.n = len(x)
            self.x = x
            self.dx = self.x[1] - self.x[0]
            # Initialize state
            self.r = np.ones(self.n) * state[0].density
            self.u = np.ones(self.n) * state[1]
            self.p = np.ones(self.n) * state[0].P
            self.Y = np.zeros((self.n, gas.n_species))
            for k in range(self.__nsp):
                self.Y[:, k] = state[0].Y[k]
            self.gamma = np.ones(self.n) * (state[0].cp / state[0].cv)
            # No flame thickening
            self.F = np.ones_like(self.r)
        #######################################################################
        def initializeRiemannProblem(self,leftState,rightState,geometry): 
            '''
            Method: initializeRiemannProblem
            ----------------------------------------------------------------------
            This helper function initializes a Riemann Problem
                inputs:
                    leftState = a tuple containing the Cantera solution object at the
                               the desired thermodynamic state and the velocity:
                               (canteraSolution,u)  
                    rightState = a tuple containing the Cantera solution object at the
                               the desired thermodynamic state and the velocity:
                               (canteraSolution,u)  
                    geometry = a tuple containing the relevant geometry for the 
                               problem: (numberCells,xMinimum,xMaximum,shockLocation)
            '''
            if leftState[0].species_names!=gas.species_names or \
              rightState[0].species_names!=gas.species_names: 
                  raise Exception("Inputed gasses must be the same as the initialized gas.")
            self.n=geometry[0]
            self.x=np.linspace(geometry[1],geometry[2],self.n)
            self.dx = self.x[1]-self.x[0]
            #initialization for left state
            self.r=np.ones(self.n)*leftState[0].density
            self.u=np.ones(self.n)*leftState[1]
            self.p=np.ones(self.n)*leftState[0].P
            self.Y=np.zeros((self.n,gas.n_species)) 
            for kSp in range(self.__nsp): self.Y[:,kSp]=leftState[0].Y[kSp]
            self.gamma=np.ones(self.n)*(leftState[0].cp/leftState[0].cv)
            #right state
            index=self.x>=geometry[3]
            self.r[index]=rightState[0].density
            self.u[index]=rightState[1]
            self.p[index]=rightState[0].P
            for kSp in range(self.__nsp): self.Y[index,kSp]=rightState[0].Y[kSp]
            self.gamma[index]=rightState[0].cp/rightState[0].cv  
            self.F = np.ones_like(self.r)
        #######################################################################
        def initializeDiffuseInterface(self,leftState,rightState,geometry,Delta): 
            '''
            Method: initializeDiffuseInterface
            ----------------------------------------------------------------------
            This helper function initializes an interface smoothed over a distance
                inputs:
                    leftState = a tuple containing the Cantera solution object at the
                               the desired thermodynamic state and the velocity:
                               (canteraSolution,u)  
                    rightState =  a tuple containing the Cantera solution object at the
                               the desired thermodynamic state and the velocity:
                               (canteraSolution,u)  
                    geometry = a tuple containing the relevant geometry for the 
                               problem: (numberCells,xMinimum,xMaximum,shockLocation)
                    Delta =    distance over which the interface is smoothed linearly
            '''
            if leftState[0].species_names!=gas.species_names or \
              rightState[0].species_names!=gas.species_names: 
                  raise Exception("Inputed gasses must be the same as the initialized gas.")
            self.n=geometry[0]
            self.x=np.linspace(geometry[1],geometry[2],self.n)
            self.dx = self.x[1]-self.x[0]
            xShock = geometry[3]
            leftGas = leftState[0]
            uLeft = leftState[1]
            gammaLeft = leftGas.cp/leftGas.cv
            rightGas = rightState[0]
            uRight = rightState[1]
            gammaRight = rightGas.cp/rightGas.cv
            #initialization for left state
            self.r = smoothingFunction(self.x,xShock,Delta,leftGas.density,rightGas.density)
            self.u = smoothingFunction(self.x,xShock,Delta,uLeft,uRight)
            self.p = smoothingFunction(self.x,xShock,Delta,leftGas.P,rightGas.P)
            self.Y=np.zeros((self.n,self.gas.n_species)) 
            for kSp in range(self.__nsp): self.Y[:,kSp]=smoothingFunction(self.x,xShock,Delta,leftGas.Y[kSp],rightGas.Y[kSp])
            self.gamma=smoothingFunction(self.x,xShock,Delta,gammaLeft,gammaRight)
            self.F = np.ones_like(self.r)
        #########################################################################
        #initialize the class
        self.cfl=1.0 #stability condition
        self.dx=1.0 #grid spacing
        self.n=10  #grid size
        self.boundaryConditions=['outflow','outflow']
        self.x=np.linspace(0.0,self.dx*(self.n-1),self.n) 
        self.gas = gas #cantera solution object for the gas
        self.r=np.ones(self.n)*gas.density #density
        self.u=np.zeros(self.n) #velocity
        self.p=np.ones(self.n)*gas.P #pressure
        self.gamma=np.ones(self.n)*gas.cp/gas.cv #specific heat ratio
        self.F=np.ones(self.n) #thickening
        self.t = 0.0 #time
        self.__nsp=gas.n_species #number os chemical species
        self.Y=np.zeros((self.n,gas.n_species)) #species mass fractions
        self.Y[:,0]=np.ones(self.n)
        self.verbose=True #console output switch
        self.outputEvery=1 #number of iterations of simulation advancement between logging updates
        self.h=None #height of the channel (used for plotting only)
        self.dlnAdt = None #area of the shock tube as a function of time (needed for quasi-1D)
        self.dlnAdx = None #area of the shock tube as a function of x (needed for quasi-1D)
        self.sourceTerms = None #source term function
        self.injector = None #injector model
        self.ox_def = None #oxidizer definition
        self.fuel_def = None #fuel definition
        self.prog_def = None #progress variable definition
        self.fluxFunction=HLLC
        self.probes=[] #list of probe objects
        self.XTDiagrams=dict() #dictionary of XT diagram objects
        self.cf = None #skin friction functor
        self.thermoTable = thermoTable(gas) #thermodynamic table object
        self.optimizationIteration = 0 #counter to keep track of optimization
        self.reacting = False #flag to solver about whether to solve source terms
        self.inReactingRegion = lambda x,t: True #the reacting region of the shock tube. 
        self.includeDiffusion= False #exclude diffusion
        self.thickening=None #thickening function
        self.plotStateInterval=-1 #plot the state every n iterations
        #overwrite the default data
        for key in kwargs:
            if key in self.__dict__.keys(): self.__dict__[key]=kwargs[key]
            if key=='initializeConstant':
                initializeConstant(self,kwargs[key][0],kwargs[key][1])
            if key=='initializeRiemannProblem':
                initializeRiemannProblem(self,kwargs[key][0],kwargs[key][1],kwargs[key][2])
            if key=='initializeDiffuseInterface':
                initializeDiffuseInterface(self,kwargs[key][0],kwargs[key][1],kwargs[key][2],kwargs[key][3])
        if not self.n==len(self.x)==len(self.r)==len(self.u)==len(self.p)==len(self.gamma):
            raise Exception("Initialization Error")
        self.initZBilger()
        self.initProg()
##############################################################################
    class __probe(object):
        '''
        Class: probe
        -----------------------------------------------------------------------
        This class is used to store the relavant data for the probe
        '''
        def __init__(self):
            self.probeLocation=None
            self.name=None
            self.skipSteps=0 #number of timesteps to skip
            self.t=[]
            self.r=[] #density
            self.u=[] #velocity
            self.p=[] #pressure
            self.gamma=[] #specific heat ratio
            self.Y=[] #species mass fractions
##############################################################################
    def addProbe(self,probeLocation,skipSteps=0,probeName=None):
        '''
        Method: addProbe
        -----------------------------------------------------------------------
        This method adds a new probe to the solver
        '''
        if probeLocation>np.max(self.x) or probeLocation<np.min(self.x):
            raise Exception("Invalid Probe Location")
        if probeName == None: probeName="probe"+str(len(self.probes))
        newProbe = self.__probe()
        newProbe.probeLocation=probeLocation
        newProbe.skipSteps=0
        newProbe.name=probeName
        self.probes.append(newProbe)
##############################################################################
    class XTDiagram(object):
        '''
        Class: __XTDiagram
        --------------------------------------------------------------------------
        This class is used to store the relavant data for the XT diagram
        '''
        def __init__(self):
            self.name=None
            self.skipSteps=0 #number of timesteps to skip
            self.variable=[] #list of numpy arrays of the variable w.r.t x
            self.t=[] #list of times
            self.x=None #numpy array of x (the interpolated grid)
##############################################################################
    def __updateXTDiagram(self,XTDiagram):
        '''
        Method: __updateXTDiagram
        --------------------------------------------------------------------------
        This method updates the XT diagram.
            inputs:
                XTDiagram: the XTDiagram object 
        '''
        variable=XTDiagram.name
        gasSpecies = [species.lower() for species in self.gas.species_names]
        if variable in ["density","r","rho"]:
            XTDiagram.variable.append(np.interp(XTDiagram.x,self.x, self.r))
        elif variable in ["velocity","u"]:
            XTDiagram.variable.append(np.interp(XTDiagram.x,self.x, self.u))
        elif variable in ["pressure","p"]:
            XTDiagram.variable.append(np.interp(XTDiagram.x,self.x, self.p))
        elif variable in ["temperature","t"]:
            T = self.thermoTable.getTemperature(self.r,self.p,self.Y)
            XTDiagram.variable.append(np.interp(XTDiagram.x,self.x, T))
        elif variable in ["gamma","g","specific heat ratio", "heat capacity ratio"]:
            XTDiagram.variable.append(np.interp(XTDiagram.x,self.x, self.gamma))
        elif variable in gasSpecies:
            speciesIndex= gasSpecies.index(variable)
            XTDiagram.variable.append(np.interp(XTDiagram.x,self.x, self.Y[:,speciesIndex]))
        else: 
            raise Exception("Invalid Variable Name")
        XTDiagram.t.append(self.t)
        
##############################################################################
    def addXTDiagram(self,variable,skipSteps=0,x=None):
        '''
        Method: addXTDiagram
        --------------------------------------------------------------------------
        This method initiates the XT diagram.
            inputs:
                variable=string of the variable 
                skipSteps=
                
        '''
        newXTDiagram = self.XTDiagram()
        variable=variable.lower()
        newXTDiagram.skipSteps=skipSteps
        newXTDiagram.name=variable
        #check interpolation grid
        if x is None: newXTDiagram.x = self.x
        elif (x[-1]>self.x[-1]) or (x[0]<self.x[0]):
            raise Exception("Invalid Interpolation Grid")
        else: newXTDiagram.x = self.x
        self.__updateXTDiagram(newXTDiagram)
        #store the XT Diagram
        self.XTDiagrams[variable]=newXTDiagram
##############################################################################
    def plotXTDiagram(self,XTDiagram,limits=None):
        '''
        Method: plotXTDiagram
        --------------------------------------------------------------------------
        This method creates a contour plot of the XTDiagram data
            inputs:
                XTDiagram=XTDiagram object; obtained from the XTDiagrams dictionary
                limits = tuple of maximum and minimum for the pcolor (vMin,vMax)
                
        '''
        plt.figure()
        t = [t*1000.0 for t in XTDiagram.t]
        X, T = np.meshgrid(XTDiagram.x,t)
        variableMatrix = np.zeros(X.shape)
        for k, variablek in enumerate(XTDiagram.variable): 
            variableMatrix[k,:]=variablek
        variable=XTDiagram.name
        if variable in ["density","r","rho"]:
            plt.title(r"$\rho [\mathrm{kg/m^3}]$")
        elif variable in ["velocity","u"]:
            plt.title(r"$u [\mathrm{m/s}]$")
        elif variable in ["pressure","p"]:
            variableMatrix /= 1.0e5 #convert to bar
            plt.title(r"$p [\mathrm{bar}]$")
        elif variable in ["temperature","t"]:
            plt.title(r"$T [\mathrm{K}]$")
        elif variable in ["gamma","g","specific heat ratio", "heat capacity ratio"]:
            plt.title(r"$\gamma$")
        else: plt.title(r"$\mathrm{"+variable+"}$")
        if limits is None: plt.pcolormesh(X,T,variableMatrix,cmap='jet')
        else: plt.pcolormesh(X,T,variableMatrix,cmap='jet',vmin=limits[0],vmax=limits[1])
        plt.xlabel(r"$x [\mathrm{m}]$")
        plt.ylabel(r"$t [\mathrm{ms}]$")
        plt.axis([min(XTDiagram.x), max(XTDiagram.x), min(t), max(t)])
        plt.colorbar()
##############################################################################
    def add_h_plot(self, ax, scale=1.0):
        ax1 = ax.twinx()
        ax1.set_zorder(-np.inf)
        ax.patch.set_visible(False)
        ax1.plot(self.x*scale, self.h*scale, color='0.8', linestyle='--')
        ax1.axhline(0, color='0.8', linestyle='--')
        ax1.set_aspect('equal')
        ax1.set_ylabel('h [mm]')
        return ax1
##############################################################################
    def plotState(self, filename):
        xscale = 1.0e3
        T = self.thermoTable.getTemperature(self.r, self.p, self.Y)

        fig, ax = plt.subplots(8, 1, sharex=True, figsize=(6, 9))
        ax[0].plot(self.x*xscale, self.r)
        ax[0].set_ymargin(0.1)
        ax[0].set_ylabel(r'$\rho$ [kg/m$^3$]')
        if self.h is not None:
            self.add_h_plot(ax[0], scale=xscale)

        ax[1].plot(self.x*xscale, self.u)
        ax[1].set_ymargin(0.1)
        ax[1].set_ylabel(r'$u$ [m/s]')
        if self.h is not None:
            self.add_h_plot(ax[1], scale=xscale)

        ax[2].plot(self.x*xscale, self.p)
        ax[2].set_ymargin(0.1)
        ax[2].set_ylabel(r'$p$ [Pa]')
        if self.h is not None:
            self.add_h_plot(ax[2], scale=xscale)

        ax[3].plot(self.x*xscale, T)
        ax[3].set_ymargin(0.1)
        ax[3].set_ylabel(r'$T$ [K]')
        if self.h is not None:
            self.add_h_plot(ax[3], scale=xscale)

        M = np.zeros_like(self.x)
        for i in range(self.x.shape[0]):
            self.gas.TPY = T[i], self.p[i], self.Y[i, :]
            M[i] = np.abs(self.u[i]) / self.gas.sound_speed
        ax[4].plot(self.x*xscale, M)
        ax[4].axhline(1.0, color='r', linestyle='--')
        ax[4].set_ymargin(0.1)
        ax[4].set_ylabel(r'$M$ [-]')
        if self.h is not None:
            self.add_h_plot(ax[4], scale=xscale)

        ax[5].plot(self.x*xscale, self.Y[:, self.gas.species_index('H2')])
        ax[5].set_ymargin(0.1)
        ax[5].set_ylabel(r'$Y_{\mathrm{H}_2}$ [-]')
        if self.h is not None:
            self.add_h_plot(ax[5], scale=xscale)

        ax[6].plot(self.x*xscale, self.Y[:, self.gas.species_index('OH')])
        ax[6].set_ymargin(0.1)
        ax[6].set_ylabel(r'$Y_{\mathrm{OH}}$ [-]')
        if self.h is not None:
            self.add_h_plot(ax[6], scale=xscale)

        ax[7].plot(self.x*xscale, self.Y[:, self.gas.species_index('H2O')])
        ax[7].set_ymargin(0.1)
        ax[7].set_ylabel(r'$Y_{\mathrm{H}_2\mathrm{O}}$ [-]')
        if self.h is not None:
            self.add_h_plot(ax[7], scale=xscale)

        ax[7].set_xlabel('x [mm]')

        fig.suptitle('$t = {:.4f}$ ms'.format(self.t*1.0e3))

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
##############################################################################
    def soundSpeed(self,r,p,gamma):
        '''
        Method: soundSpeed
        ----------------------------------------------------------------------
        This method returns the speed of sound for the gas at its current state
            outputs:
                speed of sound
        '''
        return np.sqrt(gamma*p/r)
##############################################################################
    def waveSpeed(self): 
        '''
        Method: waveSpeed
        ----------------------------------------------------------------------
        This method determines the absolute maximum of the wave speed 
            outputs:
                speed of acoustic wave
        '''
        return abs(self.u)+self.soundSpeed(self.r,self.p,self.gamma)
##############################################################################
    def timeStep(self): 
        '''
        Method: timeStep
        ----------------------------------------------------------------------
        This method determines the maximal timestep in accord with the CFL
        condition
            outputs:
                timestep
        '''
        localDts = self.dx/self.waveSpeed()
        if self.includeDiffusion:
            T = self.thermoTable.getTemperature(self.r,self.p,self.Y)
            cv = self.thermoTable.getCp(T,self.Y)/self.gamma
            alpha, nu, diff  = np.zeros_like(T), np.zeros_like(T),np.zeros_like(T)
            for i,Ti in enumerate(T):
                #compute gas properties
                self.gas.TP = Ti,self.p[i]
                if self.gas.n_species>1: self.gas.Y= self.Y[i,:]
                nu[i]=self.gas.viscosity/self.gas.density
                alpha[i]=self.gas.thermal_conductivity/self.gas.density/cv[i]*self.F[i]
                diff[i]=np.max(self.gas.mix_diff_coeffs)*self.F[i]     
            viscousDts=0.5*self.dx**2.0/np.maximum(4.0/3.0*nu,np.maximum(alpha,diff))
            localDts = np.minimum(localDts,viscousDts)
        return self.cfl*min(localDts)
##############################################################################
    def applyBoundaryConditions(self,rLR,uLR,pLR,YLR):
        '''
        Method: applyBoundaryConditions
        ----------------------------------------------------------------------
        This method applies the prescribed BCs declared by the user.
        Currently, only reflecting (adiabatic wall) and outflow (symmetry) 
        boundary conditions are supported. The user may include Dirichlet 
        condition as well. This method returns the updated primitives. 
            inputs:
                rLR=density on left and right face [2,n+1]
                uLR=velocity on left and right face [2,n+1]
                pLR=pressure on left and right face [2,n+1]
                YLR=species ressure on left and right face [2,n+1,nsp]
            outputs:
                rLR=density on left and right face [2,n+1]
                uLR=velocity on left and right face [2,n+1]
                pLR=pressure on left and right face [2,n+1]
                YLR=species ressure on left and right face [2,n+1,nsp]
        '''
        for ibc in [0,1]:
            NAssign = ibc;
            NUse = 1-ibc;
            iX = -ibc
            rLR[NAssign,iX]=rLR[NUse,iX]
            uLR[NAssign,iX]=uLR[NUse,iX]
            pLR[NAssign,iX]=pLR[NUse,iX]
            YLR[NAssign,iX,:]=YLR[NUse,iX,:]
            if type(self.boundaryConditions[ibc]) is str:
                if self.boundaryConditions[ibc].lower()=='reflecting' or \
                      self.boundaryConditions[ibc].lower()=='symmetry':
                    uLR[NAssign,iX]=0.0
                elif self.verbose and self.boundaryConditions[ibc].lower()!='outflow':  
                    print('''Unrecognized Boundary Condition. Applying outflow by default.\n''')
            else:
                #assign Dirichlet conditions to (r,u,p,Y)
                if self.boundaryConditions[ibc][0] is not None: 
                    rLR[NAssign,iX]=self.boundaryConditions[ibc][0]
                if self.boundaryConditions[ibc][1] is not None: 
                    uLR[NAssign,iX]=self.boundaryConditions[ibc][1]
                if self.boundaryConditions[ibc][2] is not None: 
                    pLR[NAssign,iX]=self.boundaryConditions[ibc][2]
                if self.boundaryConditions[ibc][3] is not None: 
                    YLR[NAssign,iX,:]=self.boundaryConditions[ibc][3]
        return (rLR,uLR,pLR,YLR)
##############################################################################
    def primitiveToConservative(self,r,u,p,Y,gamma):
        '''
        Method: conservativeToPrimitive
        ----------------------------------------------------------------------
        This method transforms the primitive variables to conservative
            inputs:
                r=density
                u=velocity
                p=pressure
                Y=species mass fraction matrix [x,species]
                gamma=specific heat ratio
            outputs:
                r=density
                ru=momentum
                E=total energy
                rY=species density matrix
        '''
        ru=r*u
        E=p/(gamma-1.0)+0.5*r*u**2.0
        rY=Y*r.reshape((-1,1))
        return (r,ru,E,rY)
##############################################################################
    def conservativeToPrimitive(self,r,ru,E,rY,gamma):
        '''
        Method: conservativeToPrimitive
        ----------------------------------------------------------------------
        This method transforms the conservative variables to the primitives
            inputs:
                r=density
                ru=momentum
                E=total energy
                rY=species density matrix
                gamma=specific heat ratio
            outputs:
                r=density
                u=velocity
                p=pressure
                Y=species mass fraction matrix [x,species]
        '''
        u=ru/r
        p=(gamma-1.0)*(E-0.5*r*u**2.0)
        Y=rY/r.reshape((-1,1))
        #bound
        Y[Y>1.0]=1.0
        Y[Y<0.0]=0.0
        #scale
        Y=Y/np.sum(Y,axis=1).reshape((-1,1))
        return (r,u,p,Y)
##############################################################################
    def initZBilger(self):
        '''
        Method: initZBilger
        ----------------------------------------------------------------------
        This method initializes the Bilger mixture fraction
        '''
        self.Z_weights = np.zeros(self.gas.n_species)
        self.Z_offset  = 0.0

        i_C = self.gas.element_index('C')
        i_H = self.gas.element_index('H')
        i_O = self.gas.element_index('O')

        W_C = self.gas.atomic_weight(i_C)
        W_H = self.gas.atomic_weight(i_H)
        W_O = self.gas.atomic_weight(i_O)

        self.gas.X = self.ox_def
        Yo_C = self.gas.elemental_mass_fraction('C')
        Yo_H = self.gas.elemental_mass_fraction('H')
        Yo_O = self.gas.elemental_mass_fraction('O')

        self.gas.X = self.fuel_def
        Yf_C = self.gas.elemental_mass_fraction('C')
        Yf_H = self.gas.elemental_mass_fraction('H')
        Yf_O = self.gas.elemental_mass_fraction('O')

        s = 1.0 / (  2.0 * (Yf_C - Yo_C) / W_C
                   + 0.5 * (Yf_H - Yo_H) / W_H
                   - 1.0 * (Yf_O - Yo_O) / W_O)
        for k in range(self.gas.n_species):
            self.Z_weights[k] = (  2.0 * self.gas.n_atoms(k,i_C)
                                 + 0.5 * self.gas.n_atoms(k,i_H)
                                 - 1.0 * self.gas.n_atoms(k,i_O)) / self.gas.molecular_weights[k]
        self.Z_offset = -(  2.0 * Yo_C / W_C
                          + 0.5 * Yo_H / W_H
                          - 1.0 * Yo_O / W_O)
        
        self.Z_weights *= s
        self.Z_offset  *= s
##############################################################################
    def ZBilger(self,Y):
        '''
        Method: ZBilger
        ----------------------------------------------------------------------
        This method calculates the Bilger mixture fraction
            inputs:
                Y=species mass fraction matrix [x,species]
            outputs:
                Z=mixture fraction
        '''
        return np.dot(Y, self.Z_weights) + self.Z_offset
##############################################################################
    def initProg(self):
        '''
        Method: initProg
        ----------------------------------------------------------------------
        This method initializes the progress variable
        '''
        if self.prog_def is None:
            raise Exception("Progress Variable Not Defined")
        self.prog_weights = np.zeros(self.gas.n_species)
        for sp, val in self.prog_def.items():
            self.prog_weights[self.gas.species_index(sp)] = val
        if np.sum(self.prog_weights)==0.0:
            raise Exception("Progress Variable Weights Sum to Zero")
        self.prog_weights /= np.sum(self.prog_weights)
##############################################################################
    def Prog(self,Y):
        '''
        Method: Prog
        ----------------------------------------------------------------------
        This method computes the progress variable
            inputs:
                Y=species mass fraction matrix [x,species]
            outputs:
                progress variable
        '''
        return np.dot(Y, self.prog_weights)
############################################################################## 
    def flux(self,r,u,p,Y,gamma):
        '''
        Method: flux
        ----------------------------------------------------------------------
        This method calculates the advective flux
            inputs:
                r=density
                u=velocity
                p=pressure
                Y=species mass fraction matrix [x,species]
                gamma=specific heat ratio
            outputs:
                rhs=the update due to the flux 
        '''
        #find the left and right WENO states from the WENO interpolation
        nx=len(r)
        PLR=WENO5(r,u,p,Y,gamma)
        #extract and apply boundary conditions
        rLR=PLR[:,:,0];
        uLR=PLR[:,:,1];
        pLR=PLR[:,:,2];
        YLR=PLR[:,:,mt:];
        rLR,uLR,pLR,YLR = self.applyBoundaryConditions(rLR,uLR,pLR,YLR)
        #calculate the flux
        fL = self.fluxFunction(rLR,uLR,pLR,YLR,gamma[mt:-mt+1])
        fR = self.fluxFunction(rLR,uLR,pLR,YLR,gamma[mt-1:-mt])
        rhs = np.zeros((nx,mn+self.__nsp))
        rhs[mt:-mt,:]=-(fR[1:]-fL[:-1])/self.dx
        return rhs
############################################################################## 
    def viscousFlux(self,r,u,p,Y,gamma):
        '''
        Method: viscousFlux
        ----------------------------------------------------------------------
        This method calculates the viscous flux
            inputs:
                r=density
                u=velocity
                p=pressure
                Y=species mass fraction matrix [x,species]
                gamma=specific heat ratio
            outputs:
                rhs=the update due to the viscous flux 
        '''
        ##############################################################################
        def viscousFluxFunction(self,rLR,uLR,pLR,YLR):
            '''
            ------------------------------------------------------------.----------
            This method computes the viscous flux at each interface
                inputs:
                    rLR=array containing left and right density states [nLR,nFaces]
                    uLR=array containing left and right velocity states [nLR,nFaces]
                    pLR=array containing left and right pressure states [nLR,nFaces]
                    YLR=array containing left and right species mass fraction states
                        [nLR,nFaces,nSp]
                return:
                    f=modeled viscous fluxes [nFaces,mn+nSp]
            '''
            #get the temperature, pressure, and composition for each cell (including the two ghosts)
            nT = self.n+2
            T=np.zeros(nT)
            T[:-1] = self.thermoTable.getTemperature(rLR[0,:],pLR[0,:],YLR[0,:,:])
            T[-1] = self.thermoTable.getTemperature(np.array([rLR[1,-1]]),
                                                    np.array([pLR[1,-1]]),
                                                    np.array([YLR[1,-1,:]]).reshape((1,-1)))
            p, F, Y = np.zeros(nT), np.ones(nT), np.zeros((nT,self.__nsp))
            p[:-1], p[-1] = pLR[0,:], pLR[1,-1] 
            F[1:-1] = self.F 
            F[0], F[-1] = self.F[0], self.F[-1] #no gradient in F at boundary
            Y[:-1,:], Y[-1,:] = YLR[0,:,:], YLR[1,-1,:] 
            viscosity=np.zeros(nT)
            conductivity=np.zeros(nT)
            diffusivities=np.zeros((nT,self.__nsp))
            for i,Ti in enumerate(T):
                #compute gas properties
                self.gas.TP = Ti,p[i]
                if self.gas.n_species>1: self.gas.Y= Y[i,:]
                viscosity[i]=self.gas.viscosity
                conductivity[i]=self.gas.thermal_conductivity*F[i]
                diffusivities[i,:]=self.gas.mix_diff_coeffs*F[i]     
            #compute the gas properties at the face
            viscosity=(viscosity[1:]+viscosity[:-1])/2.0
            conductivity=(conductivity[1:]+conductivity[:-1])/2.0
            diffusivities=(diffusivities[1:,:]+diffusivities[:-1,:])/2.0
            r = ((rLR[0,:]+rLR[1,:])/2.0).reshape(-1,1) 
            #get the central differences
            dudx=(uLR[1,:]-uLR[0,:])/self.dx
            dTdx=(T[1:]-T[:-1])/self.dx
            dYdx=(YLR[1,:,:]-YLR[0,:,:])/self.dx
            #compute the fluxes
            f=np.zeros((nT-1,mn+self.__nsp))
            f[:,1]=4.0/3.0*viscosity*dudx
            f[:,2]=conductivity*dTdx
            f[:,mn:]=r*diffusivities*dYdx
            return f
        ##############################################################################
        #first order interpolation to the edge states and apply boundary conditions
        rLR = np.concatenate((r[mt-1:-mt].reshape(1,-1),r[mt:-mt+1].reshape(1,-1)),axis=0)
        uLR = np.concatenate((u[mt-1:-mt].reshape(1,-1),u[mt:-mt+1].reshape(1,-1)),axis=0)
        pLR = np.concatenate((p[mt-1:-mt].reshape(1,-1),p[mt:-mt+1].reshape(1,-1)),axis=0)
        YLR = np.concatenate((Y[mt-1:-mt,:].reshape(1,-1,self.__nsp),Y[mt:-mt+1,:].reshape(1,-1,self.__nsp)),axis=0)
        rLR,uLR,pLR,YLR = self.applyBoundaryConditions(rLR,uLR,pLR,YLR)
        #calculate the flux
        f = viscousFluxFunction(self,rLR,uLR,pLR,YLR)
        rhs = np.zeros((self.n+2*mt,mn+self.__nsp))
        rhs[mt:-mt,:] = (f[1:,:]-f[:-1,:])/self.dx #central difference
        return rhs
##############################################################################
    def advanceAdvection(self,dt):
        '''
        Method: advanceAdvection
        ----------------------------------------------------------------------
        This method advances the advection terms by the prescribed timestep.
        The advection terms are integrated using RK3. 
            inputs
                dt=time step
        '''
        #initialize
        r=np.ones(self.n+2*mt)
        u=np.ones(self.n+2*mt)
        p=np.ones(self.n+2*mt)
        gamma=np.ones(self.n+2*mt)
        gamma[:mt], gamma[-mt:] = self.gamma[0], self.gamma[-1]
        Y=np.ones((self.n+2*mt,self.__nsp))
        (r[mt:-mt],u[mt:-mt],p[mt:-mt], Y[mt:-mt,:],gamma[mt:-mt])= \
            (self.r,self.u,self.p,self.Y,self.gamma)
        (r,ru,E,rY)=self.primitiveToConservative(r,u,p,Y,gamma)
        #1st stage of RK3
        rhs=self.flux(r,u,p,Y,gamma)
        r1= r +dt*rhs[:,0]
        ru1=ru+dt*rhs[:,1]
        E1= E +dt*rhs[:,2]
        rY1=rY+dt*rhs[:,mn:]
        (r1,u1,p1,Y1)=self.conservativeToPrimitive(r1,ru1,E1,rY1,gamma)
        #2nd stage of RK3
        rhs=self.flux(r1,u1,p1,Y1,gamma)
        r2= 0.75*r +0.25*r1 +0.25*dt*rhs[:,0]
        ru2=0.75*ru+0.25*ru1+0.25*dt*rhs[:,1]
        E2= 0.75*E +0.25*E1 +0.25*dt*rhs[:,2]
        rY2=0.75*rY+0.25*rY1+0.25*dt*rhs[:,mn:]
        (r2,u2,p2,Y2)=self.conservativeToPrimitive(r2,ru2,E2,rY2,gamma)
        #3rd stage of RK3
        rhs=self.flux(r2,u2,p2,Y2,gamma)
        r= (1.0/3.0)*r +(2.0/3.0)*r2 +(2.0/3.0)*dt*rhs[:,0]
        ru=(1.0/3.0)*ru+(2.0/3.0)*ru2+(2.0/3.0)*dt*rhs[:,1]
        E= (1.0/3.0)*E +(2.0/3.0)*E2 +(2.0/3.0)*dt*rhs[:,2]
        rY=(1.0/3.0)*rY+(2.0/3.0)*rY2+(2.0/3.0)*dt*rhs[:,mn:]
        (r,u,p,Y)= self.conservativeToPrimitive(r,ru,E,rY,gamma)
        #update
        T0 = self.thermoTable.getTemperature(r[mt:-mt],p[mt:-mt],Y[mt:-mt])
        gamma[mt:-mt]=self.thermoTable.getGamma(T0,Y[mt:-mt])
        (self.r,self.u,self.p,self.Y,self.gamma)=(r[mt:-mt],u[mt:-mt],p[mt:-mt],Y[mt:-mt],gamma[mt:-mt])
##############################################################################
    def advanceChemistry(self,dt):
        '''
        Method: advanceChemistry
        ----------------------------------------------------------------------
        This method advances the combustion chemistry of a reacting system. It
        is only called if the "reacting" flag is set to True. 
            inputs
                dt=time step
        '''
        if self.injector is not None:
            self.advanceChemistryFPV(dt)
        else:
            self.advanceChemistryFRC(dt)
##############################################################################
    def advanceChemistryFPV(self,dt):
        '''
        Method: advanceChemistryFPV
        ----------------------------------------------------------------------
        This method advances the combustion chemistry of a reacting system using
        the flamelet progress variable approach. It is only called if the "reacting"
        flag is set to True. 
            inputs
                dt=time step
        '''
        C = self.Prog(self.Y)
        W = self.gas.molecular_weights
        wDot = self.r.reshape((-1,1)) * self.injector.get_chemical_sources(C,self.t) # kg/m^3 * 1/s = kg/m^3/s
        wHatDot = wDot/W
        eRT = self.gas.standard_int_energies_RT
        YDot = wDot/self.r.reshape((-1,1))
        T = self.thermoTable.getTemperature(self.r,self.p,self.Y)
        TDot = -np.sum(eRT*wHatDot, axis=1)*ct.gas_constant*T/(self.r*self.gas.cv_mass)
        self.Y += YDot*dt
        self.Y[self.Y>1.0] = 1.0
        self.Y[self.Y<0.0] = 0.0
        self.Y /= np.sum(self.Y,axis=1).reshape((-1,1))
        T += TDot*dt
        for k in range(self.n):
            self.gas.TDY = T[k],self.r[k],self.Y[k,:]
            self.p[k] = self.gas.P
        T = self.thermoTable.getTemperature(self.r,self.p,self.Y)
        self.gamma = self.thermoTable.getGamma(T,self.Y)
##############################################################################
    def advanceChemistryFRC(self,dt):
        '''
        Method: advanceChemistryFRC
        ----------------------------------------------------------------------
        This method advances the combustion chemistry of a reacting system using
        finite rate chemistry. It is only called if the "reacting" flag is set to True. 
            inputs
                dt=time step
        '''
        #######################################################################
        def dydt(t,y,args):
            '''
            function: dydt
            -------------------------------------------------------------------
            this function gives the source terms of a constant volume reactor
                inputs
                    dt=time step
            '''
            #unpack the input
            r = args[0]
            F = args[1]
            Y = y[:-1]
            T = y[-1]
            #set the state for the gas object
            self.gas.TDY= T,r,Y
            #gas properties
            cv = self.gas.cv_mass
            nSp=self.gas.n_species
            W = self.gas.molecular_weights
            wHatDot = self.gas.net_production_rates #kmol/m^3.s
            wDot = wHatDot*W #kg/m^3.s
            eRT= self.gas.standard_int_energies_RT
            #compute the derivatives
            YDot = wDot/r
            TDot = -np.sum(eRT*wHatDot)*ct.gas_constant*T/(r*cv) 
            f = np.zeros(nSp+1)
            f[:-1]=YDot
            f[-1]=TDot
            return f/F
        #######################################################################
        from scipy import integrate
        #get indices
        indices = [k for k in range(self.n) if self.inReactingRegion(self.x[k],self.t)]
        Ts= self.thermoTable.getTemperature(self.r[indices],self.p[indices],self.Y[indices,:])
        #initialize integrator
        y0=np.zeros(self.gas.n_species+1)
        integrator = integrate.ode(dydt).set_integrator('lsoda')
        for TIndex, k in enumerate(indices):
            #initialize
            y0[:-1]=self.Y[k,:]
            y0[-1]=Ts[TIndex]
            args = [self.r[k],self.F[k]];
            integrator.set_initial_value(y0,0.0)
            integrator.set_f_params(args)
            #solve
            integrator.integrate(dt)
            #clip and normalize
            Y=integrator.y[:-1]
            Y[Y>1.0] = 1.0
            Y[Y<0.0] = 0.0
            Y /= np.sum(Y)
            #update
            self.Y[k,:]= Y
            T=integrator.y[-1]
            self.gas.TDY = T,self.r[k],Y
            self.p[k]=self.gas.P
        #update gamma
        T = self.thermoTable.getTemperature(self.r,self.p,self.Y)
        self.gamma=self.thermoTable.getGamma(T,self.Y)
##############################################################################
    def advanceQuasi1D(self,dt):
        '''
        Method: advanceQuasi1D
        ----------------------------------------------------------------------
        This method advances the quasi-1D terms used to model area changes in 
        the shock tube. The client must supply the functions dlnAdt and dlnAdx
        to the StanScram object.
            inputs
                dt=time step
        '''
        #######################################################################
        def dydt(t,y,args):
            '''
            function: dydt
            -------------------------------------------------------------------
            this function gives the source terms for the quasi 1D
                inputs
                    dt=time step
            '''
            #unpack the input and initialize
            x, gamma = args
            r, ru, E = y
            p=(gamma-1.0)*(E-0.5*ru**2.0/r)
            f=np.zeros(3)
            #create quasi-1D right hand side
            if self.dlnAdt!=None:
                dlnAdt=self.dlnAdt(x,t)[0]
                f[0]-=r*dlnAdt
                f[1]-=ru*dlnAdt
                f[2]-=E*dlnAdt
            if self.dlnAdx!=None:
                dlnAdx=self.dlnAdx(x,t)[0]
                f[0]-=ru*dlnAdx
                f[1]-=(ru**2.0/r)*dlnAdx
                f[2]-=(ru/r*(E+p))*dlnAdx
            return f
        #######################################################################
        from scipy import integrate
        #initialize integrator
        y0=np.zeros(3)
        integrator = integrate.ode(dydt).set_integrator('lsoda')
        (r,ru,E,_)=self.primitiveToConservative(self.r,self.u,self.p,self.Y,self.gamma)
        #determine the indices
        iIn = []
        eIn = np.arange(self.x.shape[0])
        if self.dlnAdt is not None:
            dlnAdt = self.dlnAdt(self.x,self.t)
            iIn = np.arange(self.x.shape[0])[dlnAdt!=0.0]
            eIn = np.arange(self.x.shape[0])[dlnAdt==0.0]
        #integrate implicitly
        for i in iIn:
            #initialize
            y0[:] = r[i],ru[i],E[i]
            args = np.array([self.x[i]]),self.gamma[i]
            integrator.set_initial_value(y0,self.t)
            integrator.set_f_params(args)
            #solve
            integrator.integrate(self.t+dt)
            #update
            r[i], ru[i], E[i] = integrator.y
        #integrate explicitly
        rhs = np.zeros((mn,eIn.shape[0]))
        if self.dlnAdt!=None:
            dlnAdt=self.dlnAdt(self.x,self.t)[eIn]
            rhs[0]-=r[eIn]*dlnAdt
            rhs[1]-=ru[eIn]*dlnAdt
            rhs[2]-=E[eIn]*dlnAdt
        if self.dlnAdx!=None:
            dlnAdx=self.dlnAdx(self.x,self.t)[eIn]
            rhs[0]-=ru[eIn]*dlnAdx
            rhs[1]-=(ru[eIn]**2.0/r[eIn])*dlnAdx
            rhs[2]-=(self.u[eIn]*(E[eIn]+self.p[eIn]))*dlnAdx
        #update
        r[eIn] +=dt*rhs[0]
        ru[eIn]+=dt*rhs[1]
        E[eIn] +=dt*rhs[2]
        rY = r.reshape((r.shape[0],1))*self.Y
        (self.r,self.u,self.p,_)=self.conservativeToPrimitive(r,ru,E,rY,self.gamma)
        T = self.thermoTable.getTemperature(self.r,self.p,self.Y)
        self.gamma=self.thermoTable.getGamma(T,self.Y)
##############################################################################
    def advanceDiffusion(self,dt):
        '''
        Method: advanceDiffusion
        ----------------------------------------------------------------------
        This method advances the diffusion terms in the axial direction
            inputs
                dt=time step
        '''
        #initialize
        r=np.ones(self.n+2*mt)
        u=np.ones(self.n+2*mt)
        p=np.ones(self.n+2*mt)
        gamma=np.ones(self.n+2*mt)
        gamma[:mt], gamma[-mt:] = self.gamma[0], self.gamma[-1]
        Y=np.ones((self.n+2*mt,self.__nsp))
        (r[mt:-mt],u[mt:-mt],p[mt:-mt], Y[mt:-mt,:],gamma[mt:-mt])= \
            (self.r,self.u,self.p,self.Y,self.gamma)
        (r,ru,E,rY)=self.primitiveToConservative(r,u,p,Y,gamma)
        if self.thickening!=None: self.F=self.thickening(self)
        #1st stage of RK2
        rhs=self.viscousFlux(r,u,p,Y,gamma)
        r1= r +dt*rhs[:,0]
        ru1=ru+dt*rhs[:,1]
        E1= E +dt*rhs[:,2]
        rY1=rY+dt*rhs[:,mn:]
        (r1,u1,p1,Y1)=self.conservativeToPrimitive(r1,ru1,E1,rY1,gamma)
        #2nd stage of RK2
        rhs=self.viscousFlux(r1,u1,p1,Y1,gamma)
        r=  0.5*(r+ r1 +dt*rhs[:,0])
        ru= 0.5*(ru+ru1+dt*rhs[:,1])
        E=  0.5*(E +E1 +dt*rhs[:,2])
        rY= 0.5*(rY+rY1+dt*rhs[:,mn:])
        (r,u,p,Y)=self.conservativeToPrimitive(r,ru,E,rY,gamma)
        #update
        T0 = self.thermoTable.getTemperature(r[mt:-mt],p[mt:-mt],Y[mt:-mt])
        gamma[mt:-mt]=self.thermoTable.getGamma(T0,Y[mt:-mt])
        (self.r,self.u,self.p,self.Y,self.gamma)=(r[mt:-mt],u[mt:-mt],p[mt:-mt],Y[mt:-mt],gamma[mt:-mt])
##############################################################################
    def advanceSourceTerms(self,dt):
        '''
        Method: advanceSourceTerms
        ----------------------------------------------------------------------
        This method advances the source terms in the axial direction
            inputs
                dt=time step
        '''
        #initialize
        (r,ru,E,rY)=self.primitiveToConservative(self.r,self.u,self.p,self.Y,self.gamma)
        #1st stage of RK2
        rhs = self.sourceTerms(r,ru,E,rY,self.gamma,self.x,self.t)
        r1= r +dt*rhs[:,0]
        ru1=ru+dt*rhs[:,1]
        E1= E +dt*rhs[:,2]
        rY1=rY+dt*rhs[:,mn:]
        (r1,u1,p1,Y1)=self.conservativeToPrimitive(r1,ru1,E1,rY1,self.gamma)
        #2nd stage of RK2
        rhs = self.sourceTerms(r1,ru1,E1,rY1,self.gamma,self.x,self.t+dt)
        r=  0.5*(r+ r1 +dt*rhs[:,0])
        ru= 0.5*(ru+ru1+dt*rhs[:,1])
        E=  0.5*(E +E1 +dt*rhs[:,2])
        rY= 0.5*(rY+rY1+dt*rhs[:,mn:])
        (r,u,p,Y)=self.conservativeToPrimitive(r,ru,E,rY,self.gamma)
        #update
        T0 = self.thermoTable.getTemperature(r,p,Y)
        self.gamma=self.thermoTable.getGamma(T0,Y)
        (self.r,self.u,self.p,self.Y)=(r,u,p,Y)
##############################################################################
    def advanceInjector(self,dt):
        '''
        Method: advanceInjector
        ----------------------------------------------------------------------
        This method advances the source terms from the injector using the
        jet-in-crossflow model.
            inputs
                dt=time step
        '''
        #initialize
        (r,ru,E,rY)=self.primitiveToConservative(self.r,self.u,self.p,self.Y,self.gamma)
        self.injector.update_fluid_tip_position(dt,self.t,self.u)
        #1st stage of RK2
        rhs = self.injector.get_injector_sources(r,ru,E,rY,self.gamma,self.t)
        r1= r +dt*rhs[:,0]
        ru1=ru+dt*rhs[:,1]
        E1= E +dt*rhs[:,2]
        rY1=rY+dt*rhs[:,mn:]
        (r1,u1,p1,Y1)=self.conservativeToPrimitive(r1,ru1,E1,rY1,self.gamma)
        #2nd stage of RK2
        rhs = self.injector.get_injector_sources(r1,ru1,E1,rY1,self.gamma,self.t+dt)
        r=  0.5*(r+ r1 +dt*rhs[:,0])
        ru= 0.5*(ru+ru1+dt*rhs[:,1])
        E=  0.5*(E +E1 +dt*rhs[:,2])
        rY= 0.5*(rY+rY1+dt*rhs[:,mn:])
        (r,u,p,Y)=self.conservativeToPrimitive(r,ru,E,rY,self.gamma)
        #update
        T0 = self.thermoTable.getTemperature(r,p,Y)
        self.gamma=self.thermoTable.getGamma(T0,Y)
        (self.r,self.u,self.p,self.Y)=(r,u,p,Y)
##############################################################################
    def updateProbes(self,iters):
        '''
        Method: updateProbes
        ----------------------------------------------------------------------
        This method updates all the probes to the current value
        '''
        def interpolate(xArray,qArray,x):
            '''
            function: interpolate
            ----------------------------------------------------------------------
            helper function for the probe
            '''
            xUpper = (xArray[xArray>=x])[0]
            xLower = (xArray[xArray<x])[-1]
            qUpper = (qArray[xArray>=x])[0]
            qLower = (qArray[xArray<x])[-1]
            q = qLower+(qUpper-qLower)/(xUpper-xLower)*(x-xLower)
            return q
        #update probes
        nSp = len(self.Y[0])
        YProbe= np.zeros(nSp)
        for probe in self.probes:
            if iters%(probe.skipSteps+1)==0:
                probe.t.append(self.t)
                probe.r.append((interpolate(self.x,self.r,probe.probeLocation)))
                probe.u.append((interpolate(self.x,self.u,probe.probeLocation)))
                probe.p.append((interpolate(self.x,self.p,probe.probeLocation)))
                probe.gamma.append((interpolate(self.x,self.gamma,probe.probeLocation)))
                YProbe=np.array([(interpolate(self.x,self.Y[:,kSp],probe.probeLocation))\
                                  for kSp in range(nSp)])
                probe.Y.append(YProbe)
##############################################################################
    def updateXTDiagrams(self,iters):
        '''
        Method: updateXTDiagrams
        ----------------------------------------------------------------------
        This method updates all the XT Diagrams to the current value.
        '''
        #update diagrams
        for XTDiagram in self.XTDiagrams.values():
            if iters%(XTDiagram.skipSteps+1)==0:
                self.__updateXTDiagram(XTDiagram)
##############################################################################
    def advanceSimulation(self,tFinal,res_p_target=-1.0):
        '''
        Method: advanceSimulation
        ----------------------------------------------------------------------
        This method advances the simulation until the prescribed time, tFinal 
            inputs
                    tFinal=final time
        '''
        iters = 0
        res_p = np.inf
        while self.t<tFinal and res_p>res_p_target:
            p_old = self.p
            dt=min(tFinal-self.t,self.timeStep())
            #advance advection and chemistry with Strang Splitting
            if self.reacting: self.advanceChemistry(dt/2.0)
            self.advanceAdvection(dt)
            if self.reacting: self.advanceChemistry(dt/2.0)
            #advance other terms
            if self.includeDiffusion: self.advanceDiffusion(dt)
            if self.dlnAdt!=None or self.dlnAdx!=None: self.advanceQuasi1D(dt)
            if self.sourceTerms!=None: self.advanceSourceTerms(dt)
            if self.injector!=None: self.advanceInjector(dt)
            #perform other updates
            self.t+=dt
            self.updateProbes(iters)
            self.updateXTDiagrams(iters)
            iters+=1
            res_p = np.linalg.norm(self.p-p_old)
            if self.verbose and iters%self.outputEvery==0: 
                print("Iteration: %i. Current time: %f. Time step: %e. Residual(p): %e." \
                % (iters,self.t,dt,res_p))
            if (self.plotStateInterval > 0) and (iters % self.plotStateInterval == 0):
                self.plotState("figures/anim/test_{0:05d}.png".format(iters//self.plotStateInterval))
