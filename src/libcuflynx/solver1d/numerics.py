#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:48:16 2019

@author: beatriceghitti
"""

import math
import numpy as np
import sys


class numericalMethod:
    """
    Class for numerical methods
    """
    def __init__(self, CFL, mod):
        
        self.CFL = CFL # Courant-Friedrichs-Lewy condition number     
        self.mod = mod # MATHEMATICAL MODEL on which schemes will be applied
        self.tol = self.mod.tol
     
    def vessCellNumber(self, length, dxMax, nCellsMin):
        """
        Number of computational cells M for each vessel
        - length: vessel length [cm]
        - dxMax: maximum allowed cell size [cm]
        - nCellsMin: minimum number of cells per vessel
        """
        M = max( math.ceil(length/dxMax), nCellsMin )
        
        return M
        
    def timeStep(self, Q, dx, K, a0, m, n):
        """
        Time Step resticted to the CFL condition
        """
        nCells = Q.shape[0]
        
        eigMax = 0.
        for i in range(nCells): 
            [eig1, eig2] = self.mod.Eigenvalues(Q[i, 0], Q[i, 1]/Q[i, 0], K, a0, m, n)
            eigMax = max([abs(eig1), abs(eig2), eigMax])
            
        dt = self.CFL*dx/(eigMax + 1.0e-12)
        
        return dt
    
    def computeVesselVolume(self, vess):
        """
        Compute vessel volume
        """
        vess.V = 0.
        for i in range(vess.nCells):
            vess.V = vess.V + vess.dx*vess.Q[i,0]
        
    
    # NUMERICAL FLUXES
    def numFluxLF(self, dx, dt, QL, QR, K, a0, m, n):
        """
        Lax-Friedrichs numerical flux
        """
        FL = self.mod.pF(QL, K, a0, m, n)
        FR = self.mod.pF(QR, K, a0, m, n)
        
        fLF = 0.5*(FL+FR) - 0.5*dx/dt*(QR-QL)
        
        return fLF
    
    def numFluxLW(self, dx, dt, QL, QR, K, a0, m, n):
        """
        Lax-Wendroff numerical flux
        """
        FL = self.mod.pF(QL, K, a0, m, n)
        FR = self.mod.pF(QR, K, a0, m, n)
        
        QLW = 0.5*(QL+QR) - 0.5*dt/dx*(FR-FL)
        fLW = self.mod.pF(QLW, K, a0, m, n)
        
        return fLW
    
    def numFluxFORCE(self, dx, dt, QL, QR, K, a0, m, n):
        """
        FORCE numerical flux
        """
        fLF = self.numFluxLF(dx, dt, QL, QR, K, a0, m, n)
        fLW = self.numFluxLW(dx, dt, QL, QR, K, a0, m, n)
        
        fFORCE = 0.5*(fLF + fLW)
        
        return fFORCE
    
    def numFluxGodExactArt(self, QL, QR, K, a0, m, n):
        """
        Godunov numerical flux
        """       
        # Exact solution of the Riemann problem
        QS = self.mod.QStarExactArt(QL, QR, K, a0, m, n) # GODUNOV STATE
        # Apply the Godunov Method
        fGodExact = self.mod.pF(QS, K, a0, m, n)
        
        return fGodExact
    
    def numFluxGodExact(self, QL, QR, K, a0, m, n):
        """
        Godunov numerical flux
        """
        # Exact solution of the Riemann problem
        QS = self.mod.QStarExact(QL, QR, K, a0, m, n) # GODUNOV STATE
        # Apply the Godunov Method
        fGodExact = self.mod.pF(QS, K, a0, m, n)
        
        return fGodExact
    
    # APPROXIMATE RIEMANN SOLVERS
    def HLLfluxArt(self, QL, QR, K, a0, m, n):
        """
        HLL numerical flux for arteries
        """
        aL = QL[0]
        uL = QL[1]/aL
        aR = QR[0]
        uR = QR[1]/aR
        
        cL = self.mod.waveSpeed(aL, K, a0, m, n)
        cR = self.mod.waveSpeed(aR, K, a0, m, n)
        
        # From the two-rarefaction Riemann solver
        gamma = K/(3. * self.mod.rho * math.sqrt(a0))
        cS = 1./2.*(cL+cR) - 1./8.*(uR-uL)
        aS = (2./(3.*gamma))**2. * cS**4.
        
        # Wave speed estimates
        if aS<=aL:
            sL = uL - cL
        else:
            # Mass flux
            # mL = sqrt(gamma*aL*aS*(aS**(3/2) - aL**(3/2))/(aS-aL))
            mL = self.mod.mKart(aL, aS, K, a0)
            sL = uL - mL/aL
            
        if aS<=aR:
            sR = uR + cR
        else:
            # Mass flux 
            # mR = sqrt(gamma*aR*aS*(aS**(3/2) - aR**(3/2))/(aS-aR))
            mR = self.mod.mKart(aR, aS, K, a0)
            sR = uR + mR/aR
            
        FL = self.mod.pF(QL, K, a0, m, n)
        FR = self.mod.pF(QR, K, a0, m, n)
        
        # HLL numerical flux
        if sL>=0.:
            fHLL = FL 
        elif (sL<0. and sR>=0.):
            fHLL = (sL*sR*(QR-QL) - sL*FR + sR*FL)/(sR-sL)
        else:
            fHLL = FR
            
        return fHLL

    def HLLflux(self, QL, QR, K, a0, m, n):
        """
        General HLL numerical flux for arteries and veins
        """
        aL = QL[0]
        uL = QL[1]/aL
        aR = QR[0]
        uR = QR[1]/aR
        
        cL = self.mod.waveSpeed(aL, K, a0, m, n)
        cR = self.mod.waveSpeed(aR, K, a0, m, n)
        
        # From the two-rarefaction Riemann solver
        [aS, uS] = self.mod.TwoRarsolveERP(aL, aR, uL, uR, K, a0, m, n)
        
        # cS = self.mod.waveSpeed(aS)
        
        # Mass flux
        mL = self.mod.mK(aL, aS, K, a0, m, n)
        mR = self.mod.mK(aR, aS, K, a0, m, n)
        
        # Wave speed estimates
        if aS<=aL:
            sL = uL - cL
        else:
            sL = uL - mL/aL
            
        if aS<=aR:
            sR = uR + cR
        else:
            sR = uR + mR/aR
            
        FL = self.mod.pF(QL, K, a0, m, n)
        FR = self.mod.pF(QR, K, a0, m, n)
        
        # HLL numerical flux
        if (sL>=0.) and (sR>sL) :
            fHLL = FL 
        elif (sL<0.) and (sR>=0.) :
            fHLL = (sL*sR*(QR-QL) - sL*FR + sR*FL)/(sR-sL)
        else:
            fHLL = FR
            
        return fHLL
    
    # BOUNDARY CONDITIONS
    def LeftBC(self, leftBCtype, QL, QR, time, K, a0, m, n, P0):
       
        if leftBCtype[0] == 'transmissive' :
            QBL = np.copy(QL)
        elif leftBCtype[0] == 'periodic' :
            QBL = np.copy(QR)
        elif leftBCtype[0] == 'reflective' :
            QBL[0] = QL[0]
            QBL[1] = -1.0*QL[1]    
        elif leftBCtype[0] == 'ImposedPressure' :
            # GAUSSIAN WAVE - PRESSURE
            pMax = 35.0*133.32 # maximum value
            pFixed = 80*133.32 + pMax*math.exp( -100000.*(time-0.01)**2. ) # exponential function of time
            # IMPOSE BOUNDARY CONDITION
            QBL = self.mod.ImposedPressure(pFixed, QL, K, a0, m, n, P0)    
        elif leftBCtype[0] == 'ImposedFlow' :
            # GAUSSIAN WAVE - FLOW
            # V = 2.0*1.0e-4
            # qFixed = V*exp( -100000*(time-0.01)**2 ) # exponential function of time
            qFixed = 1.0e-6*math.exp( -10000.*(time-0.05)**2. )*1.0e+6
            # IMPOSE BOUNDARY CONDITION
            QBL = self.mod.ImposedFlow(qFixed, QL, K, a0, m, n)    
        # elif leftBCtype[0] == 'none' :
        #     QBL = np.array([0.0, 0.0])
        elif leftBCtype[0] == 'ImposedQLV' :
            QBL = self.mod.ImposedFlow(leftBCtype[1], QL, K, a0, m, n) 
        elif leftBCtype[0] == 'inflow' :
            # QBL = self.mod.ImposedFlow(leftBCtype[1], QL, K, a0, m, n)
            # QBL = self.mod.ImposedFlowArt(leftBCtype[1], QL, K, a0, m, n)
            QBL = np.copy(leftBCtype[1])
        elif leftBCtype[0] == 'inflow_o2' :
            QBL = np.copy(leftBCtype[1])
        # elif leftBCtype[0] == 'junction' :
        #     QBL = np.copy(leftBCtype[1])
        elif leftBCtype[0] == 'J2' or leftBCtype[0] == 'J3' or leftBCtype[0] == 'JN' :
            QBL = np.copy(leftBCtype[1])
        elif leftBCtype[0] == 'J2hyb' or leftBCtype[0] == 'J3hyb' or leftBCtype[0] == 'JNhyb' :
            # QBL = np.copy(leftBCtype[1])
            sys.exit('LeftBC :: Hybrid coupling NOT yet implemented')
        
        return QBL

    def RightBC(self, rightBCtype, QL, QR, time, K, a0, m, n, P0):
    
        if rightBCtype[0] == 'transmissive' :
            QBR = np.copy(QR)
        elif rightBCtype[0] == 'periodic' :
            QBR = np.copy(QL)
        elif rightBCtype[0] == 'reflective' :
            QBR[0] = QR[0]
            QBR[1] = -1.0*QR[1]    
        # elif rightBCtype[0] == 'none' :
        #     QBR = np.array([0.0, 0.0])
        elif rightBCtype[0] == 'ImposedPRA' :
            aR = QR[0]
            uR = QR[1]/aR
            aFixed = self.mod.aFp(rightBCtype[1], K, a0, m, n, P0)
            qFixed = ( uR - self.mod.IntegralRI(aR, aFixed, K, a0, m, n) )*aFixed
            QBR = np.array([aFixed, qFixed])
        # elif rightBCtype[0] == 'junction' :
        #     QBR = np.copy(rightBCtype[1])
        elif rightBCtype[0] == 'J2' or rightBCtype[0] == 'J3' or rightBCtype[0] == 'JN' :
            QBR = np.copy(rightBCtype[1])
        elif rightBCtype[0] == 'J2hyb' or rightBCtype[0] == 'J3hyb' or rightBCtype[0] == 'JNhyb' :
            # QBR = np.copy(rightBCtype[1])
            sys.exit('RightBC :: Hybrid coupling NOT yet implemented')
        # elif rightBCtype[0] == 'terminal' :
        #     QBR = np.copy(rightBCtype[1]) 
        elif rightBCtype[0] == 'T_singleR' or rightBCtype[0] == 'T_RCR' :
            QBR = np.copy(rightBCtype[1])
        
        return QBR
 
    
    # FIRST-ORDER EVOLVE
    def evolve1D(self, Q, dx, dt, leftBCtype, rightBCtype, NumMethod, time, K, a0, m, n, P0, mu):
        """
        Update solution from time t^n to time t^{n+1} by adoptong a first-order numerical method
        """
        nCells = Q.shape[0]
        nVar = Q.shape[1]
    
        # Enlarge the domain with left and right ghost cells
        Qnum = np.zeros((nCells+2, nVar))
        Qnum[1:nCells+1, :] = np.copy(Q)
        
        Qnum[0, :] = Q[0, :]*10000
        Qnum[-1, :] = Q[-1, :]*10000
        
        # Impose BCs
        QBL = self.LeftBC(leftBCtype, Q[0, :], Q[-1, :], time, K, a0, m, n, P0)
        QBR = self.RightBC(rightBCtype, Q[0, :], Q[-1, :], time, K, a0, m, n, P0)
        
        # Initialize the NUMERICAL FLUX and SOURCE TERM vectors
        Fnum = np.zeros((nCells+2, nVar))
        Snum = np.zeros_like(Q)
        
        # Compute left and right boundary numerical fluxes
        Fnum[1,:] = self.mod.pF(QBL, K, a0, m, n)
        Fnum[nCells+1, :] = self.mod.pF(QBR, K, a0, m, n)
        
        # Evolve the solution
        for i in range(1, nCells):
            QL = Qnum[i, :]
            QR = Qnum[i+1, :]
        
            # Numerical method
            # Compute interface numerical flux
            if NumMethod == 'GodunovExactArteries' :
                Fnum[i+1, :] = self.numFluxGodExactArt(QL, QR, K, a0, m, n) # GODUNOV EXACT
                
            elif NumMethod == 'GodunovExact' :
                Fnum[i+1, :] = self.numFluxGodExact(QL, QR, K, a0, m, n) # GODUNOV EXACT
        
            elif NumMethod == 'HLLArteries' :
                Fnum[i+1, :] = self.HLLfluxArt(QL, QR, K, a0, m, n) # HLL
                
            elif NumMethod == 'HLL' :
                Fnum[i+1, :] = self.HLLflux(QL, QR, K, a0, m, n) # HLL
                
            elif NumMethod == 'LF' :
                Fnum[i+1, :] = self.numFluxLF(dx, dt, QL, QR, K, a0, m, n) # Lax-Friedrichs
                
            elif NumMethod == 'LW' :
                Fnum[i+1, :] = self.numFluxLW(dx, dt, QL, QR, K, a0, m, n) # Lax-Wendroff
                
            elif NumMethod == 'FORCE' :
                Fnum[i+1, :] = self.numFluxFORCE(dx, dt, QL, QR, K, a0, m, n) # FORCE         
        
        # Compute source term
        for i in range(nCells):
            Snum[i,:] = self.mod.S(Q[i,:], mu)
            
        # Finite volume formula to update the solution
        Q = Q - dt/dx * ( Fnum[2:nCells+2, :] - Fnum[1:nCells+1, :] ) + dt * Snum
        
        return Q
    
    def evolve1D_mod(self, Q, dx, dt, FnumQBCL, FnumQBCR, NumMethod, time, K, a0, m, n, P0, mu):
        """
        Update solution from time t^n to time t^{n+1} by adoptong a first-order numerical method
        """
        nCells = Q.shape[0]
        nVar = Q.shape[1]
    
        # Enlarge the domain
        Qnum = np.zeros((nCells+2, nVar))
        Qnum[1:nCells+1, :] = np.copy(Q)
        
        # Qnum[0, :] = Q[0, :]*10000
        # Qnum[-1, :] = Q[-1, :]*10000 
        
        # Initialize the NUMERICAL FLUX and SOURCE TERM vectors
        Fnum = np.zeros((nCells+2, nVar))
        Snum = np.zeros_like(Q)
        
        # Impose BCs: assign left and right boundary numerical fluxes (already computed outside the evolve function)
        Fnum[1,:] = FnumQBCL
        Fnum[nCells+1, :] = FnumQBCR
        
        # Evolve the solution
        for i in range(1, nCells):
            QL = Qnum[i, :]
            QR = Qnum[i+1, :]
        
            # Numerical method
            # Compute interface numerical flux
            if NumMethod == 'GodunovExactArteries' :
                Fnum[i+1, :] = self.numFluxGodExactArt(QL, QR, K, a0, m, n) # GODUNOV EXACT
                
            elif NumMethod == 'GodunovExact' :
                Fnum[i+1, :] = self.numFluxGodExact(QL, QR, K, a0, m, n) # GODUNOV EXACT
        
            elif NumMethod == 'HLLArteries' :
                Fnum[i+1, :] = self.HLLfluxArt(QL, QR, K, a0, m, n) # HLL
                
            elif NumMethod == 'HLL' :
                Fnum[i+1, :] = self.HLLflux(QL, QR, K, a0, m, n) # HLL
                
            elif NumMethod == 'LF' :
                Fnum[i+1, :] = self.numFluxLF(dx, dt, QL, QR, K, a0, m, n) # Lax-Friedrichs
                
            elif NumMethod == 'LW' :
                Fnum[i+1, :] = self.numFluxLW(dx, dt, QL, QR, K, a0, m, n) # Lax-Wendroff
                
            elif NumMethod == 'FORCE' :
                Fnum[i+1, :] = self.numFluxFORCE(dx, dt, QL, QR, K, a0, m, n) # FORCE         
        
        # Compute source term
        for i in range(nCells):
            Snum[i,:] = self.mod.S(Q[i,:], mu)
            
        # Finite volume formula to update the solution
        Q = Q - dt/dx * ( Fnum[2:nCells+2, :] - Fnum[1:nCells+1, :] ) + dt * Snum
        
        return Q
    
        
    def slope(self, Q, dx, slopeType):
        nCells = Q.shape[0]
        nVar = Q.shape[1]
        
        dQ = np.zeros((nCells, nVar))
        for i in range(1, nCells-1):
            # we consider 2 ghost cells, one at the left boundary and one at the right boundary
            # of the domain --> nCells = total number of cells including also these 2 ghost cells  
            QL = Q[i-1,:]
            QC = Q[i,:]
            QR = Q[i+1,:]
            
            DL = (QC-QL)/dx
            DR = (QR-QC)/dx
            
            for k in range(nVar): # loop over all the variables
            
                if slopeType=='ENO':
                    if np.abs(DR[k])<np.abs(DL[k]):
                        dQ[i,k] = DR[k]
                    else:
                        dQ[i,k] = DL[k]
                
                elif slopeType=='1stOrder':
                    dQ[i,k] = 0.
                    
                elif slopeType=='Fromm':
                    dQ[i,k] = 0.5*(DR[k]+DL[k])
                    
                elif slopeType=='minmod':
                    if np.abs(DL[k])<=np.abs(DR[k]) and DL[k]*DR[k]>0.:
                        dQ[i,k] = DL[k]
                    elif np.abs(DL[k])<np.abs(DR[k]) and DL[k]*DR[k]>0.:
                        dQ[i,k] = DR[k]
                    elif DL[k]*DR[k]<0.:
                        dQ[i,k] = 0.
                        
        return dQ
    
    # SECOND-ORDER EVOLVE with MUSCL-Hancock scheme
    def evolve1D_MUSCLHancock(self, Q, dx, dt, leftBCtype, rightBCtype, slopeType, time, K, a0, m, n, P0, mu):
        """
        Update solution from time t^n to time t^{n+1} by adoptong the second-order MUSCL-Hancock scheme
        """
        nCells = Q.shape[0]
        nVar = Q.shape[1]
    
        # Enlarge the domain
        Qnum = np.zeros((nCells+2, nVar))
        Qnum[1:nCells+1, :] = np.copy(Q)
        
        Qnum[0, :] = Q[0, :]*10000.
        Qnum[-1, :] = Q[-1, :]*10000. 
        
        # Impose BCs
        QBL = self.LeftBC(leftBCtype, Q[0, :], Q[-1, :], time, K, a0, m, n, P0)
        QBR = self.RightBC(rightBCtype, Q[0, :], Q[-1, :], time, K, a0, m, n, P0)
        
        # Compute slopes
        dQ = self.slope(Qnum, dx, slopeType)
    
        # Initialize NUMERICAL FLUX and SOURCE TERM vectors
        Fnum = np.zeros((nCells+2, nVar))
        Snum = np.zeros((nCells+2, nVar))
        
        # Compute left and right boundary numerical fluxes
        Fnum[1, :] = self.mod.pF(QBL, K, a0, m, n)
        Fnum[nCells+1, :] = self.mod.pF(QBR, K, a0, m, n)
        
        # Evolve the solution
        # Numerical flux
        for i in range(1, nCells):
            dQL = dQ[i,:]
            dQR = dQ[i+1,:]
            
            # Cell-boundary extrapolated values
            QL = Qnum[i,:] - 0.5*dx*dQL 
            QR = Qnum[i,:] + 0.5*dx*dQL 
            QLL = Qnum[i+1,:] - 0.5*dx*dQR   
            QRR = Qnum[i+1,:] + 0.5*dx*dQR 
            
            # Evolution of cell-boundary values for a time 0.5*dt
            QRbar = QR - 0.5*dt/dx*( self.mod.pF(QR, K, a0, m, n) - self.mod.pF(QL, K, a0, m, n) ) + 0.5*dt*self.mod.S(QR, mu)
            QLLbar = QLL - 0.5*dt/dx*( self.mod.pF(QRR, K, a0, m, n) - self.mod.pF(QLL, K, a0, m, n) ) + 0.5*dt*self.mod.S(QLL, mu)
            
            # Solution of the classic NONLINEAR Riemann problem RP(QRbar, QLLbar) to compute the numerical flux F_{i+1/2}
            # Godunov exact
            # QS = self.mod.QStarExactArt(QRbar, QLLbar, K, a0, m, n)
            # Fnum[i+1,:] = self.mod.pF(QS, K, a0, m, n)
            # HLL
            Fnum[i+1, :] = self.HLLfluxArt(QRbar, QLLbar, K, a0, m, n)
        
        # Numerical source term
        for i in range(1, nCells+1):
            Q0 = Qnum[i,:]
            dQ0 = dQ[i,:]
            
            # Jacobian matrix of the 2x2 blood flow model evaluated in Qnum[i+1,:]
            A = self.mod.Jacobian(Q0, K, a0, m, n)
            
            # Taylor expansion in time for Qnum[i+1,:] for a time 0.5*dt
            dQ0t = np.zeros((len(dQ0), 1)) # column vector
            for j in range(len(dQ0)):
                dQ0t[j,0] = dQ0[j]
            AdQ0 = (np.dot(A, dQ0t).T)[0] # back to "row" vector
            Qhalf = Q0 + 0.5*dt*( self.mod.S(Q0, mu) - AdQ0 )
            # Qhalf = Q0 + 0.5*dt*( self.mod.S(Q0, mu) - np.dot(A, np.array([dQ0]).T).T )
            # Qhalf = Qhalf[0]
            # Qhalf = Q0 + 0.5*dt*( self.mod.S(Q0, mu) - np.dot(A, dQ0) )
            
            Snum[i,:] = self.mod.S(Qhalf, mu)
            
        # Finite volume formula to update the solution
        Q = Q - dt/dx*( Fnum[2:nCells+2, :] - Fnum[1:nCells+1, :] ) + dt*Snum[1:nCells+1]
        
        return Q
    
    def evolve1D_MUSCLHancock_mod(self, Q, dx, dt, FnumQBCL, FnumQBCR, slopeType, time, K, a0, m, n, P0, mu):
        """
        Update solution from time t^n to time t^{n+1} by adoptong the second-order MUSCL-Hancock scheme
        """
        nCells = Q.shape[0]
        nVar = Q.shape[1]
    
        # Enlarge the domain
        Qnum = np.zeros((nCells+2, nVar))
        Qnum[1:nCells+1, :] = np.copy(Q)
        
        Qnum[0, :] = Q[0, :]*10000.
        Qnum[-1, :] = Q[-1, :]*10000. 
        
        # Compute slopes
        dQ = self.slope(Qnum, dx, slopeType)
    
        # Initialize NUMERICAL FLUX and SOURCE TERM vector
        Fnum = np.zeros((nCells+2, nVar))
        Snum = np.zeros((nCells+2, nVar))
        
        # Impose BCs: assign left and right boundary numerical fluxes (already computed outside the evolve function)
        Fnum[1, :] = FnumQBCL
        Fnum[nCells+1, :] = FnumQBCR
    
        # Evolve the solution
        # Numerical flux
        for i in range(1, nCells):
            dQL = dQ[i,:]
            dQR = dQ[i+1,:]
            
            # Cell-boundary extrapolated values 
            QL = Qnum[i,:] - 0.5*dx*dQL 
            QR = Qnum[i,:] + 0.5*dx*dQL 
            QLL = Qnum[i+1,:] - 0.5*dx*dQR   
            QRR = Qnum[i+1,:] + 0.5*dx*dQR 
            
            # Evolution of cell-boundary values for a time 0.5*dt
            QRbar = QR - 0.5*dt/dx*( self.mod.pF(QR, K, a0, m, n) - self.mod.pF(QL, K, a0, m, n) ) + 0.5*dt*self.mod.S(QR, mu)
            QLLbar = QLL - 0.5*dt/dx*( self.mod.pF(QRR, K, a0, m, n) - self.mod.pF(QLL, K, a0, m, n) ) + 0.5*dt*self.mod.S(QLL, mu)
            
            # Solution of the classic NONLINEAR Riemann problem RP(QRbar, QLLbar) to compute the numerical flux F_{i+1/2}
            # # Godunov exact
            # if (np.abs(m-0.5)<self.tol and np.abs(n)<self.tol):
            #     QS = self.mod.QStarExactArt(QRbar, QLLbar, K, a0, m, n)
            #     Fnum[i+1,:] = self.mod.pF(QS, K, a0, m, n)
            # else:
            #     QS = self.mod.QStarExact(QRbar, QLLbar, K, a0, m, n)
            #     Fnum[i+1,:] = self.mod.pF(QS, K, a0, m, n)
            # HLL
            if (np.abs(m-0.5)<self.tol and np.abs(n)<self.tol):
                Fnum[i+1, :] = self.HLLfluxArt(QRbar, QLLbar, K, a0, m, n)
            else:
                Fnum[i+1, :] = self.HLLflux(QRbar, QLLbar, K, a0, m, n)
                
        # Numerical source term
        for i in range(1, nCells+1):
            Q0 = Qnum[i,:]
            dQ0 = dQ[i,:]
            
            # Jacobian matrix of the 2x2 blood flow model evaluated in Qnum[i+1,:]
            A = self.mod.Jacobian(Q0, K, a0, m, n)
            
            # Taylor expansion in time for Qnum[i+1,:] for a time 0.5*dt
            dQ0t = np.zeros((len(dQ0), 1)) # column vector
            for j in range(len(dQ0)):
                dQ0t[j,0] = dQ0[j]
            AdQ0 = (np.dot(A, dQ0t).T)[0] # back to "row" vector
            Qhalf = Q0 + 0.5*dt*( self.mod.S(Q0, mu) - AdQ0 )
            # Qhalf = Q0 + 0.5*dt*( self.mod.S(Q0, mu) - np.dot(A, np.array([dQ0]).T).T )
            # Qhalf = Qhalf[0]
            # Qhalf = Q0 + 0.5*dt*( self.mod.S(Q0, mu) - np.dot(A, dQ0) )
            
            Snum[i,:] = self.mod.S(Qhalf, mu)
            
        # Finite volume formula to update the solution
        Q = Q - dt/dx*( Fnum[2:nCells+2, :] - Fnum[1:nCells+1, :] ) + dt*Snum[1:nCells+1]
        
        return Q
    