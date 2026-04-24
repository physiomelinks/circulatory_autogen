#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:21:55 2019

@author: beatriceghitti
"""

import math
import numpy as np
from scipy.integrate import quadrature
from scipy.optimize import fsolve
import sys


class model:
    """
    2x2 blood flow model
    
    Tube law: p = pe + p0 + K * ( (A/a0)^m - (A/a0)^n )
    
    That is: p = p(A; a0, K, m, n, pe, p0)
    """
    def __init__(self, rho, xi, Pe, nVar, coriolis=1., slopeTy='1stOrder'):
        
        self.rho = rho # constant blood density
        # self.mu = mu # blood dynamic viscosity, can be different for each vessel, according to Pries law for instance
        self.xi = xi # 2*(velProfOrd+2)
        self.coriolis = coriolis # Coriolis coefficient: set always to 1.
        
        self.nVar = nVar # number of state variables
        self.Q = np.zeros(self.nVar) # vector of state variables
     
        self.Pe = Pe # external pressure
        # self.P0 = P0 # reference pressure for which A=a0
        
        # tolerance parameters
        self.tol = 1.0e-11
        self.epsi = 1.0e-8
        self.nMax = 1000 # 1000000
        
        # self.num = num # NUMERICAL METHODS to solve the model
        self.slopeTy = slopeTy # slope type for second-order reconstruction
        
    def wallThicknessArtADAN(self, a0):
        """
        Wall thickness h0 as a function of reference radius r0 
        r0 must be in [cm]
        Empirical law valid for arteries only.
        """
        aa = 0.2802 # [-]
        bb = -5.053 # [1/cm]
        cc = 0.1324 # [-]
        dd = -0.1114 # [1/cm] 
        
        r0 = np.sqrt(a0/np.pi)
        h0 = r0*(aa*np.exp(bb*r0)+cc*np.exp(dd*r0))
        
        return h0
    
    def wallThicknessVen(self, r0, hr_ratio=0.05):

        h0 = hr_ratio*r0

        return h0
    
    def viscosityPries(self, a0, hd=0.5, muPlasma=0.011):
        """
        Blood viscosity as a function of vessel radius r0 according to the constitutive relation
        proposed by Pries et al. (1996)
        d = 2*r0*1.0e4 : reference vessel diameter in micrometers [um] !!!
        hd : value of hematocrit content in undisturbed blood
        muPlasma : plasma viscosity, so that mu for ADAN vessels is approx 0.04 [g/cm/s]
        muRel : relative blood viscosity with respect to viscosity of plasma
        """
        r0 = np.sqrt(a0/np.pi) # vessel reference radius in [cm]
        d = 2*r0*1.0e4 # vessel diameter in [um]
        
        c = (0.8 + np.exp(-0.075*d))*(-1. + 1./(1. + (10.**(-11.))*(d**12.))) + 1./(1. + (10.**(-11.))*(d**12.))
        mu45 = 6.*np.exp(-0.085*d) + 3.2 - 2.44*np.exp(-0.06*(d**0.645))
        muRel = (1. + (mu45 - 1.)*(((1.-hd)**c - 1.)/((1.-0.45)**c - 1.))*(d/(d-1.1))**2.) * (d/(d-1.1))**2.

        mu = muRel*muPlasma

        return mu
    
    def stiffnessParam(self, a0, Ee, m, n, h0=-1.):
        """
        Vessel wall stiffness K as a function of material properties of the vessel wall
        - reference area a0
        - elastic/Young's modulus Ee
        - wall thickness h0
        """
        r0 = np.sqrt(a0/np.pi)
        if (np.abs(m-0.5)<self.tol and np.abs(n)<self.tol): # arteries
            if h0<0.:
                h0 = self.wallThicknessArtADAN(a0)
            # K = 4./3.*np.sqrt(np.pi/a0)*h0*Ee
            K = 4./3.*Ee*(h0/r0)
        else: # veins
            if h0<0.:
                h0 = self.wallThicknessVen(r0)
            K = 1./9.*Ee*(h0/r0)**3.
        
        return K
    
    def frictionParam(self, mu):
        """
        Viscous resistance coefficient kR as a function of velocity profile
        """
        kR = self.xi*np.pi*mu/self.rho
        
        return kR

    def pFa(self, a, K, a0, m, n, P0):
        """
        Pressure p as a function of area a
        """
        psi = (a/a0)**m - (a/a0)**n
        p = K*psi + self.Pe + P0
        # transmural pressure + external pressure + reference pressure
        
        return p
    
    def aFp(self, p, K, a0, m, n, P0):
        """
        Area a as a function of pressure p
        """
        if (np.abs(m-0.5)<self.tol and np.abs(n)<self.tol):
            a = a0*( (p - self.Pe - P0)/K + 1.)**2.
            
        else:
            # INITIAL GUESS
            a = a0
            # GLOBALLY CONVERGENT NEWTON
            for i in range(self.nMax):
                f = self.pFa(a, K, a0, m, n, P0) - p
                
                if np.abs(f)<self.tol:
                    break
                else:
                    # Evaluate the derivative
                    fM = self.pFa(a - self.epsi, K, a0, m, n, P0) - p
                    fP = self.pFa(a + self.epsi, K, a0, m, n, P0) - p
                
                    df = (fP - fM)/(2.*self.epsi)
                    
                    alpha = 1.
                    for i2 in range(self.nMax):
                        aAux = a - alpha*f/df
                        if aAux>0.:
                            fAux = self.pFa(aAux, K, a0, m, n, P0) - p
                            if np.abs(fAux)<np.abs(f):
                                break
                            else:
                                alpha *= 0.8
                        else:
                            alpha *= 0.8
                    # Update a
                    a = a - alpha*f/df
                    
            if np.abs(f)>self.tol:
                print(i,a0,a,np.abs(f))
                sys.exit('aFp :: Newton algortihm did NOT convergence')
                
        return a                
                
    def dpda(self, a, K, a0, m, n):
        """
        dp/da : Derivative of pressure p with respect to area a
        """ 
        dpda = K/a0 * ( m*(a/a0)**(m-1.) - n*(a/a0)**(n-1.) )
        
        return dpda
    
    def dpsida(self, a, a0, m, n):
        """
        dpsi/da : Derivative of term psi = ((a/a0)^m-(a/a0)^n) with respect to area a
        """ 
        dpsida = 1./a0 * ( m*(a/a0)**(m-1.) - n*(a/a0)**(n-1.) )
        
        return dpsida
    
    def waveSpeed(self, a, K, a0, m, n):
        """
        Wave speed / Celerity c
        """
        dpdaVal = self.dpda(a, K, a0, m, n)   
        c = np.sqrt( (a/self.rho)*dpdaVal )
        
        return c
    
    def characteristicImpedance(self, a0, K, m, n):
        """
        Characteristic vessel impedance Z0
        """
        c0 = self.waveSpeed(a0, K, a0, m, n)
        Z0 = self.rho*c0/a0
        
        return Z0
    
    def computeKfromRefWaveSpeed(self, a0, m, n):
        """
        Vessel wall stiffness K computed from reference wave speed c0;
        ARTERIES: Reference wave speed c0 computed from empirical law proposed in [Olufsen (2000),
        Structured tree outflow condition for blood flow in larger systemic arteries.]
        c0^2 = 2/(3*rho)*(k1*exp(k2*r0)+k3),
        where k1, k2, and k3 are empirical constants chosen to achieve normal wave speeds
        in the large vessels for a young adult human and a reasonable increase in smaller vessels.
        Here, values of k1, k2, and k3 are taken from Mynard and Smolich (2015).
        Similar values are available in Toro et al. (2021)
        VEINS: Reference wave speed c0 computed using the function for reference wave speeds
        in veins proposed in Mueller and Toro (2014a) and used in Toro et al. (2021)
        c0 = c0max - (c0max-c0min)*((r0-rmin)/(rmax-rmin))^0.5,
        where rmin and rmax are the minimum and maximum vein radii in the network,
        and c0min and c0max the corresponding wave speeds. Here, values are taken 
        from Toro et al. (2021) and similar values are available in Mueller and Toro (2014a)
        """
        r0 = np.sqrt(a0/np.pi)
        
        if (np.abs(m-0.5)<self.tol and np.abs(n)<self.tol):
            # Mynard and Smolich (2015) - coefficients for systemic arteries
            k1 = 3.0e+6 # [g/s2/cm]
            k2 = -9.0 # [1/cm]
            k3 = 33.7e+4 # [g/s2/cm]
            # Toro et al. (2021)
            # k1 = 3.0e+6 # [g/s2/cm]
            # k2 = -7.0 # [1/cm]
            # k3 = 40.0e+4 # [g/s2/cm]
            c02 = 2./(3.*self.rho)*(k1*np.exp(k2*r0) + k3)
        else:
            # Strategy 1
            # Mynard and Smolich (2015) - coefficients for systemic veins
            # k1 = 0.6e+6 # [g/s2/cm]
            # k2 = -5.0 # [1/cm]
            # k3 = 2.8e+4 # [g/s2/cm]
            # c02 = 2./(3.*self.rho)*(k1*np.exp(k2*r0) + k3)
            
            # Strategy 2
            # from Mueller and Toro (2014a) & Toro et al. (2021)
            rmin = 0.08 # [cm] min vein radius in the network
            rmax = 0.8 # [cm] max vein radius in the network
            c0min = 150. # [cm/s]
            c0max = 400. # [cm/s]
            c0 = c0max - (c0max-c0min)*((r0-rmin)/(rmax-rmin))**0.5
            c02 = c0**2.
        
        dpsida0 = self.dpsida(a0, a0, m, n)
        K = self.rho*c02/(a0*dpsida0)
        
        return K
    
    def compliance1Dperunitlength(self, a, K, a0, m, n):
        """
        1D wall compliance per unit of vessel length
        defined as the differential change in area with respect to pressure
        """ 
        dpdaVal = self.dpda(a, K, a0, m, n) 
        C1D = 1./dpdaVal
        
        return C1D
    
    def Eigenvalues(self, a, u, K, a0, m, n):
        """
        Eigenvalues lambda_1 and lambda_2
        """
        c = self.waveSpeed(a, K, a0, m, n)
        lambda1 = u-c
        lambda2 = u+c
        
        return [lambda1, lambda2]
    
    def pF(self, Q, K, a0, m, n):
        """
        Physical flux
        """
        # primitive variables
        a = Q[0]
        q = Q[1]
        u = q/a
        
        F = np.zeros(self.nVar)
        
        F[0] = q
        F[1] = self.coriolis*a*u**2. + K*a/self.rho * ( m/(m+1.)*(a/a0)**m - n/(n+1.)*(a/a0)**n )
        
        return F
    
    def S(self, Q, mu):
        """
        Source term
        """
        # primitive variables
        a = Q[0]
        u = Q[1]/a
        
        S = np.zeros(self.nVar)
        
        S[0] = 0.
        S[1] = -1*(self.xi*np.pi*mu/self.rho)*u
        
        return S
    
    def Jacobian(self, Q, K, a0, m, n):
        """
        Jacobian matrix of the 2x2 blood flow model
        """
        a = Q[0]
        u = Q[1]/a
        c = self.waveSpeed(a, K, a0, m, n)
        
        J = np.zeros((2,2))
        
        J[0,0] = 0.
        J[0,1] = 1.
        J[1,0] = c**2. - u**2.
        J[1,1] = 2.*u
        
        return J
    
    def RightEigMatrix(self, Q, K, a0, m, n):
        """ 
        Matrix of right eigenvectors
        """
        a = Q[0]
        u = Q[1]/Q[0]
        c = self.waveSpeed(a, K, a0, m, n)
        
        R = np.zeros((2,2))
        
        R[0,0] = 1.
        R[0,1] = 1.
        R[1,0] = u-c 
        R[1,1] = u+c
                   
        return R

    def InvRightEigMatrix(self, Q, K, a0, m, n):
        """
        Inverse of the right eigenvector matrix
        """
        a = Q[0]
        u = Q[1]/Q[0]
        c = self.waveSpeed(a, K, a0, m, n)
        
        iR = np.zeros((2,2))
        
        iR[0,0] = (c+u)/(2.*c)
        iR[0,1] = -1./(2.*c)
        iR[1,0] = (c-u)/(2.*c)
        iR[1,1] = 1./(2.*c)
        
        return iR
      
    def lambdaMatrix(self, Q, K, a0, m, n):
        """
        Diagonal matrix of eigenvalues
        """
        a = Q[0]
        u = Q[1]/Q[0]
        c = self.waveSpeed(a, K, a0, m, n)
        
        L = np.zeros((2,2))
        
        L[0,0] = u-c 
        L[1,1] = u+c 
        
        return L
        
    def IntegralRI(self, aL, aR, K, a0, m, n):
        """
        Integral of the function c(a)/a
        """
        if (np.abs(m-0.5)<self.tol and np.abs(n)<self.tol): # exact integration
            cL = self.waveSpeed(aL, K, a0, m, n)
            cR = self.waveSpeed(aR, K, a0, m, n)
            intRI = 4.*(cR - cL)        
        else: # numerical integration
            funRI = lambda t : self.waveSpeed(t, K, a0, m, n)/t
            intRI, abserr = quadrature(funRI, aL, aR, maxiter=500)
            
        return intRI
    
    def ImposedFlowArt(self, qFixed, QL, K, a0, m, n):
        
        aL = QL[0]
        uL = QL[1]/aL
        
        # right Riemann invariant
        cL = self.waveSpeed(aL, K, a0, m, n)
        rL = uL-4.*cL # right wave with the sign MINUS
        
        # INITIAL GUESS
        a = aL
        # GLOBALLY CONVERGENT NEWTON
        for i in range(self.nMax):
            
            f = rL - qFixed/a + 4.*self.waveSpeed(a, K, a0, m, n)
        
            if np.abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fM = rL - qFixed/(a - self.epsi) + 4.*self.waveSpeed(a - self.epsi, K, a0, m, n)
                fP = rL - qFixed/(a + self.epsi) + 4.*self.waveSpeed(a + self.epsi, K, a0, m, n) 
        
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = a - alpha*f/df
                    if aAux>0.:
                        fAux = rL - qFixed/aAux + 4.*self.waveSpeed(aAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update a
                a = a - alpha*f/df
        
        if np.abs(f)>self.tol:
            print(i,a0,aL,a,np.abs(f))
            sys.exit('ImposedFlowArt :: Newton algortihm did NOT convergence')
            
        # output left boundary vector Q
        Q = np.zeros(self.nVar)
            
        Q[0] = a
        Q[1] = qFixed
        
        return Q
    
    def ImposedFlowArt_o2(self, dx, dt, qFixed, Q1D, K, a0, m, n, mu):
        
        # STENCIL for ENO reconstruction for the first vessel to impose an inflow at the inlet of the vessel
        S = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
        dQ = self.slope_single(S, dx)
        
        QL = Q1D[0] - 0.5*dx*dQ
        QR = Q1D[0] + 0.5*dx*dQ
        QLbar = QL - 0.5*dt/dx*( self.pF(QR, K, a0, m, n) - self.pF(QL, K, a0, m, n) ) + 0.5*dt*self.S(QL, mu)
        
        aL = QLbar[0]
        uL = QLbar[1]/aL
        
        # right Riemann invariant
        cL = self.waveSpeed(aL, K, a0, m, n)
        rL = uL-4.*cL # right wave with the sign MINUS
        
        # INITIAL GUESS
        a = aL
        # GLOBALLY CONVERGENT NEWTON
        for i in range(self.nMax):
            
            f = rL - qFixed/a + 4.*self.waveSpeed(a, K, a0, m, n)
        
            if np.abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fM = rL - qFixed/(a - self.epsi) + 4.*self.waveSpeed(a - self.epsi, K, a0, m, n)
                fP = rL - qFixed/(a + self.epsi) + 4.*self.waveSpeed(a + self.epsi, K, a0, m, n) 
        
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = a - alpha*f/df
                    if aAux>0.:
                        fAux = rL - qFixed/aAux + 4.*self.waveSpeed(aAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update a
                a = a - alpha*f/df
                
        if np.abs(f)>self.tol:
            print(i,a0,aL,a,np.abs(f))
            sys.exit('ImposedFlowArt_o2 :: Newton algortihm did NOT convergence')
            
        # output left boundary vector Q
        QStar = np.zeros(self.nVar)
        QStar[0] = a
        QStar[1] = qFixed
        # QStar = [a, qFixed]
        
        return QStar
    
    def ImposedFlow(self, qFixed, QL, K, a0, m, n):
        
        aL = QL[0]
        uL = QL[1]/aL
        
        # INITIAL GUESS
        a = aL
        # GLOBALLY CONVERGENT NEWTON
        for i in range(self.nMax):
            
            f = uL - qFixed/a + self.IntegralRI(aL, a, K, a0, m, n)
            
            if abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fM = uL - qFixed/(a - self.epsi) + self.IntegralRI(aL, a - self.epsi, K, a0, m, n)
                fP = uL - qFixed/(a + self.epsi) + self.IntegralRI(aL, a + self.epsi, K, a0, m, n)
            
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = a - alpha*f/df
                    if aAux>0.:
                        fAux = uL - qFixed/aAux + self.IntegralRI(aL, aAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update a
                a = a - alpha*f/df
        
        if np.abs(f)>self.tol:
            print(i,a0,aL,a,np.abs(f))
            sys.exit('ImposedFlow :: Newton algortihm did NOT convergence')         
            
        # output left boundary vector Q
        Q = np.zeros(self.nVar)
            
        Q[0] = a
        Q[1] = qFixed
        
        return Q
    
    def ImposedFlow_o2(self, dx, dt, qFixed, Q1D, K, a0, m, n, mu):
        
        # STENCIL for ENO reconstruction for the first vessel to impose an inflow at the inlet of the vessel
        S = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
        dQ = self.slope_single(S, dx)
        
        QL = Q1D[0] - 0.5*dx*dQ
        QR = Q1D[0] + 0.5*dx*dQ
        QLbar = QL - 0.5*dt/dx*( self.pF(QR, K, a0, m, n) - self.pF(QL, K, a0, m, n) ) + 0.5*dt*self.S(QL, mu)
        
        aL = QLbar[0]
        uL = QLbar[1]/aL
        
        # INITIAL GUESS
        a = aL
        # GLOBALLY CONVERGENT NEWTON
        for i in range(self.nMax):
            
            f = uL - qFixed/a + self.IntegralRI(aL, a, K, a0, m, n)
            
            if abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fM = uL - qFixed/(a - self.epsi) + self.IntegralRI(aL, a - self.epsi, K, a0, m, n)
                fP = uL - qFixed/(a + self.epsi) + self.IntegralRI(aL, a + self.epsi, K, a0, m, n)
            
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = a - alpha*f/df
                    if aAux>0.:
                        fAux = uL - qFixed/aAux + self.IntegralRI(aL, aAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update a
                a = a - alpha*f/df
        
        if np.abs(f)>self.tol:
            print(i,a0,aL,a,np.abs(f))
            sys.exit('ImposedFlow_o2 :: Newton algortihm did NOT convergence')         
            
        # output left boundary vector Q
        Q = np.zeros(self.nVar)
            
        Q[0] = a
        Q[1] = qFixed
        
        return Q
    
    def ImposedPressureArt(self, pFixed, g, Q1D, K, a0, m, n, P0):
        
        a1D = Q1D[0]
        u1D = Q1D[1]/a1D
        
        # right GRI for left boundary --> right wave with g=-1
        # or
        # left GRI for right boundary --> left wave with g=+1
        c1D = self.waveSpeed(a1D, K, a0, m, n)
        r1D = u1D + g*4.*c1D
        
        aStar = self.aFp(pFixed, K, a0, m, n, P0)
        cStar = self.waveSpeed(aStar, K, a0, m, n)
        
        qStar = (r1D - g*4.*cStar)*aStar
        
        # output left/right boundary vector QStar
        QStar = np.zeros(self.nVar)
            
        QStar[0] = aStar
        QStar[1] = qStar
        
        return QStar
    
    def ImposedPressure(self, pFixed, g, Q1D, K, a0, m, n, P0):
        
        a1D = Q1D[0]
        u1D = Q1D[1]/a1D
        
        aStar = self.aFp(pFixed, K, a0, m, n, P0)
        
        # right GRI for left boundary --> right wave with g=-1
        # or
        # left GRI for right boundary --> left wave with g=+1
        intRI = self.IntegralRI(a1D, aStar, K, a0, m, n)
        qStar = ( u1D - g*intRI )*aStar
        
        # output left/right boundary vector QStar
        QStar = np.zeros(self.nVar)
            
        QStar[0] = aStar
        QStar[1] = qStar
        
        return QStar
    
    def ImposedPressure_o2(self, dx, dt, pFixed, g, Q1D, K, a0, m, n, P0, mu):
    
        # STENCIL for ENO reconstruction to impose a prescribed pressure at the vessel inlet or outlet 
        if g==-1: # left boundary --> g=-1
            S = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
            dQ = self.slope_single(S, dx)        
            QL = Q1D[0] - 0.5*dx*dQ
            QR = Q1D[0] + 0.5*dx*dQ
            QLbar = QL - 0.5*dt/dx*( self.pF(QR, K, a0, m, n) - self.pF(QL, K, a0, m, n) ) + 0.5*dt*self.S(QL, mu)       
            a1D = QLbar[0]
            u1D = QLbar[1]/a1D
        elif g==1: # right boundary --> g=+1
            S = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
            dQ = self.slope_single(S, dx)     
            QL = Q1D[1] - 0.5*dx*dQ
            QR = Q1D[1] + 0.5*dx*dQ
            QRbar = QR - 0.5*dt/dx*( self.pF(QR, K, a0, m, n) - self.pF(QL, K, a0, m, n) ) + 0.5*dt*self.S(QR, mu)
            a1D = QRbar[0]
            u1D = QRbar[1]/a1D
            
        aStar = self.aFp(pFixed, K, a0, m, n, P0)
        
        # right GRI for left boundary --> right wave with g=-1
        # or
        # left GRI for right boundary --> left wave with g=+1
        intRI = self.IntegralRI(a1D, aStar, K, a0, m, n)
        qStar = ( u1D - g*intRI )*aStar
        
        # output left/right boundary vector QStar
        QStar = np.zeros(self.nVar)
            
        QStar[0] = aStar
        QStar[1] = qStar
        
        return QStar    
    
    # Mass flux function for arteries
    def mKart(self, aK, aS, K, a0):
        
        gamma = K/(3. * self.rho * math.sqrt(a0))
    
        mK = math.sqrt( (gamma*aK*aS*(aS**(3./2.) - aK**(3./2.))) / (aS-aK+self.tol) )
        
        return mK
    
    # General Mass flux function
    def ShockAuxFunction(self, aK, aS, K, a0, m, n):

        bK = K/self.rho * ( (m/(m+1.)*aS**(m+1.)/a0**m - n/(n+1.)*aS**(n+1.)/a0**n) - (m/(m+1.)*aK**(m+1.)/a0**m - n/(n+1.)*aK**(n+1.)/a0**n) )
        
        return bK
    
    def mK(self, aK, aS, K, a0, m, n):
        
        bK = self.ShockAuxFunction(aK, aS, K, a0, m, n)
        
        mK = math.sqrt( bK*aK*aS / (aS-aK+self.tol) )
        
        return mK
    
    # Non-linear function f for arteries
    def fKart(self, aK, aS, cK, K, a0):
        
        gamma = K/(3. * self.rho * math.sqrt(a0))
        
        if aS<=aK :
            cS = math.sqrt( gamma*(3./2.)*math.sqrt(aS) )
            fK = 4.*(cS-cK)
        else:
            fK = math.sqrt( (gamma*(aS-aK)*(aS**(3./2.) - aK**(3./2.))) / (aK*aS))
        
        return fK
    
    # General non-linear function f
    def fK(self, aK, aS, K, a0, m, n):
        
        if aS<=aK : # Rarefaction
            fK = self.IntegralRI(aK, aS, K, a0, m, n)
        else: # Shock
            bK = self.ShockAuxFunction(aK, aS, K, a0, m, n)
            fK = math.sqrt( bK*(aS-aK) / (aK*aS))
            
        return fK
    
    # Inside left rarefaction for arteries
    def rarLart(self, uL, cL, xt, K, a0):
        
        gamma = K/(3. * self.rho * math.sqrt(a0))
        
        urar = (1./5.)*(uL + 4.*cL + 4.*xt)
        crar = (1./5.)*(uL + 4.*cL - xt)
        arar = ((2./3.)*(crar**2.)/gamma)**2.
        
        return [arar, urar]
    
    # Inside right rarefaction for arteries
    def rarRart(self, uR, cR, xt, K, a0):
        
        gamma = K/(3. * self.rho * math.sqrt(a0))
        
        urar = (1./5.)*(uR - 4.*cR + 4.*xt)
        crar = (1./5.)*(-uR + 4.*cR + xt)
        arar = ((2./3.)*(crar**2.)/gamma)**2.
        
        return [arar, urar]
    
    # Exact Riemann solver for arteries   
    def solveERPart(self, aL, aR, uL, uR, K, a0, m, n):
        
        gamma = K/(3. * self.rho * math.sqrt(a0))
        
        cL = self.waveSpeed(aL, K, a0, m, n)
        cR = self.waveSpeed(aR, K, a0, m, n)
        
        # INITIAL GUESS from the two-rarefaction Riemann solver
        cS = 1./2.*(cL+cR) - 1./8.*(uR-uL)
        aS = (2./(3.*gamma))**2. * cS**4.
    
        # GLOBALLY CONVERGENT NEWTON
        # Compute aS
        for i in range(self.nMax):
            fL = self.fKart(aL, aS, cL, K, a0)
            fR = self.fKart(aR, aS, cR, K, a0)
        
            f = fL + fR + uR - uL
        
            if abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fL = self.fKart(aL, aS+self.epsi, cL, K, a0)
                fR = self.fKart(aR, aS+self.epsi, cR, K, a0)
                fP = fL + fR + uR - uL
        
                fL = self.fKart(aL, aS-self.epsi, cL, K, a0)
                fR = self.fKart(aR, aS-self.epsi, cR, K, a0)
                fM = fL + fR + uR - uL
        
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = aS - alpha*f/df
                    if aAux>0.:
                        fLAux = self.fKart(aL, aAux, cL, K, a0)
                        fRAux = self.fKart(aR, aAux, cR, K, a0)
                        fAux = fLAux + fRAux + uR - uL
                        if np.abs(fAux) < np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update aS
                aS = aS - alpha*f/df
                # print(i, aS, f)
                
        if np.abs(f)>self.tol:
            print(i,aL,uL,aR,uR,aS,np.abs(f))
            sys.exit('solveERPart :: Newton algortihm did NOT convergence')
            
        # def fun(x, aL, uL, cL, aR, uR, cR, K, a0):
        #     fL = self.fKart(aL, x, cL, K, a0)
        #     fR = self.fKart(aR, x, cR, K, a0)
        #     fun = fL + fR + uR - uL
        #     return fun
        
        # x0 = aS
        # xsol = fsolve(fun, x0, args=(aL, uL, cL, aR, uR, cR, K, a0))
        
        # aS= xsol   
        
        fL = self.fKart(aL, aS, cL, K, a0)
        fR = self.fKart(aR, aS, cR, K, a0)
        
        # Compute uS
        uS = 0.5*(uL+uR) + 0.5*(fR-fL)
        
        return [aS, uS]
    
    # Sampling for arteries
    def sampleERPart(self, aL, aR, uL, uR, aS, uS, xt, K, a0, m, n):
        
        #gamma = K/(3. * self.rho * a0**(0.5))
        
        cS = self.waveSpeed(aS, K, a0, m, n)
        cL = self.waveSpeed(aL, K, a0, m, n)
        cR = self.waveSpeed(aR, K, a0, m, n)
        
        if xt<uS: #LEFT OF CONTACT
            if aS<=aL: #LEFT RAREFACTION
                if xt<(uL-cL):
                    a = aL
                    u = uL
                elif xt<=(uS-cS):
                    arar, urar = self.rarLart(uL, cL, xt, K, a0)
                    a = arar
                    u = urar
                elif xt>(uS-cS):
                    a = aS
                    u = uS
                    
            else: #LEFT SHOCK
                mL = self.mKart(aL, aS, K, a0)
                sL = uL - (mL/aL)
                
                if xt<sL:
                    a = aL
                    u = uL
                else:
                    a = aS
                    u = uS
                    
        else: #RIGHT OF CONTACT
            if aS<=aR: #RIGHT RAREFACTION
                if xt<(uS+cS):
                    a = aS
                    u = uS
                elif xt<=(uR+cR):
                    arar, urar = self.rarRart(uR, cR, xt, K, a0)
                    a = arar
                    u = urar
                elif xt>(uR+cR):
                    a = aR
                    u = uR
                    
            else: #RIGHT SHOCK
                mR = self.mKart(aR, aS, K, a0)
                sR = uR + (mR/aR)
                
                if xt<sR:
                    a = aS
                    u = uS
                else:
                    a = aR
                    u = uR
                    
        Q = np.zeros(self.nVar)
        Q[0] = a
        Q[1] = a*u
        
        return Q
    
    # Exact Riemann problem for arteries
    def ExactRPart(self, aL, aR, uL, uR, aS, uS, x, time, gate, K, a0, m, n):
        
        #Initialize vector Q
        nCells = x.shape[0]
        Q = np.zeros((nCells, self.nVar))
    
        for i in range(nCells):
            xt = (x[i]-gate)/(time+1.0e-10)
            Q[i, :] = self.sampleERPart(aL, aR, uL, uR, aS, uS, xt, K, a0, m, n)
            
        return Q
    
    # Exact Riemann solver for arteries - Godunov method
    # GODUNOV STATE for arteries
    def QStarExactArt(self, QL, QR, K, a0, m, n):
        
        # Define values
        aL = QL[0]
        aR = QR[0]
        uL = QL[1]/aL
        uR = QR[1]/aR
            
        # Find aS and uS
        [aS, uS] = self.solveERPart(aL, aR, uL, uR, K, a0, m, n)
            
        # Sample the Godunov State: sampling of the solution along the characteristic x/t=0
        xt = 0. # t-axis --> x/t=0
        QS = self.sampleERPart(aL, aR, uL, uR, aS, uS, xt, K, a0, m, n)
    
        return QS
    
    # General TWO-RAREFACTION RIEMANN SOLVER
    def TwoRarRPfun(self, aL, aR, uL, uR, aS, K, a0, m, n):
        fL = self.IntegralRI(aL, aS, K, a0, m, n)
        fR = self.IntegralRI(aR, aS, K, a0, m, n)
        
        f = fL + fR + uR - uL
        
        return f
    
    def TwoRarsolveERP(self, aL, aR, uL, uR, K, a0, m, n):
        """
        Solve the Star Problem
        """
        # INITIAL GUESS for aS in Newton-Raphson algorithm
        aS = 0.5*(aL + aR)
        # GLOBALLY CONVERGENT NEWTON
        # Compute aS
        for i in range(self.nMax):
            f = self.TwoRarRPfun(aL, aR, uL, uR, aS, K, a0, m, n)
        
            if np.abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fM = self.TwoRarRPfun(aL, aR, uL, uR, aS-self.epsi, K, a0, m, n)
                fP = self.TwoRarRPfun(aL, aR, uL, uR, aS+self.epsi, K, a0, m, n)
        
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = aS - alpha*f/df
                    if aAux>0.:
                        fAux = self.TwoRarRPfun(aL, aR, uL, uR, aAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update aS
                aS = aS - alpha*f/df
        
        if np.abs(f)>self.tol:
            print(i,aL,uL,aR,uR,aS,np.abs(f))
            sys.exit('TwoRarsolveERP :: Newton algortihm did NOT convergence')
                
        fL = self.fK(aL, aS, K, a0, m, n)
        fR = self.fK(aR, aS, K, a0, m, n)
        
        # Compute uS
        uS = 0.5*(uL+uR) + 0.5*(fR-fL)
        
        return [aS, uS]
    
    def sampleInsideLeftRar(self, aL, uL, xt, K, a0, m, n):
        
        # INITIAL GUESS for arar
        arar = aL
        #GLOBALLY CONVERGENT NEWTON
        for i in range(self.nMax):
            crar = self.waveSpeed(arar, K, a0, m, n)
            intRIrar = self.IntegralRI(aL, arar, K, a0, m, n)
            
            fun = xt + crar + intRIrar - uL
            
            if np.abs(fun)<self.tol:
                break
            else:
                crarM = self.waveSpeed(arar-self.epsi, K, a0, m, n)
                intRIrarM = self.IntegralRI(aL, arar-self.epsi, K, a0, m, n)
                funM = xt + crarM + intRIrarM - uL
                
                crarP = self.waveSpeed(arar+self.epsi, K, a0, m, n)
                intRIrarP = self.IntegralRI(aL, arar+self.epsi, K, a0, m, n)
                funP = xt + crarP + intRIrarP - uL
                
                df = (funP - funM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = arar - alpha*fun/df
                    if aAux>0.:
                        crarAux = self.waveSpeed(aAux, K, a0, m, n)
                        intRIrarAux = self.IntegralRI(aL, aAux, K, a0, m, n)
                        fAux = xt + crarAux + intRIrarAux - uL
                        if np.abs(fAux)<np.abs(fun):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update arar
                arar = arar - alpha*fun/df
                
        if np.abs(fun)>self.tol:
            print(i,aL,uL,arar,np.abs(fun))
            sys.exit('sampleInsideLeftRar :: Newton algortihm did NOT convergence')
                
        crar = self.waveSpeed(arar, K, a0, m, n)
        urar = xt + crar
        
        return [arar, urar, crar]
                
    
    def sampleInsideRightRar(self, aR, uR, xt, K, a0, m, n):
        
        # INITIAL GUESS for arar
        arar = aR
        # GLOBALLY CONVERGENT NEWTON
        for i in range(self.nMax):
            crar = self.waveSpeed(arar, K, a0, m, n)
            intRIrar = self.IntegralRI(aR, arar, K, a0, m, n)
            
            fun = xt - crar - intRIrar - uR
            
            if np.abs(fun)<self.tol:
                break
            else:
                crarM = self.waveSpeed(arar-self.epsi, K, a0, m, n)
                intRIrarM = self.IntegralRI(aR, arar-self.epsi, K, a0, m, n)
                funM = xt - crarM - intRIrarM - uR
                
                crarP = self.waveSpeed(arar+self.epsi, K, a0, m, n)
                intRIrarP = self.IntegralRI(aR, arar+self.epsi, K, a0, m, n)
                funP = xt - crarP - intRIrarP - uR
                
                df = (funP - funM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = arar - alpha*fun/df
                    if aAux>0.:
                        crarAux = self.waveSpeed(aAux, K, a0, m, n)
                        intRIrarAux = self.IntegralRI(aR, aAux, K, a0, m, n)
                        fAux = xt - crarAux - intRIrarAux - uR
                        if np.abs(fAux)<np.abs(fun):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update arar
                arar = arar - alpha*fun/df
                
        if np.abs(fun)>self.tol:
            print(i,aR,uR,arar,np.abs(fun))
            sys.exit('sampleInsideRightRar :: Newton algortihm did NOT convergence')
                
        crar = self.waveSpeed(arar, K, a0, m, n)
        urar = xt - crar
        
        return [arar, urar, crar]
    
    # General Exact Riemann solver for arteries and veins 
    def solveERP(self, aL, aR, uL, uR, K, a0, m, n):
        """
        Solve the Star Problem
        """
        # INITIAL GUESS for aS in Newton-Raphson algorithm
        aS = 0.5*(aL + aR)
        # GLOBALLY CONVERGENT NEWTON
        # Compute aS
        for i in range(self.nMax):
            fL = self.fK(aL, aS, K, a0, m, n)
            fR = self.fK(aR, aS, K, a0, m, n)
        
            f = fL + fR + uR - uL
        
            if np.abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fL = self.fK(aL, aS+self.epsi, K, a0, m, n)
                fR = self.fK(aR, aS+self.epsi, K, a0, m, n)
                fP = fL + fR + uR - uL
        
                fL = self.fK(aL, aS-self.epsi, K, a0, m, n)
                fR = self.fK(aR, aS-self.epsi, K, a0, m, n)
                fM = fL + fR + uR - uL
        
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = aS - alpha*f/df
                    if aAux>0.:
                        fLAux = self.fK(aL, aAux, K, a0, m, n)
                        fRAux = self.fK(aR, aAux, K, a0, m, n)
                        fAux = fLAux + fRAux + uR - uL
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update aS
                aS = aS - alpha*f/df
        
        if np.abs(f)>self.tol:
            print(i,aL,uL,aR,uR,aS,np.abs(f))
            sys.exit('sampleInsideRightRar :: Newton algortihm did NOT convergence')
            
        fL = self.fK(aL, aS, K, a0, m, n)
        fR = self.fK(aR, aS, K, a0, m, n)
        
        # Compute uS
        uS = 0.5*(uL+uR) + 0.5*(fR-fL)
        
        return [aS, uS]
    
    # General sampling for arteries and veins
    def sampleERP(self, aL, aR, uL, uR, aS, uS, xt, K, a0, m, n):
        
        Q = np.zeros(self.nVar)
        
        cS = self.waveSpeed(aS, K, a0, m, n)
        cL = self.waveSpeed(aL, K, a0, m, n)
        cR = self.waveSpeed(aR, K, a0, m, n)
        
        if xt<uS: #LEFT OF CONTACT
            if aS<=aL: #LEFT RAREFACTION
                if xt<(uL-cL):
                    a = aL
                    u = uL
                elif xt<=(uS-cS):
                    arar, urar, crar = self.sampleInsideLeftRar(aL, uL, xt, K, a0, m, n)
                    a = arar
                    u = urar
                elif xt>(uS-cS):
                    a = aS
                    u = uS
                    
            else: #LEFT SHOCK
                mL = self.mK(aL, aS, K, a0, m, n)
                sL = uL - (mL/aL)
                
                if xt<sL:
                    a = aL
                    u = uL
                else:
                    a = aS
                    u = uS
                    
        else: #RIGHT OF CONTACT
            if aS<=aR: #RIGHT RAREFACTION
                if xt<(uS+cS):
                    a = aS
                    u = uS
                elif xt<=(uR+cR):
                    arar, urar, crar = self.sampleInsideRightRar(aR, uR, xt, K, a0, m, n)
                    a = arar
                    u = urar
                elif xt>(uR+cR):
                    a = aR
                    u = uR
                    
            else: #RIGHT SHOCK
                mR = self.mK(aR, aS, K, a0, m, n)
                sR = uR + (mR/aR)
                
                if xt<sR:
                    a = aS
                    u = uS
                else:
                    a = aR
                    u = uR
                    
        Q[0] = a
        Q[1] = a*u
        
        return Q
    
    # General exact Riemann problem for arteries and veins
    def ExactRP(self, aL, aR, uL, uR, aS, uS, x, time, gate, K, a0, m, n):
        
        #Initialize vector Q
        nCells = x.shape[0]
        Q = np.zeros((nCells, self.nVar))
    
        for i in range(nCells):
            xt = (x[i] - gate)/(time+1.0e-10)
            Q[i, :] = self.sampleERP(aL, aR, uL, uR, aS, uS, xt, K, a0, m, n)
            
        return Q
    
    # General GODUNOV STATE for arteries and veins
    def QStarExact(self, QL, QR, K, a0, m, n):
        
        # Define values
        aL = QL[0]
        aR = QR[0]
        uL = QL[1]/aL
        uR = QR[1]/aR
            
        # Find aS and uS
        [aS, uS] = self.solveERP(aL, aR, uL, uR, K, a0, m, n)
            
        # Sample the Godunov State: sampling of the solution along the characteristic x/t=0
        xt = 0. # t-axis --> x/t=0
        QS = self.sampleERP(aL, aR, uL, uR, aS, uS, xt, K, a0, m, n)
    
        return QS
    
    # GODUNOV STATE for the linearised system --> d_t Q + A * d_x Q = 0
    def QStarExactLinear(self, QL, QR, A):
        
        # Find eigenvalues and eigenvectors of matrix A 
        [lambda1, lambda2], [R1, R2] = np.linalg.eig(A) 
        
        n = len(QL)
        I = np.identity(n)
        
        R = np.zeros((n,n))
        R[0,:] = R1
        R[1,:] = R2
        invR = np.linalg.inv(R)
        
        D = np.zeros((n,n))
        D[0,0] = lambda1
        D[1,1] = lambda2
        
        xt=0.
        
        QS = 0.5*( R*(I + np.sign(D-xt*I))*invR )*QL + 0.5*( R*(I - np.sign(D-xt*I))*invR )*QR
        
        return QS
    
    #XXX FIRST-ORDER COUPLING - JUNCTIONS
    def Junction2fun(self, x, QL, QR, K, A0, m, n, P0):
        
        # define pressures
        pL = self.pFa(x[0], K[0], A0[0], m, n, P0)
        pR = self.pFa(x[2], K[1], A0[1], m, n, P0)
        
        # initialize junction function
        f = np.zeros(self.nVar*2)
        f[0] = x[0]*x[1] - x[2]*x[3] # conservation of mass
        f[1] = pL + 0.5*self.rho*x[1]**2. - pR - 0.5*self.rho*x[3]**2. # conservation of TOTAL PRESSURE
        f[2] = QL[1]/QL[0] - x[1] - self.IntegralRI(QL[0], x[0], K[0], A0[0], m, n) # continuity of GRI
        f[3] = QR[1]/QR[0] - x[3] + self.IntegralRI(QR[0], x[2], K[1], A0[1], m, n) # continuity of GRI
        
        return f
    
    
    def Junction2(self, QL, QR, K, A0, m, n, P0):
        
        aL = QL[0]
        uL = QL[1]/aL
        aR = QR[0]
        uR = QR[1]/aR
        
        # INITIAL GUESS
        x0 = np.array([aL, uL, aR, uR])
        
        # FUNCTION
        def fun(x, QL, QR, K, A0, m, n, P0):
            fun = self.Junction2fun(x, QL, QR, K, A0, m, n, P0)
            return fun
            
        xsol = fsolve(fun, x0, args=(QL, QR, K, A0, m, n, P0), xtol=1.0e-8)
        
        fsol = fun(xsol, QL, QR, K, A0, m, n, P0)
        if np.max(np.abs(fsol))>self.tol:
            print('******')
            print(x0)
            print(xsol)
            print(fsol)
            sys.exit('Junction2 :: fsolve did NOT convergence')
        
        QBR = np.zeros(self.nVar)
        QBL = np.zeros(self.nVar)
        
        QBR[0] = xsol[0]
        QBR[1] = xsol[1]*xsol[0]
        
        QBL[0] = xsol[2]
        QBL[1] = xsol[3]*xsol[2]
        
        return [QBR, QBL]
    
    def Junction3fun(self, x, QL, QR1, QR2, K, A0, m, n, P0):
        
        # define pressures
        pL = self.pFa(x[0], K[0], A0[0], m, n, P0)
        pR1 = self.pFa(x[2], K[1], A0[1], m, n, P0)
        pR2 = self.pFa(x[4], K[2], A0[2], m, n, P0)
        
        # initialize junction function
        f = np.zeros(self.nVar*3)
        f[0] = x[0]*x[1] - x[2]*x[3] - x[4]*x[5] # conservation of mass
        f[1] = pL + 0.5*self.rho*x[1]**2. - pR1 - 0.5*self.rho*x[3]**2. # conservation of TOTAL PRESSURE
        f[2] = pL + 0.5*self.rho*x[1]**2. - pR2 - 0.5*self.rho*x[5]**2. # conservation of TOTAL PRESSURE
        f[3] = QL[1]/QL[0] - x[1] - self.IntegralRI(QL[0], x[0], K[0], A0[0], m, n) # continuity of GRI
        f[4] = QR1[1]/QR1[0] - x[3] + self.IntegralRI(QR1[0], x[2], K[1], A0[1], m, n) # continuity of GRI
        f[5] = QR2[1]/QR2[0] - x[5] + self.IntegralRI(QR2[0], x[4], K[2], A0[2], m, n) # continuity of GRI
        
        return f
    
    def Junction3(self, QL, QR1, QR2, K, A0, m, n, P0):
        
        aL = QL[0]
        uL = QL[1]/aL
        aR1 = QR1[0]
        uR1 = QR1[1]/aR1
        aR2 = QR2[0]
        uR2 = QR2[1]/aR2
        
        # INITIAL GUESS
        x0 = np.array([aL, uL, aR1, uR1, aR2, uR2])
        
        # FUNCTION
        def fun(x, QL, QR1, QR2, K, A0, m, n, P0):
            fun = self.Junction3fun(x, QL, QR1, QR2, K, A0, m, n, P0)
            return fun
        
        xsol = fsolve(fun, x0, args=(QL, QR1, QR2, K, A0, m, n, P0), xtol=1.0e-8)
        
        fsol = fun(xsol, QL, QR1, QR2, K, A0, m, n, P0)
        if np.max(np.abs(fsol))>self.tol:
            print('******')
            print(x0)
            print(xsol)
            print(fsol)
            sys.exit('Junction3 :: fsolve did NOT convergence')
           
        QBR = np.zeros(self.nVar)
        QBL1 = np.zeros(self.nVar)
        QBL2 = np.zeros(self.nVar)
        
        QBR[0] = xsol[0]
        QBR[1] = xsol[1]*xsol[0]
        
        QBL1[0] = xsol[2]
        QBL1[1] = xsol[3]*xsol[2]
        
        QBL2[0] = xsol[4]
        QBL2[1] = xsol[5]*xsol[4]
        
        return [QBR, QBL1, QBL2]
    
    
    def JunctionNfun(self, x, N, Q1D, g, K, A0, m, n, P0):
        """
        Generic 1D junction
        
        Parameters
        ----------
        x : (column) vector of the 2N unknown star values (N star states) --> x = [AS1, AS2, ..., ASN, uS1, uS2, ..., uSN]^T
        N : number of 1D vessels sharing a vertex at the junction
        Q1D : list/matrix of N vectors containing the 1D state from each 1D vessel converging at the junction, i.e.
              Q1D[k]=[A1Dk, q1Dk], that is the 1D solution in the last cell of k-th vessel if k-th vessel shares its outlet vertex,
              or the 1D solution in the first cell of k-th vessel if k-th vessel shares its inlet vertex
        g : auxiliary function s.t. g[k]=+1 if k-th vessel shares its outlet vertex (last cell) at the junction,
                                    g[k]=-1 if k-th vessel shares its inlet vertex (first cell) at the junction
        """
        
        if not isinstance(m, (list, tuple, np.ndarray)):
            mVal = m
            m = np.zeros(N)
            m[:] = mVal
        if not isinstance(n, (list, tuple, np.ndarray)):
            nVal = n
            n = np.zeros(N)
            n[:] = nVal
        if not isinstance(P0, (list, tuple, np.ndarray)):
            P0Val = P0
            P0 = np.zeros(N)
            P0[:] = P0Val
        
        # define pressures
        pStar = np.zeros(N)
        pStarTot = np.zeros(N)
        for i in range(N):
            pStar[i] = self.pFa(x[i,0], K[i], A0[i], m[i], n[i], P0[i])
            pStarTot[i] = pStar[i] + 0.5*self.rho*x[N+i,0]**2.
        
        # initialize junction function
        f = np.zeros_like(x) # column vector
        
        for i in range(N): # conservation of mass
            f[0,0] = f[0,0] + g[i]*x[i,0]*x[N+i,0] 
        
        for i in range(1,N): # continuity of TOTAL PRESSURE
            f[i,0] = pStarTot[0] - pStarTot[i]
            
        for i in range(N): # continuity of GRI
            a1D = Q1D[i,0]
            q1D = Q1D[i,1]
            u1D = q1D/a1D
            f[N+i,0] = x[N+i,0] - u1D + g[i]*self.IntegralRI(a1D, x[i,0], K[i], A0[i], m[i], n[i])   

        return f
    
    def JunctionN(self, N, Q1D, g, K, A0, m, n, P0):
        
        Q1Dsol = np.zeros((N, self.nVar), dtype=np.double)
        
        # INITIAL GUESS
        x0 = np.zeros((N*self.nVar, 1), dtype=np.double) # column vector
        
        for i in range(N):
            a1D = Q1D[i][0]
            q1D = Q1D[i][1]
            u1D = q1D/a1D
            Q1Dsol[i,0] = a1D
            Q1Dsol[i,1] = q1D
            x0[i,0] = a1D
            x0[N+i,0] = u1D
            
        # FUNCTION
        fun = lambda x : self.JunctionNfun(x, N, Q1Dsol, g, K, A0, m, n, P0)
        
        # NONLINEAR SOLVER PARAMETERS
        tol = 1.0e-10
        tolMass = 1.0e-8
        nmax = 100
        
        # NEWTON METHOD 
        xsol = self.NewtonNLSystem(fun, x0, A0, tol, tolMass, nmax) 
        # GLOBALLY CONVERGENT NEWTON METHOD
        # xsol = self.GlobConvNewtonNLSystem(fun, x0, A0, tol, tolMass, nmax) 
        
        # fsol = fun(xsol)
        
        QBC = np.zeros((N, self.nVar), dtype=np.double)
        for i in range(N):
            QBC[i,0] = xsol[i,0]
            QBC[i,1] = xsol[i,0]*xsol[N+i,0]
        
        return QBC
    
    #XXX HIGH-ORDER COUPLING - SECOND-ORDER JUNCTIONS
    def Junction2_o2(self, dx, dt, QL, QR, K, A0, m, n, P0, mu):
        
        # STENCIL for ENO reconstruction for the left (parent) vessel
        SL = [ QL[0], QL[1], QL[1]*10000. ] # i=M-1, i=M, i=M *10000
        dQL = self.slope_single(SL, dx[0])
        
        # STENCIL for ENO reconstruction for the right (daughter) vessel
        SR = [ QR[0]*10000., QR[0], QR[1] ] # i=1 *10000, i=1, i=2
        dQR = self.slope_single(SR, dx[1])
                
        QLL1 = QL[1] - 0.5*dx[0]*dQL
        QLL2 = QL[1] + 0.5*dx[0]*dQL
        QLLbar = QLL2 - 0.5*dt/dx[0]*( self.pF(QLL2, K[0], A0[0], m, n) - self.pF(QLL1, K[0], A0[0], m, n) ) + 0.5*dt*self.S(QLL2, mu[0])
        
        QRR1 = QR[0] - 0.5*dx[1]*dQR
        QRR2 = QR[0] + 0.5*dx[1]*dQR
        QRRbar = QRR1 - 0.5*dt/dx[1]*( self.pF(QRR2, K[1], A0[1], m, n) - self.pF(QRR1, K[1], A0[1], m, n) ) + 0.5*dt*self.S(QRR1, mu[1])
        
        Q1DL = np.copy(QLLbar)
        Q1DR = np.copy(QRRbar)
        
        aL = Q1DL[0]
        uL = Q1DL[1]/aL
        aR = Q1DR[0]
        uR = Q1DR[1]/aR
        
        # INITIAL GUESS
        x0 = np.array([aL, uL, aR, uR])
        
        # FUNCTION
        def fun(x, Q1DL, Q1DR, K, A0, m, n, P0):
            fun = self.Junction2fun(x, Q1DL, Q1DR, K, A0, m, n, P0)
            return fun
        
        xsol = fsolve(fun, x0, args=(Q1DL, Q1DR, K, A0, m, n, P0), xtol=1.0e-8)
        
        fsol = fun(xsol, Q1DL, Q1DR, K, A0, m, n, P0)
        if np.max(np.abs(fsol))>self.tol:
            print('******')
            print(x0)
            print(xsol)
            print(fsol)
            sys.exit('Junction2_o2 :: fsolve did NOT convergence')
        
        QBR = np.zeros(self.nVar)
        QBL = np.zeros(self.nVar)
        
        QBR[0] = xsol[0]
        QBR[1] = xsol[1]*xsol[0]
        
        QBL[0] = xsol[2]
        QBL[1] = xsol[3]*xsol[2]
        
        return [QBR, QBL]
    
    def Junction3_o2(self, dx, dt, QL, QR1, QR2, K, A0, m, n, P0, mu):
        
        # STENCIL for ENO reconstruction for the left (parent) vessel
        SL = [ QL[0], QL[1], QL[1]*10000. ] # i=M-1, i=M, i=M *10000
        dQL = self.slope_single(SL, dx[0])
        
        # STENCIL for ENO reconstruction for the first right (daughter 1) vessel
        SR1 = [ QR1[0]*10000., QR1[0], QR1[1] ] # i=1 *10000, i=1, i=2
        dQR1 = self.slope_single(SR1, dx[1])
        
        # STENCIL for ENO reconstruction for the second right (daughter 2) vessel
        SR2 = [ QR2[0]*10000., QR2[0], QR2[1] ] # i=1 *10000, i=1, i=2
        dQR2 = self.slope_single(SR2, dx[2])
        
        QLL1 = QL[1] - 0.5*dx[0]*dQL
        QLL2 = QL[1] + 0.5*dx[0]*dQL
        QLLbar = QLL2 - 0.5*dt/dx[0]*( self.pF(QLL2, K[0], A0[0], m, n) - self.pF(QLL1, K[0], A0[0], m, n) ) + 0.5*dt*self.S(QLL2, mu[0])
        
        QRR1 = QR1[0] - 0.5*dx[1]*dQR1
        QRR2 = QR1[0] + 0.5*dx[1]*dQR1
        QRRbar1 = QRR1 - 0.5*dt/dx[1]*( self.pF(QRR2, K[1], A0[1], m, n) - self.pF(QRR1, K[1], A0[1], m, n) ) + 0.5*dt*self.S(QRR1, mu[1])
        
        QRR1 = QR2[0] - 0.5*dx[2]*dQR2
        QRR2 = QR2[0] + 0.5*dx[2]*dQR2
        QRRbar2 = QRR1 - 0.5*dt/dx[2]*( self.pF(QRR2, K[2], A0[2], m, n) - self.pF(QRR1, K[2], A0[2], m, n) ) + 0.5*dt*self.S(QRR1, mu[2])
        
        Q1DL = np.copy(QLLbar)
        Q1DR1 = np.copy(QRRbar1)
        Q1DR2 = np.copy(QRRbar2)
        
        aL = Q1DL[0]
        uL = Q1DL[1]/aL
        aR1 = Q1DR1[0]
        uR1 = Q1DR1[1]/aR1
        aR2 = Q1DR2[0]
        uR2 = Q1DR2[1]/aR2
        
        # INITIAL GUESS
        x0 = np.array([aL, uL, aR1, uR1, aR2, uR2])
        
        # FUNCTION
        def fun(x, Q1DL, Q1DR1, Q1DR2, K, A0, m, n, P0):
            fun = self.Junction3fun(x, Q1DL, Q1DR1, Q1DR2, K, A0, m, n, P0)
            return fun
        
        xsol = fsolve(fun, x0, args=(Q1DL, Q1DR1, Q1DR2, K, A0, m, n, P0), xtol=1.0e-8)
        
        fsol = fun(xsol, Q1DL, Q1DR1, Q1DR2, K, A0, m, n, P0)
        if np.max(np.abs(fsol))>self.tol:
            print('******')
            print(x0)
            print(xsol)
            print(fsol)
            sys.exit('Junction3_o2 :: fsolve did NOT convergence')
             
        QBR = np.zeros(self.nVar)
        QBL1 = np.zeros(self.nVar)
        QBL2 = np.zeros(self.nVar)
        
        QBR[0] = xsol[0]
        QBR[1] = xsol[1]*xsol[0]
        
        QBL1[0] = xsol[2]
        QBL1[1] = xsol[3]*xsol[2]
        
        QBL2[0] = xsol[4]
        QBL2[1] = xsol[5]*xsol[4]
        
        return [QBR, QBL1, QBL2]
    
    def JunctionN_o2(self, dx, dt, N, Q1D, g, K, A0, m, n, P0, mu):
        
        Q1Dbar = np.zeros((N, self.nVar), dtype=np.double)
        
        # INITIAL GUESS
        x0 = np.zeros((N*self.nVar, 1), dtype=np.double) # column vector

        if not isinstance(m, (list, tuple, np.ndarray)):
            mVal = m
            m = np.zeros(N)
            m[:] = mVal
        if not isinstance(n, (list, tuple, np.ndarray)):
            nVal = n
            n = np.zeros(N)
            n[:] = nVal
        if not isinstance(P0, (list, tuple, np.ndarray)):
            P0Val = P0
            P0 = np.zeros(N)
            P0[:] = P0Val
        
        for i in range(N):
            if g[i]==1:
                # STENCIL for ENO reconstruction
                Seno = [ Q1D[i][0], Q1D[i][1], Q1D[i][1]*10000. ] # i=M-1, i=M, i=M *10000
                dQeno = self.slope_single(Seno, dx[i])
                QL1 = Q1D[i][1] - 0.5*dx[i]*dQeno
                QL2 = Q1D[i][1] + 0.5*dx[i]*dQeno
                QLbar = QL2 - 0.5*dt/dx[i]*( self.pF(QL2, K[i], A0[i], m[i], n[i]) - self.pF(QL1, K[i], A0[i], m[i], n[i]) ) + 0.5*dt*self.S(QL2, mu[i])
                Q1Dbar[i,:] = np.copy(QLbar)
                a1D = QLbar[0] 
                q1D = QLbar[1] 
                u1D = q1D/a1D
                x0[i,0] = a1D
                x0[N+i,0] = u1D
            elif g[i]==-1:
                # STENCIL for ENO reconstruction
                Seno = [ Q1D[i][0]*10000., Q1D[i][0], Q1D[i][1] ] # i=1 *10000, i=1, i=2
                dQeno = self.slope_single(Seno, dx[i])
                QR1 = Q1D[i][0] - 0.5*dx[i]*dQeno
                QR2 = Q1D[i][0] + 0.5*dx[i]*dQeno
                QRbar = QR1 - 0.5*dt/dx[i]*( self.pF(QR2, K[i], A0[i], m[i], n[i]) - self.pF(QR1, K[i], A0[i], m[i], n[i]) ) + 0.5*dt*self.S(QR1, mu[i])
                Q1Dbar[i,:] = np.copy(QRbar)
                a1D = QRbar[0] 
                q1D = QRbar[1] 
                u1D = q1D/a1D
                x0[i,0] = a1D
                x0[N+i,0] = u1D
        
        # FUNCTION
        fun = lambda x : self.JunctionNfun(x, N, Q1Dbar, g, K, A0, m, n, P0)
        
        # NONLINEAR SOLVER PARAMETERS
        tol = 1.0e-10
        tolMass = 1.0e-8
        nmax = 100
        
        # choose NEWTON METHOD or GLOBALLY CONVERGENT NEWTON METHOD
        xsol = self.NewtonNLSystem(fun, x0, A0, tol, tolMass, nmax) 
        # xsol = self.GlobConvNewtonNLSystem(fun, x0, A0, tol, tolMass, nmax) 
        
        # fsol = fun(xsol)
        
        QBC = np.zeros((N, self.nVar), dtype=np.double)
        for i in range(N):
            QBC[i,0] = xsol[i,0]
            QBC[i,1] = xsol[i,0]*xsol[N+i,0]
        
        return QBC
    
    
    #XXX FIRST-ORDER COUPLING - TERMINALS
    # def Terminal1DRCR(self, dt, Q1D, Pwk, Rw1, Cw, Rw2, Pout, K, a0, m, n, P0):
    def Terminal1DRCR(self, Q1D, Pwk, Rw1, Cw, Rw2, Pout, K, a0, m, n, P0):
        
        AL = Q1D[0]
        QL = Q1D[1]
        
        # INITIAL GUESS
        AS = AL
        # GLOBALLY CONVERGENT NEWTON   
        for i in range(self.nMax):
            PS = self.pFa(AS, K, a0, m, n, P0)
            QS = (PS-Pwk)/Rw1
            fun = QS/AS - QL/AL + self.IntegralRI(AL, AS, K, a0, m, n)
                
            if np.abs(fun)<self.tol:
                break
            else:
                # Evaluate the derivative
                PSM = self.pFa(AS-self.epsi, K, a0, m, n, P0)
                QSM = (PSM-Pwk)/Rw1
                funM = QSM/(AS-self.epsi) - QL/AL + self.IntegralRI(AL, AS-self.epsi, K, a0, m, n)
            
                PSP = self.pFa(AS+self.epsi, K, a0, m, n, P0)
                QSP = (PSP-Pwk)/Rw1
                funP = QSP/(AS+self.epsi) - QL/AL + self.IntegralRI(AL, AS+self.epsi, K, a0, m, n)
            
                df = (funP - funM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    AAux = AS - alpha*fun/df
                    if AAux>0.:
                        PSAux = self.pFa(AAux, K, a0, m, n, P0)
                        QSAux = (PSAux-Pwk)/Rw1
                        fAux = QSAux/AAux - QL/AL + self.IntegralRI(AL, AAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(fun):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # update Astar
                AS = AS - alpha*fun/df
                
        if np.abs(fun)>self.tol:
            print(i,AL,QL/AL,AS,np.abs(fun))
            sys.exit('Terminal1DRCR :: Newton algortihm did NOT convergence')
        
        PS = self.pFa(AS, K, a0, m, n, P0)
        QS = (PS-Pwk)/Rw1
        
        QStar = np.zeros(self.nVar)
        QStar[0] = AS
        QStar[1] = QS
        # QStar = [AS, QS]
        
        # Explicit Euler scheme to evolve pressure Pwk in the Windkessel
        # PwkNew = Pwk + dt/Cw*( QS - (Pwk-Pout)/Rw2 )
        
        # return [ QStar, PwkNew ]
        return QStar
    
    def Terminal1DsingleR(self, Q1D, Rt, Pout, K, a0, m, n, P0):
        
        AL = Q1D[0]
        QL = Q1D[1]
        
        # # INITIAL GUESS
        # x0 = AL
        # # FUNCTION
        # def fun(x, AL, QL, Rt, Pout, K, a0, m, n, P0):
        #     PS = self.pFa(x, K, a0, m, n, P0)
        #     QS = (PS-Pout)/Rt
        #     fun = QS/x - QL/AL + self.IntegralRI(AL, x, K, a0, m, n)
        #     return fun
        
        # xsol = fsolve(fun, x0, args=(AL, QL, Rt, Pout, K, a0, m, n, P0))
        
        # fsol = fun(xsol, AL, QL, Rt, Pout, K, a0, m, n, P0)
        # if abs(fsol)>self.tol:
        #     sys.exit('NO convergence')
            
        # AS = xsol
        
        # INITIAL GUESS
        AS = AL
        # GLOBALLY CONVERGENT NEWTON 
        for i in range(self.nMax):
            PS = self.pFa(AS, K, a0, m, n, P0)
            QS = (PS-Pout)/Rt
            fun = QS/AS - QL/AL + self.IntegralRI(AL, AS, K, a0, m, n)
                
            if np.abs(fun)<self.tol:
                break
            else:
                # Evaluate the derivative
                PSM = self.pFa(AS-self.epsi, K, a0, m, n, P0)
                QSM = (PSM-Pout)/Rt
                funM = QSM/(AS-self.epsi) - QL/AL + self.IntegralRI(AL, AS-self.epsi, K, a0, m, n)
            
                PSP = self.pFa(AS+self.epsi, K, a0, m, n, P0)
                QSP = (PSP-Pout)/Rt
                funP = QSP/(AS+self.epsi) - QL/AL + self.IntegralRI(AL, AS+self.epsi, K, a0, m, n)
            
                df = (funP - funM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    AAux = AS - alpha*fun/df
                    if AAux>0.:
                        PSAux = self.pFa(AAux, K, a0, m, n, P0)
                        QSAux = (PSAux-Pout)/Rt
                        fAux = QSAux/AAux - QL/AL + self.IntegralRI(AL, AAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(fun):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # update Astar
                AS = AS - alpha*fun/df
                
        if np.abs(fun)>self.tol:
            print(i,AL,QL/AL,AS,np.abs(fun))
            sys.exit('Terminal1DsingleR :: Newton algortihm did NOT convergence')
        
        PS = self.pFa(AS, K, a0, m, n, P0)
        QS = (PS-Pout)/Rt
        
        QStar = np.zeros(self.nVar)
        QStar[0] = AS
        QStar[1] = QS
        # QStar = [AS, QS]
        
        return QStar
    
    #XXX HIGH-ORDER COUPLING - SECOND-ORDER TERMINALS
    def Terminal1DRCR_o2(self, dx, dt, Q1D, Pwk, Rw1, Cw, Rw2, Pout, K, a0, m, n, P0, mu):
        
        # STENCIL for ENO reconstruction for the terminal vessel
        S = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
        dQ = self.slope_single(S, dx)
        
        QL = Q1D[1] - 0.5*dx*dQ
        QR = Q1D[1] + 0.5*dx*dQ
        QRbar = QR - 0.5*dt/dx*( self.pF(QR, K, a0, m, n) - self.pF(QL, K, a0, m, n) ) + 0.5*dt*self.S(QR, mu)
        
        aL = QRbar[0]
        qL = QRbar[1]
        
        # INITIAL GUESS
        AS = aL
        # GLOBALLY CONVERGENT NEWTON   
        for i in range(self.nMax):
            PS = self.pFa(AS, K, a0, m, n, P0)
            QS = (PS-Pwk)/Rw1
            fun = QS/AS - qL/aL + self.IntegralRI(aL, AS, K, a0, m, n)
                
            if np.abs(fun)<self.tol:
                break
            else:
                # Evaluate the derivative
                PSM = self.pFa(AS-self.epsi, K, a0, m, n, P0)
                QSM = (PSM-Pwk)/Rw1
                funM = QSM/(AS-self.epsi) - qL/aL + self.IntegralRI(aL, AS-self.epsi, K, a0, m, n)
            
                PSP = self.pFa(AS+self.epsi, K, a0, m, n, P0)
                QSP = (PSP-Pwk)/Rw1
                funP = QSP/(AS+self.epsi) - qL/aL + self.IntegralRI(aL, AS+self.epsi, K, a0, m, n)
            
                df = (funP - funM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    AAux = AS - alpha*fun/df
                    if AAux>0.:
                        PSAux = self.pFa(AAux, K, a0, m, n, P0)
                        QSAux = (PSAux-Pwk)/Rw1
                        fAux = QSAux/AAux - qL/aL + self.IntegralRI(aL, AAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(fun):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # update Astar
                AS = AS - alpha*fun/df
                
        if np.abs(fun)>self.tol:
            print(i,aL,qL/aL,AS,np.abs(fun))
            sys.exit('Terminal1DRCR_o2 :: Newton algortihm did NOT convergence')
        
        PS = self.pFa(AS, K, a0, m, n, P0)
        QS = (PS-Pwk)/Rw1
        
        QStar = np.zeros(self.nVar)
        QStar[0] = AS
        QStar[1] = QS
        # QStar = [AS, QS]
        
        # Explicit Euler scheme to evolve pressure Pwk in the Windkessel
        PwkNew = Pwk + dt/Cw*( QS - (Pwk-Pout)/Rw2 )
        
        return [ QStar, PwkNew ]
    
    def Terminal1DsingleR_o2(self, dx, dt, Q1D, Rt, Pout, K, a0, m, n, P0, mu):
        
        # STENCIL for ENO reconstruction for the terminal vessel
        S = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
        dQ = self.slope_single(S, dx)
        
        QL = Q1D[1] - 0.5*dx*dQ
        QR = Q1D[1] + 0.5*dx*dQ
        QRbar = QR - 0.5*dt/dx*( self.pF(QR, K, a0, m, n) - self.pF(QL, K, a0, m, n) ) + 0.5*dt*self.S(QR, mu)
        
        aL = QRbar[0]
        qL = QRbar[1]
        
        # # INITIAL GUESS
        # x0 = aL
        
        # # FUNCTION
        # def fun(x, aL, qL, Rt, Pout, K, a0, m, n, P0):
        #     PS = self.pFa(x, K, a0, m, n, P0)
        #     QS = (PS-Pout)/Rt
        #     fun = QS/x - qL/aL + self.IntegralRI(aL, x, K, a0, m, n)
        #     return fun
        
        # xsol = fsolve(fun, x0, args=(aL, qL, Rt, Pout, K, a0, m, n, P0))
        
        # AS = xsol
            
        # INITIAL GUESS
        AS = aL
        # GLOBALLY CONVERGENT NEWTON 
        for i in range(self.nMax):
            PS = self.pFa(AS, K, a0, m, n, P0)
            QS = (PS-Pout)/Rt
            fun = QS/AS - qL/aL + self.IntegralRI(aL, AS, K, a0, m, n)
                
            if np.abs(fun)<self.tol:
                break
            else:
                # Evaluate the derivative
                PSM = self.pFa(AS-self.epsi, K, a0, m, n, P0)
                QSM = (PSM-Pout)/Rt
                funM = QSM/(AS-self.epsi) - qL/aL + self.IntegralRI(aL, AS-self.epsi, K, a0, m, n)
            
                PSP = self.pFa(AS+self.epsi, K, a0, m, n, P0)
                QSP = (PSP-Pout)/Rt
                funP = QSP/(AS+self.epsi) - qL/aL + self.IntegralRI(aL, AS+self.epsi, K, a0, m, n)
            
                df = (funP - funM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    AAux = AS - alpha*fun/df
                    if AAux>0.:
                        PSAux = self.pFa(AAux, K, a0, m, n, P0)
                        QSAux = (PSAux-Pout)/Rt
                        fAux = QSAux/AAux - qL/aL + self.IntegralRI(aL, AAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(fun):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # update Astar
                AS = AS - alpha*fun/df
                
        if np.abs(fun)>self.tol:
            print(i,aL,qL/aL,AS,np.abs(fun))
            sys.exit('Terminal1DsingleR_o2 :: Newton algortihm did NOT convergence')
        
        PS = self.pFa(AS, K, a0, m, n, P0)
        QS = (PS-Pout)/Rt
        
        QStar = np.zeros(self.nVar)
        QStar[0] = AS
        QStar[1] = QS
        # QStar = [AS, QS]
        
        return QStar
    
    
    # SOLUTION OF NONLINEAR SYSTEMS
    # JACOBIAN MATRIX CALCULATION/APPROXIMATION
    def jacobianNumFD(self, fun, x, h=1.0e-12):
        """
        Approximation of Jacobian matrix Jf(x) of function f(x) by finite differences
        """
        
        N = x.shape[0] # x is a column vector
        jac = np.zeros((N,N))
        
        # f0 = fun(x)
        
        for j in range(N): 
            xtmp = x[j,0]
            
            x[j,0] = xtmp + h
            fP = fun(x)
            
            # forward difference
            # x[j,0] = xtmp
            # jac[:,j] = (fP.T - f0.T)/h
            
            # central difference
            x[j,0] = xtmp - h
            fM = fun(x)
            jac[:,j] = (fP.T - fM.T)/(2.*h)
            x[j,0] = xtmp
        
        return jac 
    
    # NEWTON ALGORITHM 
    def NewtonNLSystem(self, f, xIG, A0, tol, tolMass, nmax):
        """
        Newton algorithm for nonlinear systems
        Inputs:
        - f : nonlinear vector function f(x) for which we want to find the solution vector x* s.t. f(x*)=0
        - xIG : numpy.array (column) vector of the initial guess for the iterative procedure
        - tol : tolerance for stopping criterium
        - nmax : maximum number of iteration
        Outputs:
        - x : numpy.array (column) vector of the solution x*
        - i : number of iterations performed
        - L2err : list of L2-norm error for all iterations
        - xFull : list of x values for all iterations
        """
        
        # print('NewtonNLSystem STARTS')
        
        N = int(xIG.shape[0]/2)
        
        x = np.copy(xIG)
        
        # Jfk = self.jacobianNumFD(f,x)
        # iJfk = np.linalg.inv(Jfk)
        
        # L2err = []
        # xFull = []
        
        for i in range(nmax):
            
            # xFull.append( np.copy(x) )
            
            fk = f(x)
            Jfk = self.jacobianNumFD(f,x)
            iJfk = np.linalg.inv(Jfk)
            
            dx = -1.*np.dot(iJfk,fk)
            # x = x + dx
            
            conv = 1
            for p in range(N):
                if np.abs(dx[p,0])/A0[p] > tol:
                    conv = 0
                if np.abs(dx[N+p,0])/1.0 > tol:
                    conv = 0
            
            if np.abs(fk[0,0]) > tolMass:
                conv = 0

            if conv==1:
                break
            
            x = x + dx

            # L2errk = np.linalg.norm(dx,2)
            # # L2err.append( L2errk )
            # if L2errk < tol:
            #     break
        
        # print(i)
        # print(xIG)
        # print(x)
        # print(np.any(x[:N]<=0.))
        # print(x[:N]*x[N:])
        # print(dx)
        # print(f(x))
        # print(tol, tolMass)
        
        # convergence check
        if i>=nmax:
            print('Maximum number of iteration reached : ', i, nmax)
            print(xIG)
            print(x)
            print(dx)
            print(f(x))
            sys.exit('JunctionN :: NewtonNLSystem did NOT convergence')
        if np.any(x[:N]<=0.):
            print('Junction problem solution with area <=0 : ', x[:N])
            print(A0)
            print(xIG)
            print(x)
            print(dx)
            print(f(x))
            sys.exit('JunctionN :: NewtonNLSystem returned area <=0')
            
        # print('NewtonNLSystem ENDS')
            
        # return [x, i, L2err, xFull]
        return x
    
    # GLOBALLY CONVERGENT NEWTON ALGORITHM
    def GlobConvNewtonNLSystem(self, f, xIG, A0, tol, tolMass, nmax):
        """
        Globally Convergent Newton algorithm for nonlinear systems
        Inputs:
        - f : nonlinear vector function f(x) for which we want to find the solution vector x* s.t. f(x*)=0
        - xIG : numpy.array (column) vector of the initial guess for the iterative procedure
        - tol : tolerance for stopping criterium
        - nmax : maximum number of iteration
        Outputs:
        - x : numpy.array (column) vector of the solution x*
        - i : number of iterations performed
        - L2err : list of L2-norm error for all iterations
        - xFull : list of x values for all iterations
        """
        
        # print('Inside GlobConvNewtonNLSystem')

        N = int(xIG.shape[0]/2)

        x = np.copy(xIG)
        
        # Jfk = self.jacobianNumFD(f,x)
        # iJfk = np.linalg.inv(Jfk)
        
        # L2err = []
        # xFull = []
        
        for i in range(nmax):
            
            # xFull.append( np.copy(x) )
            
            fk = f(x)
            Jfk = self.jacobianNumFD(f,x)
            iJfk = np.linalg.inv(Jfk)
            
            dx = -1.*np.dot(iJfk,fk)
            
            alpha = 1.
            for j in range(nmax):
                xAux = x + alpha*dx
                fAux = f(xAux)
                
                if np.linalg.norm(fAux,np.inf)>np.linalg.norm(fk,np.inf):
                    alpha *= 0.8
                else:
                    break
                
            conv = 1
            for p in range(N):
                if np.abs(alpha*dx[p,0])/A0[p] > tol:
                    conv = 0
                if np.abs(alpha*dx[N+p,0])/1.0 > tol:
                    conv = 0
            
            if np.abs(fk[0,0]) > tolMass:
                conv = 0

            if conv==1:
                break
    
            x = x + alpha*dx
            
            # L2errk = np.linalg.norm(dx,2)
            # # L2err.append( L2errk )
            # if L2errk < tol:
            #     break
        
        # print('Number of iterations:', i)
        
        # convergence check
        if i>=nmax:
            print('Maximum number of iteration reached : ', i, nmax)
            print(xIG)
            print(x)
            print(dx)
            print(f(x))
            sys.exit('JunctionN :: GlobConvNewtonNLSystem did NOT convergence')
        if np.any(x[:N]<=0.):
            print('Junction problem solution with area <=0 : ', x[:N])
            print(A0)
            print(xIG)
            print(x)
            print(dx)
            print(f(x))
            sys.exit('JunctionN :: GlobConvNewtonNLSystem returned area <=0')
            
        # return [x, i, L2err, xFull]
        return x


    # SLOPES FOR SECOND-ORDER RECONSTRUCTION
    def slope_single(self, St, dx):
        StL = St[0]
        StC = St[1]
        StR = St[2]
        
        DL = (StC-StL)/dx
        DR = (StR-StC)/dx
        
        dQ = np.zeros(self.nVar)
        for k in range(self.nVar):
            
            if self.slopeTy=='ENO':
                if np.abs(DR[k])<np.abs(DL[k]):
                    dQ[k] = DR[k]
                else:
                    dQ[k] = DL[k]
            
            elif self.slopeTy=='1stOrder':
                dQ[k] = 0.
                
            elif self.slopeTy=='Fromm':
                dQ[k] = 0.5*(DR[k]+DL[k])
                
            elif self.slopeTy=='minmod':
                if np.abs(DL[k])<=np.abs(DR[k]) and DL[k]*DR[k]>0.:
                    dQ[k] = DL[k]
                elif np.abs(DL[k])<np.abs(DR[k]) and DL[k]*DR[k]>0.:
                    dQ[k] = DR[k]
                elif DL[k]*DR[k]<0.:
                    dQ[k] = 0.
        
        return dQ
    
    def BoundaryExtrapValues_single(self, St, dx, g):
        
        dQ = self.slope_single(St, dx)
        QL = St[1] - 0.5*dx*dQ
        QR = St[1] + 0.5*dx*dQ
        
        if g==-1:
            Qrec = np.copy(QL)
        elif g==1:
            Qrec = np.copy(QR)
        
        return Qrec
    

    ########################################
    ########################################


    def JunctionNwConn0Dfun(self, x, N, Q1D, g1D, Q0D, g0D, K, A0, m, n, P0, couplePtot=1):
        """
        Nonlinear function for hybrid junction with N 1D vessels and a 0D element
        
        Parameters
        ----------
        x : (column) vector of the 2N unknown star values (N star states) --> x = [AS1, AS2, ..., ASN, uS1, uS2, ..., uSN]
        N : number of 1D vessels
        Q1D : list/matrix of N vectors containing the 1D state from each 1D vessel converging at the junction, i.e.
              Q1D[k]=[A1Dk, q1Dk], that is the 1D solution in the last cell of k-th vessel if k-th vessel shares its outlet vertex,
              or the 1D solution in the first cell of k-th vessel if k-th vessel shares its inlet vertex
        g1D : auxiliary function for 1D vessels, s.t. 
                g[k]=+1 if k-th 1D vessel shares its outlet vertex (last cell) at the junction,
                g[k]=-1 if k-th 1D vessel shares its inlet vertex (first cell) at the junction
        Q0D : the 0D flow state variable from the 0D element converging at the junction
        g0D : auxiliary function for 0D element, s.t.
                g0D=+1 if element shares its outlet at the junction,
                g0D=-1 if element shares its inlet at the junction
        """

        if not isinstance(m, (list, tuple, np.ndarray)):
            mVal = m
            m = np.zeros(N)
            m[:] = mVal
        if not isinstance(n, (list, tuple, np.ndarray)):
            nVal = n
            n = np.zeros(N)
            n[:] = nVal
        if not isinstance(P0, (list, tuple, np.ndarray)):
            P0Val = P0
            P0 = np.zeros(N)
            P0[:] = P0Val

        # initialize junction function
        f = np.zeros_like(x) # column vector

        if N==1:
            # modified conservation of mass, to take into account the flow rate in/out the 0D sink
            f[0,0] = g1D*x[0,0]*x[1,0] + g0D*Q0D 

            a1D = Q1D[0,0]
            q1D = Q1D[0,1]
            u1D = q1D/a1D
            f[1,0] = x[1,0] - u1D + g1D*self.IntegralRI(a1D, x[0,0], K[i], A0[i], m[i], n[i]) # Q1D[i,2], Q1D[i,3], m, n
 
            # pStarTot = self.pFa(x[0,0], Q1D[0,2], Q1D[0,3], m, n, P0, Q1D[0,4])
            # if couplePtot==1:
            #     pStarTot = pStarTot + 0.5*self.rho*x[1,0]**2.

        else:
            # define pressures
            pStarTot = np.zeros(N)
            for i in range(N):
                # pStarTot[i] = self.pFa(x[i,0], Q1D[i,2], Q1D[i,3], m, n, P0, Q1D[i,4])
                pStarTot[i] = self.pFa(x[i,0], K[i], A0[i], m[i], n[i], P0[i])
                if couplePtot==1:
                    pStarTot[i] = pStarTot[i] + 0.5*self.rho*x[N+i,0]**2.
            
            # modified conservation of mass, to take into account the flow rate in/out the 0D sink
            for i in range(N): 
                f[0,0] = f[0,0] + g1D[i]*x[i,0]*x[N+i,0] 
            f[0,0] = f[0,0] + g0D*Q0D
        
            for i in range(1,N): # continuity of TOTAL PRESSURE / PRESSURE
                f[i,0] = pStarTot[0] - pStarTot[i]
            
            for i in range(N): # CONTINUITY of GRI
                a1D = Q1D[i,0]
                q1D = Q1D[i,1]
                u1D = q1D/a1D
                f[N+i,0] = x[N+i,0] - u1D + g1D[i]*self.IntegralRI(a1D, x[i,0], K[i], A0[i], m[i], n[i]) # Q1D[i,2], Q1D[i,3], m, n
                 
        return f
    
    def JunctionNwConn0D(self, N, Q1D, g1D, Q0D, g0D, K, A0, m, n, P0, couplePtot=1):
        """
        First-order hybrid junction with N 1D vessels and a 0D element
        
        Returns
        -------
        QBC1D : star vectors to be prescribed at inlet/outlet vertex of the 1D vessels
        Ptot0D : pressure to be imposed at the inlet/outlet of the 0D element
        """
        
        Q1Dsol = np.zeros((N, self.nVar), dtype=np.double)
        
        # INITIAL GUESS
        x0 = np.zeros((N*self.nVar, 1), dtype=np.double) # column vector # self.nVarJ
        # A0 = np.zeros(N, dtype=np.double)

        if not isinstance(m, (list, tuple, np.ndarray)):
            mVal = m
            m = np.zeros(N)
            m[:] = mVal
        if not isinstance(n, (list, tuple, np.ndarray)):
            nVal = n
            n = np.zeros(N)
            n[:] = nVal
        if not isinstance(P0, (list, tuple, np.ndarray)):
            P0Val = P0
            P0 = np.zeros(N)
            P0[:] = P0Val
        
        for i in range(N):
            a1D = Q1D[i][0]
            q1D = Q1D[i][1]
            u1D = q1D/a1D
            
            Q1Dsol[i,0] = a1D
            Q1Dsol[i,1] = q1D
            # Q1Dsol[i,2] = Q1D[i][2]
            # Q1Dsol[i,3] = Q1D[i][3]
            # Q1Dsol[i,4] = Q1D[i][4]
            
            x0[i,0] = a1D
            x0[N+i,0] = u1D
            
            # A0[i] = Q1D[i][3]
            
        # FUNCTION
        fun = lambda x : self.JunctionNwConn0Dfun(x, N, Q1Dsol, g1D, Q0D, g0D, K, A0, m, n, P0, couplePtot)
        
        # NONLINEAR SOLVER PARAMETERS
        tol = 1.0e-10
        tolMass = 1.0e-8
        nmax = 100

        # NEWTON METHOD 
        # xsol = self.NewtonNLSystem(fun, x0, A0, tol, tolMass, nmax)
        # GLOBALLY CONVERGENT NEWTON METHOD
        xsol = self.GlobConvNewtonNLSystem(fun, x0, A0, tol, tolMass, nmax)
        
        # fsol = fun(xsol)
        
        QBC1D = np.zeros((N, self.nVar), dtype=np.double)
        for i in range(N):
            QBC1D[i,0] = xsol[i,0]
            QBC1D[i,1] = xsol[i,0]*xsol[N+i,0]
            # QBC1D[i,2] = Q1Dsol[i,2]
            # QBC1D[i,3] = Q1Dsol[i,3]
            # QBC1D[i,4] = Q1Dsol[i,4]
        
        # Ptot0D = self.pFa(xsol[0,0], Q1Dsol[0,2], Q1Dsol[0,3], m, n, P0, Q1Dsol[0,4])
        Ptot0D = self.pFa(xsol[0,0], K[0], A0[0], m[0], n[0], P0[0])
        if couplePtot==1:
            Ptot0D = Ptot0D + 0.5*self.rho*xsol[N,0]**2.

        return QBC1D, Ptot0D

    def JunctionNwConn0D_o2(self, dt, N, Q1D, g1D, dx, Q0D, g0D, K, A0, m, n, P0, mu, couplePtot=1):
        """
        Second-order hybrid junction with N 1D vessels and a 0D element

        Returns
        -------
        QBC1D : star vectors to be prescribed at inlet/outlet vertex of the 1D vessels
        Q1Dbar : evolved boundary extrapolated vectors of the 1D vessels
        Ptot0D : pressure to be imposed at the inlet/outlet of the 0D element
        """
        
        Q1Dbar = np.zeros((N, self.nVar), dtype=np.double)
        
        # INITIAL GUESS
        x0 = np.zeros((N*self.nVar, 1), dtype=np.double) # column vector # self.nVarJ
        # A0 = np.zeros(N, dtype=np.double)

        if not isinstance(m, (list, tuple, np.ndarray)):
            mVal = m
            m = np.zeros(N)
            m[:] = mVal
        if not isinstance(n, (list, tuple, np.ndarray)):
            nVal = n
            n = np.zeros(N)
            n[:] = nVal
        if not isinstance(P0, (list, tuple, np.ndarray)):
            P0Val = P0
            P0 = np.zeros(N)
            P0[:] = P0Val

        for i in range(N):
            if g1D[i]==1:
                # STENCIL for ENO reconstruction
                Seno = [ Q1D[i][0], Q1D[i][1], Q1D[i][1]*10000. ] # i=M-1, i=M, i=M *10000
                dQeno = self.slope_single(Seno, dx[i])
                QL1 = Q1D[i][1] - 0.5*dx[i]*dQeno
                QL2 = Q1D[i][1] + 0.5*dx[i]*dQeno
                QLbar = QL2 - 0.5*dt/dx[i]*( self.pF(QL2, K[i], A0[i], m[i], n[i]) - self.pF(QL1, K[i], A0[i], m[i], n[i]) ) + 0.5*dt*self.S(QL2, mu[i])
                Q1Dbar[i,:] = np.copy(QLbar)
                a1D = QLbar[0] 
                q1D = QLbar[1] 
                u1D = q1D/a1D
                x0[i,0] = a1D
                x0[N+i,0] = u1D
            elif g1D[i]==-1:
                # STENCIL for ENO reconstruction
                Seno = [ Q1D[i][0]*10000., Q1D[i][0], Q1D[i][1] ] # i=1 *10000, i=1, i=2
                dQeno = self.slope_single(Seno, dx[i])
                QR1 = Q1D[i][0] - 0.5*dx[i]*dQeno
                QR2 = Q1D[i][0] + 0.5*dx[i]*dQeno
                QRbar = QR1 - 0.5*dt/dx[i]*( self.pF(QR2, K[i], A0[i], m[i], n[i]) - self.pF(QR1, K[i], A0[i], m[i], n[i]) ) + 0.5*dt*self.S(QR1, mu[i])
                Q1Dbar[i,:] = np.copy(QRbar)
                a1D = QRbar[0] 
                q1D = QRbar[1] 
                u1D = q1D/a1D
                x0[i,0] = a1D
                x0[N+i,0] = u1D
        
        # FUNCTION
        fun = lambda x : self.JunctionNwConn0Dfun(x, N, Q1Dbar, g1D, Q0D, g0D, K, A0, m, n, P0, couplePtot)

        # NONLINEAR SOLVER PARAMETERS
        tol = 1.0e-10
        tolMass = 1.0e-8
        nmax = 100

        # NEWTON METHOD 
        # xsol = self.NewtonNLSystem(fun, x0, A0, tol, tolMass, nmax)
        # GLOBALLY CONVERGENT NEWTON METHOD
        xsol = self.GlobConvNewtonNLSystem(fun, x0, A0, tol, tolMass, nmax)
        
        # fsol = fun(xsol)
        
        QBC1D = np.zeros((N, self.nVar), dtype=np.double)
        for i in range(N):
            QBC1D[i,0] = xsol[i,0]
            QBC1D[i,1] = xsol[i,0]*xsol[N+i,0]
            # QBC1D[i,2] = Q1Dbar[i,2]
            # QBC1D[i,3] = Q1Dbar[i,3]
            # QBC1D[i,4] = Q1Dbar[i,4]
        
        # Ptot0D = self.pFa(xsol[0,0], Q1Dsol[0,2], Q1Dsol[0,3], m, n, P0, Q1Dsol[0,4])
        Ptot0D = self.pFa(xsol[0,0], K[0], A0[0], m[0], n[0], P0[0])
        if couplePtot==1:
            Ptot0D = Ptot0D + 0.5*self.rho*xsol[N,0]**2.
       
        # return QBC1D, Q1Dbar, Ptot0D
        return QBC1D, Ptot0D

