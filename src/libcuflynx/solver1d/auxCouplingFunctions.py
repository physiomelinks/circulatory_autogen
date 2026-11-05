#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:21:04 2025

@author: bghi639
"""

import math
import numpy as np
import os
import sys


class auxCouplingFunctions:
    """
    Class for auxiliary functions for couplings (fully 1D and hybrid 1D-0D), boundary conditions and time evolution
    """
    def __init__(self, segmArray, nodesArray, mod, num, ODEsolver, couplePtot=0):
        
        self.Segments = segmArray
        self.Nodes = nodesArray
        
        self.mod = mod
        self.tol = self.mod.tol
        self.epsi = self.mod.epsi
        self.nMax = self.mod.nMax
        self.num = num
        
        self.ODEsolver = ODEsolver

        self.couplePtot = couplePtot

    
    def coupler1Dmulti_0D_o1(self, N, Q1D, g1D, Q0D, g0D, K, a0, m, n, P0):
        
        QStar_multi, P0D =self.mod.JunctionNwConn0D(N, Q1D, g1D, Q0D, g0D, K, a0, m, n, P0) #, self.couplePtot)

        return QStar_multi, P0D
    
    def coupler1Dmulti_0D_o2(self, dtEvol, N, Q1D, g1D, dx, Q0D, g0D, K, A0, m, n, P0, mu):
        
        QStar_multi, P0D = self.mod.JunctionNwConn0D_o2(dtEvol, N, Q1D, g1D, dx, Q0D, g0D, K, A0, m, n, P0, mu) #, self.couplePtot)

        return QStar_multi, P0D
    
    
    def coupler1D_0DpressureBC_o1(self, Q0D, iS, dx, Q1D, K, a0, m, n, P0, rec=0):
        
        nVar = len(Q1D)
        
        if rec==0:
            a1D = Q1D[0]
            u1D = Q1D[1]/a1D
        elif rec==1:
            if iS==-1:
                St = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
            elif iS==1:
                St = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
            Q1Drec = self.mod.BoundaryExtrapValues_single(St, dx, iS)
            a1D = Q1Drec[0]
            u1D = Q1Drec[1]/a1D

        qS = Q0D

        # Initial guess
        aS = a1D
        # Globally convergent Newton
        for i in range(self.nMax):
            # f = u1D - qS/aS + self.mod.IntegralRI(a1D, aS, K, a0, m, n)
            f = u1D - qS/aS - iS*self.mod.IntegralRI(a1D, aS, K, a0, m, n)
            
            if abs(f)<self.tol:
                break
            else:
                # Evaluate the derivative
                fM = u1D - qS/(aS - self.epsi) - iS*self.mod.IntegralRI(a1D, aS - self.epsi, K, a0, m, n)
                fP = u1D - qS/(aS + self.epsi) - iS*self.mod.IntegralRI(a1D, aS + self.epsi, K, a0, m, n)
            
                df = (fP-fM)/(2.*self.epsi)
                
                alpha = 1.
                for i2 in range(self.nMax):
                    aAux = aS - alpha*f/df
                    if aAux>0.:
                        fAux = u1D - qS/aAux - iS*self.mod.IntegralRI(a1D, aAux, K, a0, m, n)
                        if np.abs(fAux)<np.abs(f):
                            break
                        else:
                            alpha *= 0.8
                    else:
                        alpha *= 0.8
                # Update aS
                aS = aS - alpha*f/df
        
        if np.abs(f)>self.tol:
            print(i,a0,a1D,aS,np.abs(f))
            sys.exit('coupler1D_0DpressureBC_o1 :: Newton algortihm did NOT convergence')         
            
        # 1D-0D interface boundary vector QStar
        QStar = np.zeros(nVar) 
        QStar[0] = aS
        QStar[1] = qS
        
        if self.couplePtot==1:
            P0D = self.mod.pFa(aS, K, a0, m, n, P0) + 0.5*self.mod.rho*(qS/aS)**2.
        else:
            P0D = self.mod.pFa(aS, K, a0, m, n, P0)
        
        return QStar, P0D

    def coupler1D_0DpressureBC_o2(self, Q0D, iS, dx, dtEvol, Q1D, K, a0, m, n, P0, mu):
        
        nVar = len(Q1D)
        
        if dtEvol<self.tol:
            QStar, P0D = self.coupler1D_0DpressureBC_o1(Q0D, iS, dx, Q1D, K, a0, m, n, P0, 1)
        
        else:
            if iS==-1: # left boundary --> g=-1
                St = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
                dQ = self.mod.slope_single(St, dx) 
                Q0 = np.copy(Q1D[0])      
                QL = Q0 - 0.5*dx*dQ
                QR = Q0 + 0.5*dx*dQ
                QLbar = QL - dtEvol/dx*( self.mod.pF(QR, K, a0, m, n) - self.mod.pF(QL, K, a0, m, n) ) + dtEvol*self.mod.S(QL, mu)       
                a1D = QLbar[0]
                u1D = QLbar[1]/a1D
            elif iS==1: # right boundary --> g=+1
                St = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
                dQ = self.mod.slope_single(St, dx) 
                Q0 = np.copy(Q1D[1])   
                QL = Q0 - 0.5*dx*dQ
                QR = Q0 + 0.5*dx*dQ
                QRbar = QR - dtEvol/dx*( self.mod.pF(QR, K, a0, m, n) - self.mod.pF(QL, K, a0, m, n) ) + dtEvol*self.mod.S(QR, mu)
                a1D = QRbar[0]
                u1D = QRbar[1]/a1D
            
            qS = Q0D

            # Initial guess
            aS = a1D
            # Globally convergent Newton
            for i in range(self.nMax):
                # f = u1D - qS/aS + self.mod.IntegralRI(a1D, aS, K, a0, m, n)
                f = u1D - qS/aS - iS*self.mod.IntegralRI(a1D, aS, K, a0, m, n)
                
                if abs(f)<self.tol:
                    break
                else:
                    # Evaluate the derivative
                    fM = u1D - qS/(aS - self.epsi) - iS*self.mod.IntegralRI(a1D, aS - self.epsi, K, a0, m, n)
                    fP = u1D - qS/(aS + self.epsi) - iS*self.mod.IntegralRI(a1D, aS + self.epsi, K, a0, m, n)
                
                    df = (fP-fM)/(2.*self.epsi)
                    
                    alpha = 1.
                    for i2 in range(self.nMax):
                        aAux = aS - alpha*f/df
                        if aAux>0.:
                            fAux = u1D - qS/aAux - iS*self.mod.IntegralRI(a1D, aAux, K, a0, m, n)
                            if np.abs(fAux)<np.abs(f):
                                break
                            else:
                                alpha *= 0.8
                        else:
                            alpha *= 0.8
                    # Update aS
                    aS = aS - alpha*f/df
            
            # print("coupler1D_0DpressureBC_o2 ::",i,a0,a1D,aS,np.abs(f))
            if np.abs(f)>self.tol:
                print(i,a0,a1D,aS,np.abs(f))
                sys.exit('coupler1D_0DpressureBC_o2 :: Newton algortihm did NOT convergence')         
                
            # 1D-0D interface boundary vector QStar
            QStar = np.zeros(nVar) 
            QStar[0] = aS
            QStar[1] = qS
            
            if self.couplePtot==1:
                P0D = self.mod.pFa(aS, K, a0, m, n, P0) + 0.5*self.mod.rho*(qS/aS)**2.
            else:
                P0D = self.mod.pFa(aS, K, a0, m, n, P0)
            
        return QStar, P0D

    def coupler1D_0DflowBC_fun(self, x, P0D, iS, Q1D, K, a0, m, n, P0):
        a1D = Q1D[0,0]
        q1D = Q1D[0,1]
        u1D = q1D/a1D
        
        pS = self.mod.pFa(x[0,0], K, a0, m, n, P0)
        pStot = pS + 0.5*self.mod.rho*x[1,0]**2.
        
        # initialize junction function
        f = np.zeros_like(x) # column vector
        f[0,0] = pStot - P0D # continuity of total pressure   
        f[1,0] = x[1,0] - u1D + iS*self.mod.IntegralRI(a1D, x[0,0], K, a0, m, n) # continuity of GRI

        return f

    def coupler1D_0DflowBC_o1(self, P0D, Rwk, iS, dx, Q1D, K, a0, m, n, P0, rec=0):
        
        nVar = len(Q1D)

        if rec==0:
            a1D = Q1D[0]
            q1D = Q1D[1]
            u1D = q1D/a1D
        elif rec==1:
            if iS==-1:
                St = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
            elif iS==1:
                St = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
            Q1Drec = self.mod.BoundaryExtrapValues_single(St, dx, iS)
            a1D = Q1Drec[0]
            q1D = Q1Drec[1]
            u1D = q1D/a1D

        if Rwk>0.: # coupling 1D vessel to WK-RCR terminal element
            #XXX general for both iS=1 and iS=-1
            # pS = P0D + iS*Rwk*qS
            aS = a1D
            # GLOBALLY CONVERGENT NEWTON   
            for i in range(self.nMax):
                pS = self.mod.pFa(aS, K, a0, m, n, P0)
                qS = iS*(pS-P0D)/Rwk
                fun = qS/aS - u1D + iS*self.mod.IntegralRI(a1D, aS, K, a0, m, n)

                if np.abs(fun)<self.tol:
                    break
                else:
                    # Evaluate the derivative
                    pSM = self.mod.pFa(aS-self.epsi, K, a0, m, n, P0)
                    qSM = iS*(pSM-P0D)/Rwk
                    funM = qSM/(aS-self.epsi) - u1D + iS*self.mod.IntegralRI(a1D, aS-self.epsi, K, a0, m, n)
                
                    pSP = self.mod.pFa(aS+self.epsi, K, a0, m, n, P0)
                    qSP = iS*(pSP-P0D)/Rwk
                    funP = qSP/(aS+self.epsi) - u1D + iS*self.mod.IntegralRI(a1D, aS+self.epsi, K, a0, m, n)
                
                    df = (funP - funM)/(2.*self.epsi)
                    
                    alpha = 1.
                    for i2 in range(self.nMax):
                        aAux = aS - alpha*fun/df
                        if aAux>0.:
                            pSAux = self.mod.pFa(aAux, K, a0, m, n, P0)
                            qSAux = iS*(pSAux-P0D)/Rwk
                            fAux = qSAux/aAux - u1D + iS*self.mod.IntegralRI(a1D, aAux, K, a0, m, n)
                            if np.abs(fAux)<np.abs(fun):
                                break
                            else:
                                alpha *= 0.8
                        else:
                            alpha *= 0.8
                    # update Astar
                    aS = aS - alpha*fun/df
                    
            if np.abs(fun)>self.tol:
                print(i,a1D,u1D,aS,np.abs(fun))
                sys.exit('coupler1D_0DflowBC_o1 :: Newton algortihm did NOT convergence for 1D-0DWK coupling')
            
            pS = self.mod.pFa(aS, K, a0, m, n, P0)
            if iS==1:
                qS = (pS-P0D)/Rwk
            elif iS==-1:
                qS = (P0D-pS)/Rwk

        else: # coupling 1D vessel to 0D vessel
            if self.couplePtot==1: # via total pressure 
                Q1Dsol = np.zeros((1, nVar), dtype=np.double)
                # INITIAL GUESS
                x0 = np.zeros((nVar, 1), dtype=np.double) # column vector
                
                Q1Dsol[0,0] = a1D
                Q1Dsol[0,1] = q1D
                x0[0,0] = a1D
                x0[1,0] = u1D
                    
                # FUNCTION
                fun = lambda x : self.coupler1D_0DflowBC_fun(x, P0D, iS, Q1Dsol, K, a0, m, n, P0)
                # NONLINEAR SOLVER PARAMETERS
                tol_loc = 1.0e-10
                tolMass_loc = 1.0e-8
                nmax_loc = 100
                
                # NEWTON METHOD 
                # xsol = self.mod.NewtonNLSystem(fun, x0, np.array([a0]), tol_loc, tolMass_loc, nmax_loc) 
                # GLOBALLY CONVERGENT NEWTON METHOD
                xsol = self.mod.GlobConvNewtonNLSystem(fun, x0, np.array([a0]), tol_loc, tolMass_loc, nmax_loc) 
                
                aS = xsol[0,0]
                qS = xsol[0,0]*xsol[1,0]
            else: # via pressure
                # print("coupler1D_0DflowBC_o1 ::", P0D/1333.2238, a1D)
                pS = P0D
                aS = self.mod.aFp(pS, K, a0, m, n, P0)
                # print("coupler1D_0DflowBC_o1 ::", aS)

                intRI = self.mod.IntegralRI(a1D, aS, K, a0, m, n)
                qS = (u1D - iS*intRI)*aS
                # print("coupler1D_0DflowBC_o1 ::", qS)

        # 1D-0D interface boundary vector QStar
        QStar = np.zeros(nVar)       
        QStar[0] = aS
        QStar[1] = qS
        Q0D = qS

        return QStar, Q0D

    def coupler1D_0DflowBC_o2(self, P0D, Rwk, iS, dx, dtEvol, Q1D, K, a0, m, n, P0, mu):
        
        nVar = len(Q1D)

        if dtEvol<self.tol:
            QStar, Q0D = self.coupler1D_0DflowBC_o1(P0D, Rwk, iS, dx, Q1D, K, a0, m, n, P0, 1)

        else:                                
            if iS==-1: # left boundary --> g=-1
                St = [ Q1D[0]*10000., Q1D[0], Q1D[1] ] # i=1 *10000, i=1, i=2
                dQ = self.mod.slope_single(St, dx) 
                Q0 = np.copy(Q1D[0])      
                QL = Q0 - 0.5*dx*dQ
                QR = Q0 + 0.5*dx*dQ
                QLbar = QL - dtEvol/dx*( self.mod.pF(QR, K, a0, m, n) - self.mod.pF(QL, K, a0, m, n) ) + dtEvol*self.mod.S(QL, mu)       
                a1D = QLbar[0]
                q1D = QLbar[1]
                u1D = q1D/a1D
            elif iS==1: # right boundary --> g=+1
                St = [ Q1D[0], Q1D[1], Q1D[1]*10000. ] # i=M-1, i=M, i=M *10000
                dQ = self.mod.slope_single(St, dx) 
                Q0 = np.copy(Q1D[1])   
                QL = Q0 - 0.5*dx*dQ
                QR = Q0 + 0.5*dx*dQ
                QRbar = QR - dtEvol/dx*( self.mod.pF(QR, K, a0, m, n) - self.mod.pF(QL, K, a0, m, n) ) + dtEvol*self.mod.S(QR, mu)
                a1D = QRbar[0]
                q1D = QRbar[1]
                u1D = q1D/a1D


            if Rwk>0.: # coupling 1D vessel to WK-RCR terminal element
                #XXX general for both iS=1 and iS=-1
                # pS = P0D + iS*Rwk*qS
                aS = a1D
                # GLOBALLY CONVERGENT NEWTON   
                for i in range(self.nMax):
                    pS = self.mod.pFa(aS, K, a0, m, n, P0)
                    qS = iS*(pS-P0D)/Rwk
                    fun = qS/aS - u1D + iS*self.mod.IntegralRI(a1D, aS, K, a0, m, n)
                        
                    if np.abs(fun)<self.tol:
                        break
                    else:
                        # Evaluate the derivative
                        pSM = self.mod.pFa(aS-self.epsi, K, a0, m, n, P0)
                        qSM = iS*(pSM-P0D)/Rwk
                        funM = qSM/(aS-self.epsi) - u1D + iS*self.mod.IntegralRI(a1D, aS-self.epsi, K, a0, m, n)
                    
                        pSP = self.mod.pFa(aS+self.epsi, K, a0, m, n, P0)
                        qSP = iS*(pSP-P0D)/Rwk
                        funP = qSP/(aS+self.epsi) - u1D + iS*self.mod.IntegralRI(a1D, aS+self.epsi, K, a0, m, n)
                    
                        df = (funP - funM)/(2.*self.epsi)
                        
                        alpha = 1.
                        for i2 in range(self.nMax):
                            aAux = aS - alpha*fun/df
                            if aAux>0.:
                                pSAux = self.mod.pFa(aAux, K, a0, m, n, P0)
                                qSAux = iS*(pSAux-P0D)/Rwk
                                fAux = qSAux/aAux - u1D + iS*self.mod.IntegralRI(a1D, aAux, K, a0, m, n)
                                if np.abs(fAux)<np.abs(fun):
                                    break
                                else:
                                    alpha *= 0.8
                            else:
                                alpha *= 0.8
                        # update Astar
                        aS = aS - alpha*fun/df
                        
                if np.abs(fun)>self.tol:
                    print(i,a1D,u1D,aS,np.abs(fun))
                    sys.exit('coupler1D_0DflowBC_o2 :: Newton algortihm did NOT convergence for 1D-0DWK coupling')
                
                pS = self.mod.pFa(aS, K, a0, m, n, P0)
                if iS==1:
                    qS = (pS-P0D)/Rwk
                elif iS==-1:
                    qS = (P0D-pS)/Rwk
            
            else: # coupling 1D vessel to 0D vessel
                if self.couplePtot==1: # via total pressure 
                    Q1Dsol = np.zeros((1, nVar), dtype=np.double)
                    # INITIAL GUESS
                    x0 = np.zeros((nVar, 1), dtype=np.double) # column vector
                    
                    Q1Dsol[0,0] = a1D
                    Q1Dsol[0,1] = q1D
                    x0[0,0] = a1D
                    x0[1,0] = u1D
                        
                    # FUNCTION
                    fun = lambda x : self.coupler1D_0DflowBC_fun(x, P0D, iS, Q1Dsol, K, a0, m, n, P0)
                    # NONLINEAR SOLVER PARAMETERS
                    tol_loc = 1.0e-10
                    tolMass_loc = 1.0e-8
                    nmax_loc = 100
                    
                    # NEWTON METHOD 
                    # xsol = self.mod.NewtonNLSystem(fun, x0, np.array([a0]), tol_loc, tolMass_loc, nmax_loc) 
                    # GLOBALLY CONVERGENT NEWTON METHOD
                    xsol = self.mod.GlobConvNewtonNLSystem(fun, x0, np.array([a0]), tol_loc, tolMass_loc, nmax_loc) 
                    
                    aS = xsol[0,0]
                    qS = xsol[0,0]*xsol[1,0]
                else: # via pressure
                    # print("coupler1D_0DflowBC_o2 ::", P0D/1333.2238, a1D)
                    pS = P0D
                    aS = self.mod.aFp(pS, K, a0, m, n, P0)
                    # print("coupler1D_0DflowBC_o2 ::", aS)

                    intRI = self.mod.IntegralRI(a1D, aS, K, a0, m, n)
                    qS = (u1D - iS*intRI)*aS
                    # print("coupler1D_0DflowBC_o2 ::", qS)


            # 1D-0D interface boundary vector QStar
            QStar = np.zeros(nVar)       
            QStar[0] = aS
            QStar[1] = qS
            Q0D = qS

        return QStar, Q0D


    def getBC_fully1D(self, time, dt, T0, v, T, Idxs1D, IdxsTerm, nV, nN, nVar, FVord, K, A0, mu, mTL, nTL, P0, Rt, Pout): 
        
        # SET INTERFACE CONDITIONS AT NODES (inflow, junction nodes, singleR-terminal nodes)
        QBCinLoc = np.zeros((nV, nVar))
        QBCoutLoc = np.zeros((nV, nVar))
            
        if FVord==1:
            for i in range(nN):
                if self.Nodes[i][1]=='inflow':
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    tLoc = time - math.floor(time/T0)*T0
                    Qin = v[iV].inflow(tLoc) # [ml/s]
                    # Qin = bcFunc.inflowLinearIncrFun(time, v[iV].inflow)
                    if iS==-1:
                        QBCinLoc[iV,:] = self.mod.ImposedFlowArt(Qin, v[iV].Q[0,:], K[iV], A0[iV], mTL[iV], nTL[iV])
                        
                elif self.Nodes[i][1]=='junc1D':
                    # iJ = self.Nodes[i][0]
                    nJ = self.Nodes[i][-1]
                    Q1DJ = []
                    gJ = []
                    KJ = []
                    A0J = []
                    mTLJ = []
                    nTLJ = []
                    P0J = []
                    for j in range(nJ):
                        iV = self.Nodes[i][2+j*2]
                        gV = self.Nodes[i][2+j*2+1]
                        gJ.append(gV)
                        KJ.append(K[iV])
                        A0J.append(A0[iV])
                        mTLJ.append(mTL[iV])
                        nTLJ.append(nTL[iV])
                        P0J.append(P0[iV])
                        if gV==-1:
                            Q1DJ.append( v[iV].Q[0,:] )
                        elif gV==1:
                            Q1DJ.append( v[iV].Q[-1,:] )
                            
                    QBCjunc = self.mod.JunctionN(nJ, Q1DJ, gJ, KJ, A0J, mTLJ, nTLJ, P0J)

                    for j in range(nJ):
                        iV = self.Nodes[i][2+j*2]
                        # gV = Nodes[i][2+j*2+1]
                        if gJ[j]==-1:
                            QBCinLoc[iV,:] = QBCjunc[j,:]
                        elif gJ[j]==1:
                            QBCoutLoc[iV,:] = QBCjunc[j,:]
                
                elif self.Nodes[i][1]=='singleR':
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    # Pt = Pout
                    Pt = self.PoutTerm(time, T0, Pout)
                    if iS==1:
                        QBCoutLoc[iV,:] = self.mod.Terminal1DsingleR(v[iV].Q[-1,:], Rt[iV], Pt, K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV])
                        
                elif self.Nodes[i][1]=='pOutflow':
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    # pOutflow = Pout
                    pOutflow = self.PoutTerm(time, T0, Pout)
                    if iS==1:
                        QBCoutLoc[iV,:] = self.mod.ImposedPressure(pOutflow, iS, v[iV].Q[-1,:], K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV])
        
        elif FVord==2:
            timeEvol = time+0.5*dt
            for i in range(nN):
                if self.Nodes[i][1]=='inflow_o2':
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    tLoc = timeEvol - math.floor(timeEvol/T0)*T0
                    Qin = v[iV].inflow(tLoc) # [ml/s]
                    if iS==-1:
                        QBCinLoc[iV,:] = self.mod.ImposedFlowArt_o2(v[iV].dx, dt, Qin, [v[iV].Q[0,:], v[iV].Q[1,:]], K[iV], A0[iV], mTL[iV], nTL[iV], mu[iV])
                        
                elif self.Nodes[i][1]=='junc1D':
                    # iJ = Nodes[i][0]
                    nJ = self.Nodes[i][-1]
                    dxJ = []
                    Q1DJ = []
                    gJ = []
                    KJ = []
                    A0J = []
                    muJ = []
                    mTLJ = []
                    nTLJ = []
                    P0J = []
                    for j in range(nJ):
                        iV = self.Nodes[i][2+j*2]
                        gV = self.Nodes[i][2+j*2+1]
                        dxJ.append(v[iV].dx)
                        gJ.append(gV)
                        KJ.append(K[iV])
                        A0J.append(A0[iV])
                        muJ.append(mu[iV])
                        mTLJ.append(mTL[iV])
                        nTLJ.append(nTL[iV])
                        P0J.append(P0[iV])
                        if gV==-1:
                            Q1DJ.append( [v[iV].Q[0,:], v[iV].Q[1,:]] )
                        elif gV==1:
                            Q1DJ.append( [v[iV].Q[-2,:], v[iV].Q[-1,:]] )
                            
                    QBCjunc = self.mod.JunctionN_o2(dxJ, dt, nJ, Q1DJ, gJ, KJ, A0J, mTLJ, nTLJ, P0J, muJ)
                    for j in range(nJ):
                        iV = self.Nodes[i][2+j*2]
                        # gV = Nodes[i][2+j*2+1]
                        if gJ[j]==-1:
                            QBCinLoc[iV,:] = QBCjunc[j,:]
                        elif gJ[j]==1:
                            QBCoutLoc[iV,:] = QBCjunc[j,:]
                
                elif self.Nodes[i][1]=='singleR':
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    # Pt = Pout
                    Pt = self.PoutTerm(timeEvol, T0, Pout)
                    if iS==1:
                        QBCoutLoc[iV,:] = self.mod.Terminal1DsingleR_o2(v[iV].dx, dt, [v[iV].Q[-2,:], v[iV].Q[-1,:]], Rt[iV], Pt, K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV], mu[iV])
                        
                elif self.Nodes[i][1]=='pOutflow':
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    # pOutflow = Pout
                    pOutflow = self.PoutTerm(timeEvol, T0, Pout)
                    if iS==1:
                        QBCoutLoc[iV,:] = self.mod.ImposedPressure_o2(v[iV].dx, dt, pOutflow, iS, [v[iV].Q[-2,:], v[iV].Q[-1,:]], K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV], mu[iV])
                        
        return QBCinLoc, QBCoutLoc


    def getBC_hybrid1D0D(self, time, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout): 

        # SET INTERFACE CONDITIONS AT OUTLET NODES COUPLED WITH RCR TERMINAL MODELS
        # QBCinLoc = np.zeros((nV, nVar)) #XXX NOT needed for now
        QBCoutLoc = np.zeros((nV, nVar))
        QinTerm = np.zeros(len(IdxsTerm))
        # P0DJhyb = np.zeros(len(IdxsJhyb))
        
        kT = 0
        if FVord==1:
            for i in range(nN):
                if self.Nodes[i][1]=='RCR':
                    iN = self.Nodes[i][0]
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    # Pt = Pout
                    Pt = self.PoutTerm(time, T0, Pout)
                    if iS==1:
                        QBCoutLoc[iV,:] = self.mod.Terminal1DRCR(v[iV].Q[-1,:], T[iN].Qsol[0], Rt1[iV], Ct[iV], Rt2[iV], Pt, K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV])

                        QinTerm[kT] = QBCoutLoc[iV,1]
                        kT +=1
                    
        elif FVord==2:
            for i in range(nN):
                if self.Nodes[i][1]=='RCR':
                    iN = self.Nodes[i][0]
                    iV = self.Nodes[i][2]
                    iS = self.Nodes[i][3]
                    # Pt = Pout
                    Pt = self.PoutTerm(time, T0, Pout) #XXX this 'time' is already the evolved one according to the ODE solver
                    if iS==1:
                        gT1D = iS
                        if ODEstep==1:
                            QBCrec = self.mod.BoundaryExtrapValues_single([v[iV].Q[-2,:], v[iV].Q[-1,:], v[iV].Q[-1,:]*10000.], v[iV].dx, gT1D)
                            QBCoutLoc[iV,:] = self.mod.Terminal1DRCR(QBCrec, T[iN].Qsol[0], Rt1[iV], Ct[iV], Rt2[iV], Pt, K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV])
                        else:
                            if ODEstep==2 or ODEstep==3:    
                                dtT1D = dt # because dtT1D is then divided by 2 in mod.Terminal1DRCR_o2 function
                            elif ODEstep==4:
                                dtT1D = 2.*dt # because dtT1D is then divided by 2 in mod.Terminal1DRCR_o2 function
                            QBCoutLoc[iV,:], _ = self.mod.Terminal1DRCR_o2(v[iV].dx, dtT1D, [v[iV].Q[-2,:], v[iV].Q[-1,:]], T[iN].Qsol[0], 
                                                                            Rt1[iV], Ct[iV], Rt2[iV], Pt, K[iV], A0[iV], mTL[iV], nTL[iV], P0[iV], mu[iV])
                            
                        QinTerm[kT] = QBCoutLoc[iV,1]
                        kT +=1
                    
        # return QBCinLoc, QBCoutLoc, QinTerm
        # return QBCoutLoc, P0DJhyb, QinTerm
        return QBCoutLoc, QinTerm
    

    def solve_time_step_1D(self, time, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, numFlux, slopeTy, termTy, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout, couple_zero_one=False):
        
        if FVord==1: # and ODEsolver=='explEul': # FIRST-ORDER FV SCHEME + EXPLICIT EULER METHOD
        
            # FnumQBCL = np.zeros((nV, nVar))
            # FnumQBCR = np.zeros((nV, nVar))

            QBCin1D, QBCout1D = self.getBC_fully1D(time, dt, T0, v, T, Idxs1D, IdxsTerm, nV, nN, nVar, FVord, K, A0, mu, mTL, nTL, P0, Rt1, Pout)
            QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, -1, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout) 
            
            # for i in Idxs1D:
            for i in range(nV):
                # Left boundary
                if self.Segments[i][3]=='inflow' or self.Segments[i][3]=='junc1D':
                    # FnumQBCL[i,:] = mod.pF(QBCin1D[i,:], K[i], A0[i], m, n)
                    v[i].FnumQBCL[:] = self.mod.pF(QBCin1D[i,:], K[i], A0[i], mTL[i], nTL[i])
                # Right boundary 
                if self.Segments[i][5]=='junc1D' or self.Segments[i][5]=='singleR' or self.Segments[i][5]=='pOutflow':
                    # FnumQBCR[i,:] = self.mod.pF(QBCout1D[i,:], K[i], A0[i], m, n)
                    v[i].FnumQBCR[:] = self.mod.pF(QBCout1D[i,:], K[i], A0[i], mTL[i], nTL[i])
                elif self.Segments[i][5]=='RCR':
                    # FnumQBCR[i,:] = self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                    v[i].FnumQBCR[:] = self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                # elif self.Segments[i][5]=='juncHyb':
                #     # FnumQBCR[i,:] = self.mod.pF(QS, K[i], A0[i], m, n)
                #     v[i].FnumQBCR[:] = self.mod.pF(QS, K[i], A0[i], m, n)
            
            if termTy=='RCR':
                k=0
                for i in IdxsTerm:
                    Pt = self.PoutTerm(time, T0, Pout)
                    # f0wkEul = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                    DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                    f0wkEul = np.array([DPwk])
                    
                    T[i].x = T[i].x + dt*f0wkEul
                    T[i].Qsol = np.copy(T[i].x)
                    k=k+1
                
            # UPDATE 1D SOLUTION
            # if abs(dt)>1.0e-8:
            # for i in Idxs1D:
            for i in range(nV):
                v[i].Q = self.num.evolve1D_mod(v[i].Q, v[i].dx, dt, v[i].FnumQBCL, v[i].FnumQBCR, numFlux, time, K[i], A0[i], mTL[i], nTL[i], P0[i], mu[i])

        elif FVord==2: # SECOND-ORDER FV SCHEME
            
            # FnumQBCL = np.zeros((nV, nVar))
            # FnumQBCR = np.zeros((nV, nVar))
            
            QBCin1D, QBCout1D = self.getBC_fully1D(time, dt, T0, v, T, Idxs1D, IdxsTerm, nV, nN, nVar, FVord, K, A0, mu, mTL, nTL, P0, Rt1, Pout)
            ODEstep = 1
            QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout)
 
            # for i in Idxs1D:
            for i in range(nV):
                # Left boundary
                if self.Segments[i][3]=='inflow_o2' or self.Segments[i][3]=='junc1D':
                    # FnumQBCL[i,:] = self.mod.pF(QBCin1D[i,:], K[i], A0[i], m, n)
                    v[i].FnumQBCL[:] = self.mod.pF(QBCin1D[i,:], K[i], A0[i], mTL[i], nTL[i])
                # Right boundary
                if self.Segments[i][5]=='junc1D' or self.Segments[i][5]=='singleR' or self.Segments[i][5]=='pOutflow':
                    # FnumQBCR[i,:] = self.mod.pF(QBCout1D[i,:], K[i], A0[i], m, n)
                    v[i].FnumQBCR[:] = self.mod.pF(QBCout1D[i,:], K[i], A0[i], mTL[i], nTL[i])
                # elif self.Segments[i][5]=='juncHyb':
                #     print("Num flux at hybrid junction already computed: ", v[i].FnumQBCR)
                    
            if termTy=='RCR':
                if self.ODEsolver=='midpoint': # MIDPOINT METHOD
                    Twk0 = []
                    k=0
                    for i in IdxsTerm:
                        Twk0.append([])
                        Twk0[k].append(T[i].x)
                        Pt = self.PoutTerm(time, T0, Pout)
                        # T1 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        T1 = np.array([DPwk])
                        
                        T[i].x = Twk0[k][0] + 0.5*dt*T1
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR':
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                    
                    ODEstep = 2
                    QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time+0.5*dt, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout) 
                    
                    k=0
                    for i in IdxsTerm:
                        Pt = self.PoutTerm(time+0.5*dt, T0, Pout)
                        # T2 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        T2 = np.array([DPwk])
                    
                        T[i].x = Twk0[k][0] + dt*T2
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR':
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                
                elif self.ODEsolver=='Heun': # HEUN'S METHOD
                    Twk0 = []
                    f1wkHeun = []
                    k=0
                    for i in IdxsTerm:
                        Twk0.append([])
                        f1wkHeun.append([])
                        Twk0[k].append(T[i].x)
                        Pt = self.PoutTerm(time, T0, Pout)
                        # T1 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        T1 = np.array([DPwk])
                        f1wkHeun[k].append(T1)
                        
                        T[i].x = Twk0[k][0] + dt*T1
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR' :
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                            
                    ODEstep = 4
                    QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time+dt, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout) 
                    
                    k=0
                    for i in IdxsTerm:
                        Pt = self.PoutTerm(time+dt, T0, Pout)
                        # T2 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        T2 = np.array([DPwk])
                        
                        T[i].x = Twk0[k][0] + 0.5*dt*( f1wkHeun[k][0] + T2 )
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR' :
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + 0.5*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                
                elif self.ODEsolver=='RK4': # RUNGE-KUTTA 4 METHOD
                    wRK4 = np.array([1./6., 1./3., 1./3., 1./6.])
                    Twk0 = []
                    incrwkRK4 = []
                    k=0
                    for i in IdxsTerm:
                        Twk0.append([])
                        incrwkRK4.append([])
                        Twk0[k].append(T[i].x)
                        Pt = self.PoutTerm(time, T0, Pout)
                        # K1 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        K1 = np.array([DPwk])
                        incrwkRK4[k].append(wRK4[0]*K1)
                        
                        T[i].x = Twk0[k][0] + 0.5*dt*K1
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR' :
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + wRK4[0]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + wRK4[0]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                        
                    ODEstep = 2
                    QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time+0.5*dt, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout) 
                    
                    k=0
                    for i in IdxsTerm:
                        Pt = self.PoutTerm(time+0.5*dt, T0, Pout)
                        # K2 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        K2 = np.array([DPwk])
                        incrwkRK4[k].append(wRK4[1]*K2)
                        
                        T[i].x = Twk0[k][0] + 0.5*dt*K2
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR' :
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + wRK4[1]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + wRK4[1]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                    
                    ODEstep = 3
                    QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time+0.5*dt, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout) 
                    
                    k=0
                    for i in IdxsTerm:
                        Pt = self.PoutTerm(time+0.5*dt, T0, Pout)
                        # K3 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        K3 = np.array([DPwk])
                        incrwkRK4[k].append(wRK4[2]*K3)

                        T[i].x = Twk0[k][0] + dt*K3
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR' :
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + wRK4[2]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + wRK4[2]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                            
                    ODEstep = 4
                    QBCoutHyb, QinTerm = self.getBC_hybrid1D0D(time+dt, dt, T0, v, T, Idxs1D, IdxsJhyb, IdxsTerm, nV, nN, nVar, FVord, slopeTy, ODEstep, K, A0, mu, mTL, nTL, P0, Rt1, Ct, Rt2, Pout) 
                    
                    k=0
                    for i in IdxsTerm:
                        Pt = self.PoutTerm(time+dt, T0, Pout)
                        # K4 = getT(dt, T, i, QinTerm[k], Pt, Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        DPwk = T[i].DPTerminal0D(dt, QinTerm[k], Pt, T[i].Qsol[0], Rt1[self.Nodes[i][2]], Ct[self.Nodes[i][2]], Rt2[self.Nodes[i][2]])
                        K4 = np.array([DPwk])
                        incrwkRK4[k].append(wRK4[3]*K4)
                        k=k+1
                        
                    # for i in Idxs1D:
                    for i in range(nV):
                        # Right boundary
                        if self.Segments[i][5]=='RCR' :
                            # FnumQBCR[i,:] = FnumQBCR[i,:] + wRK4[3]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], m, n)
                            v[i].FnumQBCR[:] = v[i].FnumQBCR[:] + wRK4[3]*self.mod.pF(QBCoutHyb[i,:], K[i], A0[i], mTL[i], nTL[i])
                        
                    k=0
                    for i in IdxsTerm:
                        T[i].x = Twk0[k][0] + dt*np.sum(incrwkRK4[k])
                        T[i].Qsol = np.copy(T[i].x)
                        k=k+1
                    
            # UPDATE 1D SOLUTION
            # if abs(dt)>1.0e-8:
            # for i in Idxs1D:
            for i in range(nV):
                v[i].Q = self.num.evolve1D_MUSCLHancock_mod(v[i].Q, v[i].dx, dt, v[i].FnumQBCL, v[i].FnumQBCR, slopeTy, time, K[i], A0[i], mTL[i], nTL[i], P0[i], mu[i])


    def PoutTerm(self, tt, T0, PoutVal, PoutFun=None):
        if PoutFun==None:
            return PoutVal
        else:
            ttLoc = tt - math.floor(tt/T0)*T0
            return PoutFun(ttLoc)
        
          