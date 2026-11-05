#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:25:28 2020

@author: beatriceghitti
"""

import numpy as np
from scipy.interpolate import interp1d
# from scipy.optimize import fsolve
# from scipy.optimize import root


class windkessel:
    """
    Class for a 0D/lumped-parameter module
    This can be (i) a 0D vessel, (ii) a single-resistance element, (iii) a RCR Windkessel element
    """
    def __init__(self, length, leftBCtype, rightBCtype, vessType, K, a0, mu, m, n, P0, mod, inflowData, outFolder, idxV):
        
        self.length = length # vessel length
        
        self.leftBCtype = leftBCtype
        self.rightBCtype = rightBCtype
        
        self.vessType = vessType # 0D element type 
        # number of state variables
        if self.vessType=='PinQout' or self.vessType=='QinPout' :
            self.nVar = 3
        elif self.vessType=='PinQout_SPLIT' or self.vessType=='QinPout_SPLIT' :
            self.nVar = 6
        elif self.vessType=='PinPout' :
            self.nVar = 4
        elif self.vessType=='QinQout' :
            self.nVar = 5
        elif self.vessType=='singleR' : # single-resistance terminal element
            self.nVar = 1
        elif self.vessType=='RCR': # RCR Windkessel terminal element
            self.nVar = 1
        
        self.Qsol = np.zeros(self.nVar) # state variables are Volume V, Pressure P, and Flow rate Q
        
        # model/tube law parameters
        self.m = m
        self.n = n
        self.K = K # vessel wall stiffness
        self.a0 = a0 # reference/baseline vessel cross-sectional area
        self.V0 = a0*length
        self.P0 = P0 # reference/baseline pressure for which A=a0
        
        self.mod = mod # blood flow model
        
        self.mu = mu # "vessel-specific" blood dynamic viscosity
        self.rho = self.mod.rho # constant blood density
        self.kR =  self.mod.frictionParam(self.mu)
        
        
        if (self.leftBCtype=='inflow' or self.leftBCtype=='inflow_o2'):
            tData = inflowData[:,0]
            qData = inflowData[:,1]
            self.inflow = interp1d(tData, qData, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            
            
        if self.vessType=='PinQout' or self.vessType=='QinPout' :
            self.outFile1 = open(outFolder+'sol0D_vess'+str(idxV)+'.txt','w')
            self.outFile1.write("# 0: t[s]; 1: Q[ml/s]; 2: P[dyne/cm2]; 3: A[cm2]; \n")
            self.outFile1.flush()
            # self.outFile2 = open(outFolder+'/errors_vess'+str(idxV)+'.txt','w')
            # self.outFile2.write("#0: Linf-Q; 1: Linf-P; 2: Linf-A;\n")
            # self.outFile2.flush()
        elif self.vessType=='PinQout_SPLIT' or self.vessType=='QinPout_SPLIT' :
            self.outFile1 = open(outFolder+'sol0D_vess'+str(idxV)+'.txt','w')
            self.outFile1.write("# 0: t[s]; 1: Q1[ml/s]; 2: P1[dyne/cm2]; 3: A1[cm2]; 4: Q2[ml/s]; 5: P2[dyne/cm2]; 6: A2[cm2]; \n")
            self.outFile1.flush()
        elif self.vessType=='PinPout' :
            self.outFile1 = open(outFolder+'sol0D_vess'+str(idxV)+'.txt','w')
            self.outFile1.write("# 0: t[s]; 1: Q[ml/s]; 2: P[dyne/cm2]; 3: A[cm2]; 4: Qd[ml/s]; \n")
            self.outFile1.flush()
        elif self.vessType=='QinQout' :
            self.outFile1 = open(outFolder+'sol0D_vess'+str(idxV)+'.txt','w')
            self.outFile1.write("# 0: t[s]; 1: Q[ml/s]; 2: P[dyne/cm2]; 3: A[cm2]; 4: Pd[dyne/cm2]; 5: Ad[cm2]; \n")
            self.outFile1.flush()
        elif self.vessType=='RCR' :
            self.outFile1 = open(outFolder+'sol0D_RCRterm'+str(idxV)+'.txt','w')
            self.outFile1.write("# 0: t[s]; 1: P[dyne/cm2]; \n")
            self.outFile1.flush()
            
            
    def outputResults(self, time, Qnum):
        self.outFile1.write("%.18e " % (time))
        for i in range(len(Qnum)):
            self.outFile1.write("%.18e " % (Qnum[i]))
        self.outFile1.write("\n")
        self.outFile1.flush()
        
    # def outputErrors(self, errQ, errP, errA):
    #     self.outFile2.write("%.18e %.18e %.18e \n" % (errQ, errP, errA))
    #     self.outFile2.flush()
        
        
    def meanA(self, V, l):
        """
        Compute the average cross-sectional area at time t from volume V(t)
        """
        A = V/l     
        return A
    
    def NLcompliance(self, V, l):
        """
        Compute non-linear compliance C(A)
        """
        A = self.meanA(V, l)
        
        dPdA = self.mod.dpda(A, self.K, self.a0, self.m, self.n)
        dAdP = 1./dPdA
        
        C = l*dAdP  
        return C
    
    def NLinductance(self, V, l):
        """
        Compute non-linear inductance L(A)
        """
        A = self.meanA(V, l)
        L = self.mod.rho*l/A       
        return L
    
    def NLresistance(self, V, l):
        """
        Compute non-linear resistance R(A)
        """
        A = self.meanA(V, l)
        R = self.mod.rho*self.kR*l/(A**2.)
        return R
    
    def NLconvection(self, V, l):
        """
        Compute non-linear coefficient K(A) for the convective term
        """
        A = self.meanA(V, l)     
        KK = self.mod.rho*self.mod.coriolis/A
        return KK
    
    def linearPfromA(self, A, l, V0, C0):
        V = A*l
        P = (V-V0)/C0 + self.P0 + self.mod.Pe    
        return P
    
    def linearPfromV(self, V, V0, C0):
        P = (V-V0)/C0 + self.P0 + self.mod.Pe    
        return P
        
    def linearAfromP(self, P, l, C0):
        A = self.a0 + C0/l*(P - self.P0 - self.mod.Pe)       
        return A
    
    def linearVfromP(self, P, V0, C0):
        V = V0 + C0*(P - self.P0 - self.mod.Pe)       
        return V
        
    def XQinPoutVess(self, dt, Qin, Pout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Qin, Pout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        # P = Qsol[2]
        x = [ V, Q ]
        return x
    
    def QinPoutVessState(self, x, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Pin, Qout)
        """
        V = x[0]
        Q = x[1]
        
        if nonlinear==0: # linear 0D model
            C = C0
            P = (V-V0A)/C + P0A + self.mod.Pe
        elif nonlinear==1: # nonlinear 0D model
            A = self.meanA(V, self.length)
            P = self.mod.pFa(A, self.K, self.a0, self.m, self.n, self.P0)
        
        Qsol = [ V, Q, P ]
        return Qsol
    
    def DQinPoutVess(self, dt, Qin, Pout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear, i, switchKK):
        """
        0D vessel with (Pin, Qout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        P = Qsol[2]
        
        if nonlinear==0: # linear 0D model
            C = C0
            R = R0*0.5
            Rp = R0*0.5
            L = L0
            if switchKK==1:
                KK = KK0
                A = self.linearAfromP(P, self.length, C)
                Aout = self.linearAfromP(Pout, self.length, C)
            elif switchKK==0:
                KK = 0.
        elif nonlinear==1: # nonlinear 0D model
            L = self.NLinductance(V, self.length)
            R = self.NLresistance(V, self.length)*0.5
            Rp = R
            if switchKK==1:
                KK = self.NLconvection(V, self.length)
                A = self.mod.aFp(P, self.K, self.a0, self.m, self.n, self.P0)
                Aout = self.mod.aFp(Pout, self.K, self.a0, self.m, self.n, self.P0)
            elif switchKK==0:
                KK = 0.

        dVNew = ( Qin - Q )
        
        if switchKK==0:
             dQNew = 1./L*( P - R*Q  - Pout )
        elif switchKK==1:
            dQNew = 1./L*( P - R*Q  - Pout - KK*(Q**2./Aout - Q**2./A) )
        
        dQsol = [ dVNew, dQNew ]    
        return dQsol
    
    
    def XPinQoutVess(self, dt, Pin, Qout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Pin, Qout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        # P = Qsol[2]
        x = [ V, Q ]
        return x
    
    def PinQoutVessState(self, x, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Pin, Qout)
        """
        V = x[0]
        Q = x[1]
        
        if nonlinear==0: # linear 0D model
            C = C0
            P = (V-V0A)/C + P0A + self.mod.Pe    
        elif nonlinear==1: # nonlinear 0D model
            A = self.meanA(V, self.length)
            P = self.mod.pFa(A, self.K, self.a0, self.m, self.n, self.P0)
        
        Qsol = [ V, Q, P ]       
        return Qsol
    
    def DPinQoutVess(self, dt, Pin, Qout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear, i, switchKK, Ulimit, gamma):
        """
        0D vessel with (Pin, Qout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        P = Qsol[2]
        
        if nonlinear==0: # linear 0D model
            C = C0
            R = R0*0.5
            Rd = R0*0.5
            L = L0
            if switchKK==1:
                KK = KK0
                Ain = self.linearAfromP(Pin, self.length, C)
                A = self.linearAfromP(P, self.length, C)
            elif switchKK==0:
                KK = 0.     
        elif nonlinear==1: # nonlinear 0D model
            L = self.NLinductance(V, self.length)
            R = self.NLresistance(V, self.length)*0.5
            Rd = R
            if switchKK==1:
                KK = self.NLconvection(V, self.length)
                Ain = self.mod.aFp(Pin, self.K, self.a0, self.m, self.n, self.P0)
                A = self.mod.aFp(P, self.K, self.a0, self.m, self.n, self.P0)
            elif switchKK==0:
                KK = 0.
        
        dVNew = ( Q - Qout )
        
        if switchKK==0:
            dQNew = 1./L*( Pin - R*Q - P ) 
        elif switchKK==1:
            dQNew = 1./L*( Pin - R*Q - P - KK*(Q**2./A - Q**2./Ain) )
            # dQNew = 1./L*( Pin - R*Q - Rd*Qout - Pout - KK*(Qout**2./Aout - Q**2./Ain) )
            # dQNew = 1./L*( Pin - R*Q - P - KK*(Qout**2./A - Q**2./Ain) )
            
        dQsol = [ dVNew, dQNew ]
        return dQsol


    def XPinQoutVess_SPLIT(self, dt, Pin, Qout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with two (Pin, Qout)-elements in series
        """
        V1 = Qsol[0]
        Q1 = Qsol[1]
        # P1 = Qsol[2]
        V2 = Qsol[3]
        Q2 = Qsol[4]
        # P2 = Qsol[5]
        x = [ V1, Q1, V2, Q2 ]
        return x
    
    def PinQoutVessState_SPLIT(self, x, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with two (Pin, Qout)-elements in series
        """
        V1 = x[0]
        Q1 = x[1]
        V2 = x[2]
        Q2 = x[3]
        
        if nonlinear==0: # linear model
            C1 = 0.5*C0
            C2 = 0.5*C0
            P1 = (V1-0.5*V0A)/C1 + P0A + self.mod.Pe
            P2 = (V2-0.5*V0A)/C2 + P0A + self.mod.Pe         
        elif nonlinear==1: # nonlinear model
            A1 = self.meanA(V1, 0.5*self.length)
            P1 = self.mod.pFa(A1, self.K, self.a0, self.m, self.n, self.P0)
            A2 = self.meanA(V2, 0.5*self.length)
            P2 = self.mod.pFa(A2, self.K, self.a0, self.m, self.n, self.P0)
        
        Qsol = [ V1, Q1, P1, V2, Q2, P2 ]
        return Qsol
    
    def DPinQoutVess_SPLIT(self, dt, Pin, Qout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear, i, switchKK, Ulimit, gamma):
        """
        0D vessel with two (Pin, Qout)-elements in series
        """
        V1 = Qsol[0]
        Q1 = Qsol[1]
        P1 = Qsol[2]
        V2 = Qsol[3]
        Q2 = Qsol[4]
        P2 = Qsol[5]
        
        if nonlinear==0: # linear model
            C1 = 0.5*C0
            C2 = 0.5*C0
            R1 = 0.25*R0
            Rd1 = 0.25*R0
            R2 = 0.25*R0
            Rd2 = 0.25*R0
            L1 = 0.5*L0
            L2 = 0.5*L0
            Pstar = P1 - Rd1*Q2
            if switchKK==1:
                KK1 = KK0
                KK2 = KK0
                Ain = self.linearAfromP(Pin, 0.5*self.length, C1)
                A1 = self.linearAfromP(P1, 0.5*self.length, C1)
                Astar = self.linearAfromP(Pstar, 0.5*self.length, C1)
                A2 = self.linearAfromP(P2, 0.5*self.length, C2)
            elif switchKK==0:
                KK1 = 0.
                KK2 = 0.      
        elif nonlinear==1: # nonlinear model
            L1 = self.NLinductance(V1, 0.5*self.length)
            L2 = self.NLinductance(V2, 0.5*self.length)
            R1 = self.NLresistance(V1, 0.5*self.length)*0.5
            Rd1 = R1
            R2 = self.NLresistance(V2, 0.5*self.length)*0.5
            Rd2 = R2
            Pstar = P1 - Rd1*Q2
            if switchKK==1:
                KK1 = self.NLconvection(V1, 0.5*self.length)
                KK2 = self.NLconvection(V2, 0.5*self.length)
                Ain = self.mod.aFp(Pin, self.K, self.a0, self.m, self.n, self.P0)
                A1 = self.mod.aFp(P1, self.K, self.a0, self.m, self.n, self.P0)
                Astar = self.mod.aFp(Pstar, self.K, self.a0, self.m, self.n, self.P0)
                A2 = self.mod.aFp(P2, self.K, self.a0, self.m, self.n, self.P0)
            elif switchKK==0:
                KK1 = 0.
                KK2 = 0.
        
        # dV1New = ( Q1 - Qstar )
        dV1New = ( Q1 - Q2 )
        
        if switchKK==0:
            dQ1New = 1./L1*( Pin - R1*Q1 - P1 )
        elif switchKK==1:  
            dQ1New = 1./L1*( Pin - R1*Q1 - P1 - KK1*(Q1**2./A1 - Q1**2./Ain) )
            # dQ1New = 1./L1*( Pin - R1*Q1 - Rd1*Q2 - Pstar - KK1*(Q2**2./Astar - Q1**2./Ain) )
            # dQ1New = 1./L1*( Pin - R1*Q1 - P1 - KK1*(Q2**2./A1 - Q1**2./Ain) )
        
        dV2New = ( Q2 - Qout )
        
        if switchKK==0:
            dQ2New = 1./L2*( Pstar - R2*Q2 - P2 ) 
        elif switchKK==1:  
            dQ2New = 1./L2*( Pstar - R2*Q2 - P2 - KK2*(Q2**2./A2 - Q2**2./Astar) )
            # dQ2New = 1./L2*( Pstar - R2*Q2 - Rd2*Qout - Pout - KK2*(Qout**2./Aout - Q2**2./Astar) )
            # dQ2New = 1./L2*( Pstar - R2*Q2 - P2 - KK2*(Qout**2./A2 - Q2**2./Astar) )
        
        dQsol = [ dV1New, dQ1New, dV2New, dQ2New ]
        return dQsol
    
    
    def XQinPoutVess_SPLIT(self, dt, Qin, Pout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with two (Qin, Pout)-elements in series --> x = [V1, Q1, V2, Q2]
        """
        V1 = Qsol[0]
        Q1 = Qsol[1]
        # P1 = Qsol[2]
        V2 = Qsol[3]
        Q2 = Qsol[4]
        # P2 = Qsol[5]
        x = [ V1, Q1, V2, Q2 ]
        # x = [ Qsol[0], Qsol[1], Qsol[3], Qsol[4] ]
        return x
    
    def QinPoutVessState_SPLIT(self, x, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with two (Qin, Pout)-elements in series --> Qsol = [V1, Q1, P1, V2, Q2, P2]
        """
        V1 = x[0]
        Q1 = x[1]
        V2 = x[2]
        Q2 = x[3]
        
        if nonlinear==0: # linear model
            C1 = 0.5*C0
            C2 = 0.5*C0
            P1 = (V1-0.5*V0A)/C1 + P0A + self.mod.Pe
            P2 = (V2-0.5*V0A)/C2 + P0A + self.mod.Pe     
        elif nonlinear==1: # nonlinear model
            A1 = self.meanA(V1, 0.5*self.length)
            P1 = self.mod.pFa(A1, self.K, self.a0, self.m, self.n, self.P0)
            A2 = self.meanA(V2, 0.5*self.length)
            P2 = self.mod.pFa(A2, self.K, self.a0, self.m, self.n, self.P0)
        
        Qsol = [ V1, Q1, P1, V2, Q2, P2 ]
        return Qsol
    
    def DQinPoutVess_SPLIT(self, dt, Qin, Pout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear, i, switchKK):
        """
        0D vessel with two (Qin, Pout)-elements in series --> dQsol = [dV1/dt, dQ1/dt, dV2/dt, dQ2/dt]
        """
        V1 = Qsol[0]
        Q1 = Qsol[1]
        P1 = Qsol[2]
        V2 = Qsol[3]
        Q2 = Qsol[4]
        P2 = Qsol[5]
        
        if nonlinear==0: # linear model
            C1 = 0.5*C0
            C2 = 0.5*C0
            R1 = 0.25*R0
            Rp1 = 0.25*R0
            R2 = 0.25*R0
            Rp2 = 0.25*R0
            L1 = 0.5*L0
            L2 = 0.5*L0
            Pstar = P2 + Rp2*Q1
            if switchKK==1:
                KK1 = KK0
                KK2 = KK0
                A1 = self.linearAfromP(P1, 0.5*self.length, C1)
                Astar = self.linearAfromP(Pstar, 0.5*self.length, C2)
                A2 = self.linearAfromP(P2, 0.5*self.length, C2)
                Aout = self.linearAfromP(Pout, 0.5*self.length, C2)
            elif switchKK==0:
                KK1 = 0.
                KK2 = 0.        
        elif nonlinear==1: # nonlinear model
            L1 = self.NLinductance(V1, 0.5*self.length)
            L2 = self.NLinductance(V2, 0.5*self.length)
            R1 = 0.5*self.NLresistance(V1, 0.5*self.length)
            Rp1 = R1
            R2 = 0.5*self.NLresistance(V2, 0.5*self.length)
            Rp2 = R2
            Pstar = P2 + Rp2*Q1
            if switchKK==1:
                KK1 = self.NLconvection(V1, 0.5*self.length)
                KK2 = self.NLconvection(V2, 0.5*self.length)
                A1 = self.mod.aFp(P1, self.K, self.a0, self.m, self.n, self.P0)
                Astar = self.mod.aFp(Pstar, self.K, self.a0, self.m, self.n, self.P0)
                A2 = self.mod.aFp(P2, self.K, self.a0, self.m, self.n, self.P0)
                Aout = self.mod.aFp(Pout, self.K, self.a0, self.m, self.n, self.P0)
            elif switchKK==0:
                KK1 = 0.
                KK2 = 0.
         
        dV1New = ( Qin - Q1 )
        
        if switchKK==0:
            dQ1New = 1./L1*( P1 - R1*Q1 - Pstar )
        elif switchKK==1:
            dQ1New = 1./L1*( P1 - R1*Q1 - Pstar - KK1*(Q1**2./Astar - Q1**2./A1) )
        
        dV2New = ( Q1 - Q2 )
        
        if switchKK==0:
            dQ2New = 1./L2*( P2 - R2*Q2 - Pout )
        elif switchKK==1:
            dQ2New = 1./L2*( P2 - R2*Q2 - Pout - KK2*(Q2**2./Aout - Q2**2./A2) )
        
        dQsol = [ dV1New, dQ1New, dV2New, dQ2New ]
        return dQsol
    
    def XPinPoutVess(self, dt, Pin, Pout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Pin, Pout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        # P = Qsol[2]
        Qd = Qsol[3]
        x = [ V, Q, Qd ]
        return x
    
    def PinPoutVessState(self, x, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Pin, Pout)
        """
        V = x[0]
        Q = x[1]
        Qd = x[2]
        
        if nonlinear==0: # linear model
            C = C0
            P = (V-V0A)/C + P0A + self.mod.Pe     
        elif nonlinear==1: #nonlinear model
            A = self.meanA(V, self.length)
            P = self.mod.pFa(A, self.K, self.a0, self.m, self.n, self.P0)

        Qsol = [ V, Q, P, Qd ] 
        return Qsol
    
    def DPinPoutVess(self, dt, Pin, Pout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear, i, switchKK):
        """
        0D vessel with (Pin, Pout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        P = Qsol[2]
        Qd = Qsol[3]
        
        if nonlinear==0: # linear model
            C = C0
            R = R0
            L = L0
            if switchKK==1:
                KK = KK0
                Ain = self.linearAfromP(Pin, self.length, C)
                A = self.linearAfromP(P, self.length, C)
                Aout = self.linearAfromP(Pout, self.length, C)
            elif switchKK==0:
                KK = 0.
        elif nonlinear==1: # nonlinear model
            L = self.NLinductance(V, self.length)
            R = self.NLresistance(V, self.length)
            if switchKK==1:
                KK = self.NLconvection(V, self.length)
                Ain = self.mod.aFp(Pin, self.K, self.a0, self.m, self.n, self.P0)
                A = self.mod.aFp(P, self.K, self.a0, self.m, self.n, self.P0)
                Aout = self.mod.aFp(Pout, self.K, self.a0, self.m, self.n, self.P0)
            elif switchKK==0:
                KK = 0.
        
        dVNew = ( Q - Qd )
        
        if switchKK==0:
            dQNew = 1./(0.5*L)*( Pin - 0.5*R*Q - P )
            dQdNew = 1./(0.5*L)*( P - 0.5*R*Qd - Pout )
        elif switchKK==1:  
            # dQNew = 1./(0.5*L)*( Pin - 0.5*R*Q - P - KK*(Qd**2./A - Q**2./Ain) ) 
            dQNew = 1./(0.5*L)*( Pin - 0.5*R*Q - P - KK*(Q**2./A - Q**2./Ain) )
            # dQNew = 1./(0.5*L)*( Pin - 0.5*R*Q - P - KK*(Qd**2./A - Q**2./A) )
            
            # dQdNew = 1./(0.5*L)*( P - 0.5*R*Qd - Pout - KK*(Qd**2./Aout - Q**2./A) ) 
            dQdNew = 1./(0.5*L)*( P - 0.5*R*Qd - Pout - KK*(Qd**2./Aout - Qd**2./A) )
            # dQdNew = 1./(0.5*L)*( P - 0.5*R*Qd - Pout - KK*(Qd**2./A - Q**2./A) )
        
        dQsol = [ dVNew, dQNew, dQdNew ]
        return dQsol
        
    def XQinQoutVess(self, dt, Qin, Qout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Qin, Qout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        # P = Qsol[2]
        Vd = Qsol[3]
        # Pd = Qsol[4]
        x = [ V, Q, Vd ]
        return x
    
    def QinQoutVessState(self, x, V0A, P0A, C0, R0, L0, KK0, nonlinear):
        """
        0D vessel with (Qin, Qout)
        """
        V = x[0]
        Q = x[1]
        Vd = x[2]
        
        if nonlinear==0: # linear model
            C = C0*0.5
            Cd = C0*0.5
            P = (V-0.5*V0A)/C + P0A + self.mod.Pe
            Pd = (Vd-0.5*V0A)/Cd + P0A + self.mod.Pe 
        elif nonlinear==1: # nonlinear model
            A = self.meanA(V, 0.5*self.length)
            P = self.mod.pFa(A, self.K, self.a0, self.m, self.n, self.P0)
            Ad = self.meanA(Vd, 0.5*self.length)
            Pd = self.mod.pFa(Ad, self.K, self.a0, self.m, self.n, self.P0)
            
        Qsol = [ V, Q, P, Vd, Pd ]
        return Qsol
    
    def DQinQoutVess(self, dt, Qin, Qout, Qsol, V0A, P0A, C0, R0, L0, KK0, nonlinear, i, switchKK, Ulimit, gamma):
        """
        0D vessel with (Qin, Qout)
        """
        V = Qsol[0]
        Q = Qsol[1]
        P = Qsol[2]
        Vd = Qsol[3]
        Pd = Qsol[4]
        
        if nonlinear==0: # linear model
            C = C0*0.5
            Cd = C0*0.5
            Rp = 0.5*R0
            R = 0.5*R0
            Rd = 0.
            # R = R0
            L = L0
            if switchKK==1:
                KK = KK0
                A = self.linearAfromP(P, 0.5*self.length, C)
                Ad = self.linearAfromP(Pd, 0.5*self.length, Cd)
            elif switchKK==0:
                KK = 0.       
        elif nonlinear==1: # nonlinear model
            Rp = self.NLresistance(V, 0.5*self.length) #*0.5
            R = self.NLresistance(Vd, 0.5*self.length) #*0.5
            Rd = 0.
            # R = self.NLresistance(V+Vd, self.length) 
            L = self.NLinductance(V+Vd, self.length) #*0.5
            if switchKK==1:
                KK = self.NLconvection(V+Vd, self.length)
                A = self.mod.aFp(P, self.K, self.a0, self.m, self.n, self.P0)
                Ad = self.mod.aFp(Pd, self.K, self.a0, self.m, self.n, self.P0)
            elif switchKK==0:
                KK = 0.


        dVNew = ( Qin - Q )
        
        if switchKK==0:
            dQNew = 1./L*( P - R*Q - Pd )
        elif switchKK==1:
            dQNew = 1./L*( P - R*Q - Pd - KK*(Q**2./Ad - Q**2./A) )
            # dQNew = 1./L*( Pin - Rp*Qin - R*Q - Rd*Qout - Pout - KK*(Qout**2./Aout - Qin**2./Ain) )
            # dQNew = 1./L*( Pin - (Rp+R+Rd)*Q - Pout - KK*(Qout**2./Aout - Qin**2./Ain) ) 
            # dQNew = 1./L*( Pin - R*Q - Pout - KK*(Qout**2./A - Qin**2./A) )  

        dVdNew = ( Q - Qout )
        
        dQsol = [ dVNew, dQNew, dVdNew ]
        return dQsol
    
    
    def Terminal0D(self, dt, Qin, Pout, Qsol, Rw1, Cw, Rw2):
        """
        RCR-Windkessel element attached to a (Pin, Qout)-type element to create 
        a special (Pin, Pout)-type element for terminal vessels
        Rtp : proximal terminal resistance
        Ct : terminal capacitor
        Rtd : distal terminal resistance
        The Rtp, Ct and Rtd are arranged IN SERIES 
        """
        Pw = Qsol[0]
        Qw = Qsol[1]
        
        PwNew = Pw + dt/Cw*( Qin - Qw ) # first-order explicit Euler method
        QwNew = ( PwNew - Pout )/Rw2        
        
        Qsol = [ PwNew, QwNew ]
        return Qsol
    
    def DPTerminal0D(self, dt, Qin, Pout, Pw, Rw1, Cw, Rw2):     
        dPwNew = ( Qin - (Pw - Pout)/Rw2 )/Cw
        return dPwNew
    
