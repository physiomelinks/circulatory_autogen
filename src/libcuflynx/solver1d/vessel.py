#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:27:24 2019

@author: beatriceghitti
"""

import numpy as np
from scipy.interpolate import interp1d


class vessel:
    """
    Class for a 1D vessel and its discretization
    """
    def __init__(self, length, nCells, nVar, leftBCtype, rightBCtype, vessType, K, a0, m, n, P0, mu, mod, inflowDataFile, bcFunc, outFolder, saveResSpace, idxV, nameV=None):
        
        self.length = length # vessel length
        
        self.nCells = nCells # number of computational cells
        self.nVar = nVar # number of state variables
        
        self.Q = np.zeros((nCells, nVar)) # state variables discretization
        
        self.Qcc = [[], [], []]
        self.Qspace_time = []
        
        self.dx = self.length/float(nCells) # mesh size
        
        self.dt = 0. # time step, initialized to zero
        
        self.x = np.linspace(self.dx/2., self.length-self.dx/2., self.nCells) # vector which contains the center of each cell        
        self.xInt = np.linspace(0., self.length, self.nCells+1) # vector which contains cells' interfaces
        
        self.QBCL = np.zeros(self.nVar) # auxiliary state variables vector for LEFT BCs
        self.QBCR = np.zeros(self.nVar) # auxiliary state variables vector for RIGHT BCs
        self.leftBCtype = leftBCtype
        self.rightBCtype = rightBCtype
        self.vessType = vessType
        
        self.FnumQBCL = np.zeros(self.nVar) # auxiliary vector for LEFT boundary interface numerical flux
        self.FnumQBCR = np.zeros(self.nVar) # auxiliary vector for RIGHT boundary interface numerical flux

        self.FnumQBCL_old = np.zeros(self.nVar) # auxiliary vector for LEFT boundary interface numerical flux
        self.FnumQBCR_old = np.zeros(self.nVar) # auxiliary vector for RIGHT boundary interface numerical flux
        self.FnumQBCL_new = np.zeros(self.nVar) # auxiliary vector for LEFT boundary interface numerical flux
        self.FnumQBCR_new = np.zeros(self.nVar) # auxiliary vector for RIGHT boundary interface numerical flux

        # model/tube law parameters
        self.m = m
        self.n = n
        self.K = K # vessel wall stiffness
        self.a0 = a0 # reference/baseline vessel cross-sectional area
        self.P0 = P0 # reference/baseline pressure for which A=a0

        self.V = self.a0*self.length # vessel volume
        
        self.mod = mod # blood flow model
        
        self.mu = mu # "vessel-specific" blood dynamic viscosity
        self.rho = self.mod.rho # constant blood density
        
        self.bcFunc = bcFunc
        
        if (self.leftBCtype=='inflow' or self.leftBCtype=='inflow_o2'):
            if inflowDataFile==None:
                if not self.rightBCtype.startswith('junc'):
                    self.inflow = self.bcFunc.inflowUpperThorAo
                else:
                    self.inflow = self.bcFunc.inflowAoBif
            else:
                inflowData = np.genfromtxt(inflowDataFile)
                tData = inflowData[:,0]
                qData = inflowData[:,1]
                self.inflow = interp1d(tData, qData, kind='linear', bounds_error=False, fill_value='extrapolate')
                # self.inflow = interp1d(tData, qData, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                # self.inflow = interp1d(tData, qData, kind='cubic', bounds_error=False, fill_value='extrapolate')

        if nameV is None:
            self.outFile1 = open(outFolder+'sol1D_vess'+str(idxV)+'.txt','w')
        else:
            self.outFile1 = open(outFolder+'sol1D_'+str(nameV)+'.txt','w')
        self.outFile1.write("# 0: t[s]; 1: Qprox[ml/s]; 2: Pprox[dyne/cm2]; 3: Aprox[cm2]; 4: Qmid[ml/s]; 5: Pmid[dyne/cm2]; 6: Amid[cm2]; 7: Qdist[ml/s]; 8: Pdist[dyne/cm2]; 9: Adist[cm2]; 10: Qint[ml/s]; 11: Pint[dyne/cm2]; 12: Aint[cm2]; \n")
        self.outFile1.flush()
        
        if saveResSpace:
            if nameV is None:
                self.outFileQ = open(outFolder+'space/solQ_vess'+str(idxV)+'.txt','w')
                self.outFileP = open(outFolder+'space/solP_vess'+str(idxV)+'.txt','w')
                self.outFileA = open(outFolder+'space/solA_vess'+str(idxV)+'.txt','w')
            else:
                self.outFileQ = open(outFolder+'space/solQ_'+str(nameV)+'.txt','w')
                self.outFileP = open(outFolder+'space/solP_'+str(nameV)+'.txt','w')
                self.outFileA = open(outFolder+'space/solA_'+str(nameV)+'.txt','w')
            
            self.outFileQ.write("# 0: t[s]; 1,...,nCells: Q[ml/s]; \n")
            self.outFileQ.flush()
            
            self.outFileP.write("# 0: t[s]; 1,...,nCells: P[dyne/cm2]; \n")
            self.outFileP.flush()
            
            self.outFileA.write("# 0: t[s]; 1,...,nCells: A[cm2]; \n")
            self.outFileA.flush()
        
        # self.outFile2 = open(outFolder+'/errors_vess'+str(idxV)+'.txt','w')
        # self.outFile2.write("# 0: Linf-Q; 1: Linf-P; 2: Linf-A;\n")
        # self.outFile2.flush()
        
    def setInitialConditions(self, ICtype, PAinit, Qinit, AICstate, QICstate, ICstate=-1):
        if ICstate==-1:
            if ICtype==0:
                pIC = 0.
                aIC = self.mod.aFp(pIC, self.K, self.a0, self.m, self.n, self.P0)
            elif ICtype==1 or ICtype==3:
                pIC = PAinit
                aIC = self.mod.aFp(pIC, self.K, self.a0, self.m, self.n, self.P0)
            elif ICtype==2:
                pIC = self.P0 # [dyne/cm^2] initial pressure
                aIC = self.a0 # [cm^2] initial cross-sectional area
            self.Q[:,0] = aIC
            self.Q[:,1] = Qinit
        elif ICstate==1:
            self.Q[:,0] = np.copy(AICstate)
            self.Q[:,1] = np.copy(QICstate)  

        self.FnumQBCL_old[:] = self.mod.pF(self.Q[0,:], self.K, self.a0, self.m, self.n)
        self.FnumQBCR_old[:] = self.mod.pF(self.Q[-1,:], self.K, self.a0, self.m, self.n)
        self.FnumQBCL_new[:] = np.copy( self.FnumQBCL_old[:] )
        self.FnumQBCR_new[:] = np.copy( self.FnumQBCR_old[:] )


    def computeSolIntAverage(self): #XXX to be improved with higher-order integration rules, especially if FVord>1
        Aint = self.dx/self.length*np.sum( self.Q[:,0] )
        Qint = self.dx/self.length*np.sum( self.Q[:,1] )
        
        Atmp = np.copy(self.Q[:,0])
        Ptmp = self.mod.pFa(Atmp, self.K, self.a0, self.m, self.n, self.P0)
        Pint = self.dx/self.length*np.sum( Ptmp )
        
        return Aint, Qint, Pint       
            
    def sampleResultsTime(self):
        QNum = np.zeros((3*4))
        
        # first cell
        QNum[0] =  self.Q[0,1]
        QNum[1] =  self.mod.pFa(self.Q[0,0], self.K, self.a0, self.m, self.n, self.P0)
        QNum[2] =  self.Q[0,0]
        
        # midpoint
        # Atmp = np.copy(self.Q[:,0])
        # Ptmp = self.mod.pFa(Atmp, self.K, self.a0, self.m, self.n, self.P0)
        # Mloc = int(self.nCells)
        if self.nCells % 2 == 0:
            QNum[3] = 0.5*( self.Q[int(float(self.nCells)/2-1), 1] + self.Q[int(float(self.nCells)/2), 1] )
            QNum[5] = 0.5*( self.Q[int(float(self.nCells)/2-1), 0] + self.Q[int(float(self.nCells)/2), 0] )
            QNum[4] = self.mod.pFa(QNum[5], self.K, self.a0, self.m, self.n, self.P0)
        else:
            QNum[3] = self.Q[int(float(self.nCells-1)/2), 1]
            QNum[5] = self.Q[int(float(self.nCells-1)/2), 0]
            QNum[4] = self.mod.pFa(QNum[5], self.K, self.a0, self.m, self.n, self.P0)
        
        # last cell
        QNum[6] =  self.Q[-1,1]
        QNum[7] =  self.mod.pFa(self.Q[-1,0], self.K, self.a0, self.m, self.n, self.P0)
        QNum[8] =  self.Q[-1,0]
        
        # integral average
        QNum[11], QNum[9], QNum[10] = self.computeSolIntAverage()
        
        return QNum
        
    def outputResults(self, time):
        Qnum = self.sampleResultsTime()
        
        self.outFile1.write("%.18e " % (time))
        
        for i in range(len(Qnum)):
            self.outFile1.write("%.18e " % (Qnum[i]))
        
        self.outFile1.write("\n")
        self.outFile1.flush()
        
    def outputResultsSpace(self, time):
        self.outFileQ.write("%.18e " % (time))
        self.outFileP.write("%.18e " % (time))
        self.outFileA.write("%.18e " % (time))
        
        for i in range(self.nCells):
            Ptmp = self.mod.pFa(self.Q[i,0], self.K, self.a0, self.m, self.n, self.P0)
            self.outFileQ.write("%.18e " % (self.Q[i,1]))
            self.outFileP.write("%.18e " % (Ptmp))
            self.outFileA.write("%.18e " % (self.Q[i,0]))
        
        self.outFileQ.write("\n")
        self.outFileQ.flush()
        self.outFileP.write("\n")
        self.outFileP.flush()
        self.outFileA.write("\n")
        self.outFileA.flush()
        
    # def outputErrors(self, errQ, errP, errA):
    #     self.outFile2.write("%.18e %.18e %.18e \n" % (errQ, errP, errA))
    #     self.outFile2.flush()
    
