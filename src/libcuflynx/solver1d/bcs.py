#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:16:49 2024

@author: bghi639
"""

import math

class BCs:
    
    def __init__(self, T0):
        self.T0 = T0
        
    def inflowUpperThorAo(self, t):
        """
        Aortic Bifurcation Inflow curve
        """
        PI = 3.141592653589793
        # T = 0.9550 # period 

        inflow = 500.*(0.20617+0.37759*math.sin(2*PI*t/self.T0+0.59605)+0.2804*math.sin(4*PI*t/self.T0-0.35859)+0.15337*math.sin(6*PI*t/self.T0-1.2509)-0.049889*math.sin(8*PI*t/self.T0+1.3921)+0.038107*math.sin(10*PI*t/self.T0-1.1068)-0.041699*math.sin(12*PI*t/self.T0+1.3985)-0.020754*math.sin(14*PI*t/self.T0+0.72921)+0.013367*math.sin(16*PI*t/self.T0-1.5394)-0.021983*math.sin(18*PI*t/self.T0+0.95617)-0.013072*math.sin(20*PI*t/self.T0-0.022417)+0.0037028*math.sin(22*PI*t/self.T0-1.4146)-0.013973*math.sin(24*PI*t/self.T0+0.77416)-0.012423*math.sin(26*PI*t/self.T0-0.46511)+0.0040098*math.sin(28*PI*t/self.T0+0.95145)-0.0059704*math.sin(30*PI*t/self.T0+0.86369)-0.0073439*math.sin(32*PI*t/self.T0-0.64769)+0.0037006*math.sin(34*PI*t/self.T0+0.74663)-0.0032069*math.sin(36*PI*t/self.T0+0.85926)-0.0048171*math.sin(38*PI*t/self.T0-1.0306)+0.0040403*math.sin(40*PI*t/self.T0+0.28009)-0.0032409*math.sin(42*PI*t/self.T0+1.202)-0.0032517*math.sin(44*PI*t/self.T0-0.93316)+0.0029112*math.sin(46*PI*t/self.T0+0.21405)-0.0022708*math.sin(48*PI*t/self.T0+1.1869)-0.0021566*math.sin(50*PI*t/self.T0-1.1574)+0.0025511*math.sin(52*PI*t/self.T0-0.12915)-0.0024448*math.sin(54*PI*t/self.T0+1.1185)-0.0019032*math.sin(56*PI*t/self.T0-0.99244)+0.0019476*math.sin(58*PI*t/self.T0-0.059885)-0.0019477*math.sin(60*PI*t/self.T0+1.1655)-0.0014545*math.sin(62*PI*t/self.T0-0.85829)+0.0013979*math.sin(64*PI*t/self.T0+0.042912)-0.0014305*math.sin(66*PI*t/self.T0+1.2439)-0.0010775*math.sin(68*PI*t/self.T0-0.79464)+0.0010368*math.sin(70*PI*t/self.T0-0.0043058)-0.0012162*math.sin(72*PI*t/self.T0+1.211)-0.00095707*math.sin(74*PI*t/self.T0-0.66203)+0.00077733*math.sin(76*PI*t/self.T0+0.25642)-0.00092407*math.sin(78*PI*t/self.T0+1.3954)-0.00079585*math.sin(80*PI*t/self.T0-0.49973))
        
        return inflow
        
    def inflowAoBif(self, t):
        """
        Aortic Bifurcation Inflow curve
        """
        PI = 3.141592653589793
        # T = 1.1 # period 

        inflow = 10e5*(7.9853e-06 + 2.6617e-05*math.sin(2*PI*t/self.T0+0.29498) + 2.3616e-05*math.sin(4*PI*t/self.T0-1.1403) +
                       - 1.9016e-05*math.sin(6*PI*t/self.T0+0.40435) - 8.5899e-06*math.sin(8*PI*t/self.T0-1.1892) +
                       - 2.436e-06*math.sin(10*PI*t/self.T0-1.4918) + 1.4905e-06*math.sin(12*PI*t/self.T0+1.0536) +
                       + 1.3581e-06*math.sin(14*PI*t/self.T0-0.47666) - 6.3031e-07*math.sin(16*PI*t/self.T0+0.93768) +
                       - 4.5335e-07*math.sin(18*PI*t/self.T0-0.79472) - 4.5184e-07*math.sin(20*PI*t/self.T0-1.4095) +
                       - 5.6583e-07*math.sin(22*PI*t/self.T0-1.3629) + 4.9522e-07*math.sin(24*PI*t/self.T0+0.52495) +
                       + 1.3049e-07*math.sin(26*PI*t/self.T0-0.97261) - 4.1072e-08*math.sin(28*PI*t/self.T0-0.15685) +
                       - 2.4182e-07*math.sin(30*PI*t/self.T0-1.4052) - 6.6217e-08*math.sin(32*PI*t/self.T0-1.3785) +
                       - 1.5511e-07*math.sin(34*PI*t/self.T0-1.2927) + 2.2149e-07*math.sin(36*PI*t/self.T0+0.68178) +
                       + 6.7621e-08*math.sin(38*PI*t/self.T0-0.98825) + 1.0973e-07*math.sin(40*PI*t/self.T0+1.4327) +
                       - 2.5559e-08*math.sin(42*PI*t/self.T0-1.2372) - 3.5079e-08*math.sin(44*PI*t/self.T0+0.2328))
        
        return inflow
    
    def inflowLinearIncrFun(self, t, inflowFun):
        k = 1
        q0 = inflowFun(0.)
        if t<=k*self.T0:
            inflow = t/(k*self.T0)*q0
        elif (t>k*self.T0 and t<=(k+1)*self.T0):
            inflow = q0
        else:
            tLoc = t - math.floor(t/self.T0)*self.T0
            inflow = inflowFun(tLoc)
            
        return inflow