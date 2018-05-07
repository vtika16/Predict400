# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:03:21 2018

@author: vtfx3
"""
import math as math
import numpy as np
from scipy.integrate import quad
import scipy.special as special

def integrand(t,x):
    sqr = math.sqrt(2.24**2 - (19.62*t*(2.24*math.sin(x))) + (92.6321 * t ** 2))
    return sqr
 

for a in np.arange(30,61,1):
    print(a) 
    ans, err = quad(integrand,
                   float(0.0),
                   float(2.0),
                   (a))
    print(round(ans,2))