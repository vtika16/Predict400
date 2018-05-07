import sympy as sp
x = sp.Symbol('x')
sp.diff(3*x**2+1,x)


import math
from scipy.misc import derivative

def arc(x):
    

def f(t):
    return sqrt(2.24**2 - (19.62*t*(2.24*math.sin(30))) + (92.6321 * t ** 2))


print(derivative(f))


import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special

def integrand(t):
    sqr = sqrt(2.24**2 - (19.62*t*(2.24*math.sin(30))) + (92.6321 * t ** 2))
    print(sqr)
    return sqr
    
ans, err = quad(integrand,2.24,
                   lambda t: float(0.0),
                   lambda t: float(2.0))
    



result = integrate.quad(lambda x: special.jv(2.5,x),0,4.5)