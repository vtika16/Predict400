# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:28:48 2018

@author: vtfx3
"""

from numpy import *

print ('\nMinimization Problem')
x=[2.5,4.5,5]
y=[5,3,10]
obj= matrix([54,60])
obj= transpose(obj)
corners= matrix([x,y])
corners= transpose(corners)
result= dot(corners,obj)
print ('Value of Objective Function at Each Corner Point\n'), result




import sympy as sympy
from sympy import Matrix
from sympy.abc import x,y,z
from sympy.solvers.solvers import solve_linear_system_LU

solve_linear_system_LU(Matrix([
[2.5,4.5,54],
[5,3,60]]), [x,y])






import sympy as sympy
from sympy import Matrix
from sympy.abc import x,y,z
from sympy.solvers.solvers import solve_linear_system_LU

solve_linear_system_LU(Matrix([
[2.5,4.5,5],
[5,3,10],
[.08,.09,.10]
]), [x,y,z])
-----------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import linprog
c = np.array([.08,.09,.10])
A = np.array([[-2.5,-4.5,-5.0],[-5.0,-3.0,-10.0],[-1,0,0],[0,-1,0],[0,0,-1]])
b = np.array([-54,-60,0,0,0]) 

x0_bnds = (None, None)
x1_bnds = (None, None)

res = linprog(c,A,b,bounds=(0,None))
print(res)
-------------------------------------------------------------------------------

def fun(x):
    return [(int(2.5)*x[1]) + (int(4.5)*x[2])) - 54,
            (int(5.0)*x[1]) + (int(3.0)*x[2]) - 60]


from scipy import optimize
sol = optimize.root(fun, [0, 0,], jac=None, method='hybr')
sol.x


###next what we can do is determine the system of equations, assuming each variable(s)=0
import sympy as sympy
from sympy import Matrix
from sympy.abc import x,y,z
from sympy.solvers.solvers import solve_linear_system_LU

solve_linear_system_LU(Matrix([
    [4.5,5,54],
    [3,10,60]]), [y,z])

##We get y=8 and z=3.6, which is the answer to our first problem. if we plug that into the price function we get

p = .09*8+3.6*.1
print(p)


    
solve_linear_system_LU(Matrix([
[2.5,4.5,54],
[5,3,60]]), [x,y])
    
##We get x=7.2 and y=8 if we plug that into the pricing model we get:

p=.08*7.2+.09*8
print(p)

##This doesn't equal to our goal of 1.08, so we can move on

##we will look for x and z, assuming y is 0
solve_linear_system_LU(Matrix([
[2.5,5,54],
[5,10,60]]), [x,z])

##This is not feasible for this problem. we can move on to check variables independently, or their intercepts
##First, lets find our intercepts: 
 
x1 = 54/2.5
x2 = 60/5
y1 = 54/4.5
y2 = 60/3
z1 = 54/5
z2 = 60/10

##Now, let's input each intercept accordingly into the pricing model

px1 = .08*x1
px2 = .08*x2
py1 = .09*y1
py2 = .09*y2
pz1 = .1*z1
pz2 = .1*z2

print(px1)
print(px2) 
print(py1)
print(py2)
print(pz1)
print(pz2)

print('The second solution is buying ',y1, 'of raw meat to get ',py1, ', the same as the first solution')

solve_linear_system_LU(Matrix([
[2.5,4.5,54],
[5,3,60]]), [x,y])

    
    
##now let's try to find x and y when z is equal to 0

##2.5x+4.5y=54
a = np.array([2.5,4.5])

"""The first answer will be 1.08 when x2 = 8 and x3 = 3.6. for the second answer, """

c = (.08*x) + (.09*y) + (.1*z)

##constraints from above

import array
import numpy as np

x = array.array('i',(0 for i in range (0,10)))
y = array.array('i',(0 for i in range (0,10)))
z = array.array('i',(0 for i in range (0,10)))
xlim = np.linspace(0,10)
ylim = np.linspace(0,10)
zlim = np.linspace(0,10)
c = ((int(2.5)*x) + (int(4.5)*y) + (int(5)*z)) 

np.intc

v = ((int(5.0)*a) + (int(3.0)*b) + (int(10.0)*c))



# Make plot
plt.plot(x, y1, label=r'$y\geq2$')
plt.plot(, label=r'$2y\leq25-x$')
plt.plot(x, y3, label=r'$4y\geq 2x - 8$')
plt.plot(x, y4, label=r'$y\leq 2x-5$')
plt.xlim((0, 16))
plt.ylim((0, 11))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

# Fill feasible region
y5 = np.minimum(y2, y4)
y6 = np.maximum(y1, y3)
plt.fill_between(x, y5, y6, where=y5>y6, color='grey', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

----------------------------------------------------------------------------------
import numpy as np
a = np.array([[2.5,4.5,5], [5,3,10], [0,8,3.6]])
b = np.array([54,60,1.08])
x = np.linalg.solve(a, b)

print(x)
----------------------------------------------------------------------------------
import math as math

import matplotlib.pyplot as plt

for e in range(1, 90):
    i = int(10**(e/10.0))
    test(i)



def integrate(a, b, n):
    total = 0.0
    delta = (b - a) / n
    i = 0
    while i < n:
        total = total + delta * (f(a + delta * (i + 1)) + f(a + delta * i)) / 2.0
        i = i + 1
    return total

y = []
x = []
b = 4.0
a = 1.0
rng = [1,2,3,4,5,6]


for n in rng:
    area = integrate(a,b,n)
    x = x + [n]
    y = y + [area]
    

plt.xlim(0, max(x) + 10)
plt.ylim(min(y) - 0.5, max(y) + 0.5)

plt.plot(x, y)
plt.scatter(x, y, s=30, c='r')
plt.scatter(30, 6.0, c='y')  # This plots the limiting value for the area.

plt.xlabel('Number of Subintervals')
plt.ylabel('Estimated Area')
plt.title('Plot Showing Numerical Integration Convergence')
plt.show()

area = float(format(y[-1], '0.3f'))
print("Final Estimate of Area with %r subdivisions = {}".format(x[-1], area))


import numpy as np
import matplotlib.pyplot as plt
import math as math
from matplotlib import axes
import matplotlib.patches as mpatches

x = np.linspace(-1, 2, 100)
y = (x**2) * np.log(x) 

"""axes = plt.gca()
axes.Axes.set_ylim([-2,10])"""

plt.plot((1/math.sqrt(math.e)), -1/(2*math.e), marker='o', markersize=3, color="red")
plt.plot(x,y)
plt.xlabel('$x$')
plt.ylabel('$exp(x)$')
red_patch = mpatches.Patch(color='red', label='The Inflection Point')
plt.legend(handles=[red_patch])
plt.title('Graph of exponential function')

plt.annotate('minimum=(1/√e,-1/2e)', xy=((1/math.sqrt(math.e)), -1/(2*math.e)), xytext=(1, 1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

ax.set_ylim(-2,2)
plt.show()

-------------------------------------------------------------------------------------------
import sympy as sp
from scipy.integrate import quad


x = sp.Symbol('x')
sp.integrate(3.0*x**2+1, x)

def f(x):
    return 3.0*x**2+1

i = quad(f,0,2)

print(i[0])
print(err)

-------------------------------------------------------------------------------------------

import sympy as sp
from scipy.integrate import quad
import numpy as np

x = sp.Symbol('x')
first_method = sp.integrate((0.83*np.e**(0.0133*x)), (x,0,100))
print(round(first_method,2))

def f(x):
    return 0.83*np.e**(0.0133*x)

numerical_method = quad(f,0,100)

print(round(numerical_method[0],2))