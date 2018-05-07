# -*- coding: cp1252 -*-
"""
1) 
Maximize z=10x1+15x2+10x3+5x4
Subject to:
x1+x2+x3+x4≤300
x1+2x2+3x3+x4≤360
x1≥0,x2≥0,x3≥0,x4≥0

Solution: Maximum is 3300 when x1=240,x2=60,x3=0,x4=0
"""

import scipy as scipy
from scipy.optimize import linprog
from pulp import LpVariable, LpProblem, LpMaximize, GLPK, LpStatus, value, LpMinimize

z = [5.59,4.56] 
lhs = [[.45,.2], [.3,.5],[.15,.35]]
rhs = [7,7,7]
x1_bounds = (0, None)
x2_bounds = (0, None)
x3_bounds = (0, None)
x4_bounds = (0, None)
x5_bounds = (0, None)
method='simplex'

res = linprog(c=z, A_ub=lhs, b_ub=rhs, 
              bounds=(x1_bounds,x2_bounds, x3_bounds, x4_bounds))

print('Scipy Optimize Optimal value:', res.fun, '\n x1, x2, x3, x4 :', res.x)
print('\n')





# Solution using Pulp


# declare your variables
x1 = LpVariable("x1", 0, None) # x1>=0
x2 = LpVariable("x2", 0, None) # x2>=0
x3 = LpVariable("x3", 0, None) # x3>=0

 
# defines the problem
prob = LpProblem("problem", LpMaximize)
 
# defines the constraints
prob += x1 + x2 + x3 + x4 <= 300
prob += x1 + 2*x2 + 3*x3 +x4 <= 360

# defines the objective function to maximize
prob += 10*x1 + 15*x2+ 10*x3 + 5*x4
 
# solve the problem
status = prob.solve(GLPK(msg=0))
LpStatus[status]
 
# print the results
print("Pulp Solution for x1, x2, x3 and x4")
print(value(x1))
print(value(x2))
print(value(x3))
print(value(x4))
print(value(x5))
"""
Minimize w=22y1+44y2+33y3
Subject to:
y1+2y2+y3≥3
y1+y3≥3
3y1+2y2+2y3≥8
y1≥0,y2≥0,y3≥0

Solution: Minimum is 66 when y1=3,y2=0,y3=0
"""
from scipy.optimize import linprog

w = [22, 44, 33] 
lhs = [[-1, -2, -1], [-1, 0,-1],[-3, -2, -2]]
rhs = [-3, -3, -8]
y1_bounds = (0, None)
y2_bounds = (0, None)
y3_bounds = (0, None)

res = linprog(c=w, A_ub=lhs, b_ub=rhs, 
              bounds=(y1_bounds,y2_bounds, y3_bounds))
print('\n')
print('Scipy Optimize Optimal value:', res.fun, '\n y1, y2, y3:', res.x)
print('\n')

# Solution using Pulp

# declare your variables
y1 = LpVariable("y1", 0, None) # y1>=0
y2 = LpVariable("y2", 0, None) # y2>=0
y3 = LpVariable("y3", 0, None) # y3>=0
 
# defines the problem
prob = LpProblem("problem", LpMinimize)
 
# defines the constraints
prob += y1 + 2*y2 + y3 >= 3
prob += y1 + y3 >= 3
prob += 3*y1 + 2*y2 + 2*y3 >= 8

# defines the objective function to maximize
prob += 22*y1 + 44*y2+ 33*y3
 
# solve the problem
status = prob.solve(GLPK(msg=0))
LpStatus[status]
 
# print the results
print("Pulp Solutions for y1, y2, and y3")
print(value(y1))
print(value(y2))
print(value(y3))


print ('\nMinimization Problem')
x=[.55,.3,.15]
y=[.3,.6,.75]
obj= matrix([5.59,4.56])
obj= transpose(obj)
corners= matrix([x,y])
corners= transpose(corners)
result= dot(corners,obj)
print ('Value of Objective Function at Each Corner Point\n'), result
print ('\nMaximization Problem')
x=[0,2,3.2]
y=[4,3,0]
obj= matrix([9.0,10.0])
obj= transpose(obj)
corners= matrix([x,y])
corners= transpose(corners)
result= dot(corners,obj)
print ('Value of Objective Function at Each Corner Point\n'), result


import sys
sys.exit()

import numpy as np
import fractions
from fractions import Fraction

probs = np.array([.26,.21,.11,.10,.04,.03,.03])

not_probs =np.array( 1 - probs)

total_probs = 1 - np.prod(not_probs)


r1 = 1/20
r2 = 3/5
r3 = 35/10

q_r1 = 7/10
q_r2 = 1/2
q_r3 = 2/5

r1_q = fractions.Fraction((r1*q_r1)/((r1*q_r1)+(r2*q_r2)+(r3*q_r3)))
print(r1_q)