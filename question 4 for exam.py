# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""4.	The manager of a local restaurant is expanding his dining room area.  
He will be adding square tables that seat four people and round tables that s
eat six people.  He has room for at least five new tables, but no more than 
ten.  He would also like at least three more square tables than round tables. 
Each square table costs $40 and each round table costs $60. Using Python, 
graph the feasible region and determine the number of each type of table he 
should purchase to minimize costs as well as that cost."""

import numpy as np
import matplotlib.pyplot as plt
figure()
x=arange(0,20,1)
y1 = -x + 5
y2 = -x + 10
y3 = x - 3
##determine our points

"""y4 = np.minimum(y2, y3)
plt.ylim(0,20)
plt.plot(x, y1,
         x, y2,
         x, y3)
         
plt.fill_between(x,y1,y4,y3, color='grey')
plt.show()"""

plot(x,y1,c='r',label='y=-x+5') ##red line
plot(x,y2,c='b',label='y=-x+10')##blue line
plot(x,y3,c='g',label='y=x-3') ##green line



legend(['y=-x+5','y=-x+10','y=x-3'])
fill_between(x,y1,y2,y3,color='blue',alpha='0.5')
#fill_between(x,y1,where=(x>=4 ),color='b')
fill_between(x,y1,where=(y>=1,x>=1,y<=6,x<=3,x<=10),color='b')
#fill_between(x,y1,where=(y1<=y3),color='b')
##fill_between(x,y2,where=(y2<=y3),color='b')
fill_between(x,y3,where=(y3>=y1),color='b')
title('Feasible Region Mapping')
show()

"""x = [4,5,6.5,10]
y = [0,1,3.5]
fill_betweenx()"""
##############################################################################
import numpy as numpy
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.optimize import linprog
import pulp
%matplotlib inline

"""let's denote some variables to determine our tables
x = number of square tables, y = number of round tables

lets determine the cost of each table

z = 40x + 60y"""

"""consider the constraints from the question. He has room for at least
5 tables but no more than 10. Also, there has to be at least 3 more square 
tables than round tables. we also know that there can be no less than 0 square
and round tables."""


#let's use pulp to solve this problem
x = LpVariable("x", 0, None) #x1=>0
y = LpVariable("y", 0, None) #x2=>0

#define problem
prob = LpProblem("problem", LpMinimize)

#define constraints
prob += x + y >= 5
prob += x + y <= 10
prob += x - y >=  3

##define problem to maximize
prob += 40*x + 60*y

#solve problem
status = prob.solve()


#print results
print("Pulp solution for x and y")
print(value(x))
print(value(y))
print('Solutions: {}'.format(prob.objective.value()))


#x and y greater than or equal to 0
x = numpy.linspace(0,400)
y = numpy.linspace(0,400)



##this equation is equal to x = y + 3
x1 = y + 3

#Make the plot
plt.plot(x, y)
plt.plot(x1, y)
plt.xlim((0.150))
plt.ylim((0,150))


#########################################################33

prob = pulp.LpProblem('Number of Tables', LpMinimize)

##set decision variables
squaretbls = pulp.LpVariable('Round Tables', lowBound=0, cat='Integer')
roundtbls = pulp.LpVariable('Square Tables', lowBound=0, cat='Integer')

total_tbl_cost = (40 * squaretbls) + (60 * roundtbls)

## add constraints
prob += (squaretbls + roundtbls >= 5)
prob += (squaretbls + roundtbls <= 10)
prob += (squaretbls - roundtbls >=  3)

##solve the LP using the equation below
optimization = prob.solve()

##assure we got optimal solution
assert optimization == pulp.LpStatusOptimal

for var in (squaretbls, roundtbls):
    print('Optimal weekly number of {} to produce: {:1.0f}'.format(var.name, var.value()))
    
    
    
    
    
    
import pylab as plt
import numpy as np

X=arange(0,20,1)
Y1 = np.array(-x + 5)
Y2 = np.array(-x + 10)
Y3 = np.array(x - 3)

plt.ylim(0,20)
xlim(0,20)
plt.plot(X,Y1,lw=4)
plt.plot(X,Y2,lw=4)
plt.plot(X,Y3,lw=4)

plt.fill_between(X, Y1,Y2,Y3,color='k',alpha=.5)
legend(['y=-x+5','y=-x+10','y=x-3'])

plt.show()   
    
    
    
    
    
    
    
    
    
    
    

