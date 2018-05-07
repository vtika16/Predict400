"""6.	Randall, a fitness trainer, has an exercise regimen that includes 
running, swimming, and walking. He has no more than 12 hours per week to devote
to exercise, including at most 4 hours running. He wants to walk at least three
times as many hours as he swims. Randall will burn on average 528 calories per
hour running, 492 calories per hour swimming, and 348 calories per hour walking.
Using Python, calculate how many hours per week Randall should spend on each
exercise to maximize the number of calories he burns, as well as the maximum 
number of calories he will burn.
"""


"""Determine variables: x1 = running, x2 = swimming, x3 = walking

also determine equations/constraints:
x1 + x2 +x3 => 12
x1 <= 4
x3 >= x2 * 3

528x1 + 492x2 + 348x3 = z """

import numpy as numpy
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, value, LpMaximize
#%matplotlib inline


# Solution using Pulp
# declare your variables
x1 = LpVariable("x1", 0, None) # x1>=0
x2 = LpVariable("x2", 0, None) # x2>=0
x3 = LpVariable("x3", 0, None) # x3>=0

# defines the problem
prob = LpProblem("problem", LpMaximize)

# defines the constraints
prob += x1+x2+x3 >= 12
prob += x1 <= 4
prob += x3 >= x2 * 3

# defines the objective function to maximize
prob += 528*x1 + 492*x2 + 348*x3

# solve the problem
status = prob.solve()

#print the results
print("Pulp Solution for x1, x2, x3")
print(value(x1), "hours running")
print(value(x2), "hours swimming")
print(value(x3), "hours walking")
print('Solution: {}'.format(prob.objective.value()))


(4 * 528) + (1*492) + (348 * 7)