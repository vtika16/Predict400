# MSPA 400 Session #3 Python Module #3
import matplotlib.pyplot as plt
import numpy as np

'''
This is a minimization problem. One objective is to demonstrate plotting an
unbounded region. There are four inequalities: x+3y >= 15, 3x+y >= 10,
x >= 0, y >= 0. One inequality and the objective function have been
modified for this module. The objective function is z = 2x+3y. The feasible
region will be graphed and filled. Matrix methods will be used to evaluate
the objective function at each corner.

The x and y boundaries of 0 and 20 were arbitrarily chosen as they communicate what we want
with the graph. You will find yourself adjusting these ranges depending upon the needs
of the data you are trying to communicate
'''
x = np.arange(0, 20.1, 0.1)
y0 = np.arange(0, 20.1, 0.1)
y1 = 10.0-3.0*x
y2 = 5.0-x/3.0

'''
# The shaded region (feasible region) is what we want to fill in. We will create y3 and y4 variables and fill between
them. We Need to define a new list y4 which will be that maximum of y1 or y2 at each point along our x-axis. 
The only purpose of this is to fill in the shaded region. The shaded region's lower limit will be the maximum 
of y1 or y2. Its upper limit is infinity, but we want to cap it at 20 to match our graph's y upper bounds. 
'''

# The definition of y3 will allow filling the unbounded region in the plot. Setting to 20 for each point.
y3 = 20+0.0*x

y4 = list()  # Start as an empty list and add the greater of y1/y2 to it for each point
for y1_val, y2_val in zip(y1, y2):
    # Zip allows us to loop over both lists at the same time.
    y4_val = max([y1_val, y2_val])  # Get the greater of the two
    y4.append(y4_val)  # Add it to our list


# This is the objective function plotted for illustration.
y5 = 5.5-2.0 * x/3.0

# Plot limits must be set for the graph.
plt.xlim(0, 20)
plt.ylim(0, 20)

# Plot axes need to be labled,title specified and legend shown.
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.title('Shaded Area Shows the Feasible Region')

plt.plot(x, y2, color='b')
plt.plot(x, y1, color='r')

# This next step shows how to plot a line using different symbols.
plt.plot(x, y5, 'k--')
print('Note that the dashed line passes through the optimum point')

# The order of entry of labels in the plt.legend statement must follow
# the order of the plt.plot statements to match colors.
plt.legend(['x+3y >= 15', '3x+y >= 10', '16.875 = 2x+3y'])
plt.fill_between(x, y3, y4, color='grey', alpha=0.2)

# Corner points for evaluation using the objective function.
x = [0, 1.875, 15]
y = [10, 4.375, 0]

# The minimum
plt.plot([1.875], [4.375], marker='o', markersize=7, color="red")

plt.show()

# This next step shows how to use matrix calculations to evaluate
# the objective function at each corner point and find the maximum.

obj = np.matrix([2.0, 3.0])
obj = np.transpose(obj)
corners = np.matrix([x, y])
corners = np.transpose(corners)
result = np.dot(corners, obj)
print('Value of Objective Function at Each Corner Point\n', result)

"""
Exercise 1: Refer to Lial Chapter 4 Review exercise 26. Using scipy.optimize, write
the code to solve the system. Compare your code and solutions with the answer sheet.
"""