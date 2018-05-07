
##imported numpy
import numpy as np

#created array of two datasets
x = np.array([1,2,3,4,5,6,7,8])
y = np.array([190,185,183,183,180,179,178,178])


#create xy, x^2 and y^2

n = len(x)
xy = x * y
xsquared = np.square(x)
ysquared = np.square(y)

sumx = np.sum(x)
sumy = np.sum(y)
sumxy = np.sum(xy)
sumxsquared = np.sum(xsquared)
sumysquared = np.sum(ysquared)

#create 'm' for slope and 'b' for intercept. for 'm' I got -1.60 and for 'b' I got 189.18

m = ((n*sumxy)-(sumx*sumy))/((n*sumxsquared)-(sumx**2))
b = (sumy - (m*sumx))/n

print(m,b)

##now using numpy's great tools, I tried using lstsq function which is their least squared function lol. Got the same results
Xstack = np.vstack([x, np.ones(len(x))]).T

m, b = np.linalg.lstsq(Xstack, y)[0]

print(m, b)

##finding correlation coefficient by using functions above. for 'r' I got -0.944

import math as math

num = ((n*sumxy) - (sumx*sumy)) 

den = (math.sqrt((n*sumxsquared) - (sumx**2))) * (math.sqrt((n * sumysquared) - (sumy ** 2)))

r = num / den

#time to plot the instance
fit = np.polyfit(x, y, 1)
fit_fn= np.poly1d(fit)

import matplotlib.pyplot as plt
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.xlim(.5, 8.5)
plt.ylim(150, 195)