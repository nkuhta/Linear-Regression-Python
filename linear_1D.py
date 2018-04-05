
import numpy as np
import matplotlib.pyplot as plt

# load data
x=[]
y=[]

for line in open('data_1d.csv'):
    xi,yi = line.split(',')
    #print(xi)
    x.append(float(xi))
    y.append(float(yi))

# turn into numpy arrays
x = np.array(x)
y = np.array(y)

#  plot raw data
plt.scatter(x,y)
#plt.show()

# calculate a and b (vectorized)

N = len(x)

# sum(xi^2) - first demoninator term
D1 = x.dot(x)
# 1/N * [sum(xi)]^2
D2 = x.mean()*x.sum()
# calculate a and b (vectorized)
a = (x.dot(y) - x.sum()*y.mean())/(D1-D2)
b = (D1*y.mean() - x.mean()*x.dot(y))/(D1-D2)

print('a = ',a)
print('b = ',b)

# best fit y=ax+b line
yhat = a*x+b

plt.scatter(x,y)
plt.plot(x,yhat)
plt.show()

#  R^2 value
SSres = (y-yhat).dot(y-yhat)
SStot = (y-y.mean()).dot(y-y.mean())
r_squared = 1 - SSres/SStot

print('r_squared = ',r_squared)
