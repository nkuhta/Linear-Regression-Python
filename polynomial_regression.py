
import numpy as np
import matplotlib.pyplot as plt

# load the data

x = []
y = []
for line in open('data_poly.csv'):
    xi,yi = line.split(',')
    xi = float(xi)
    x.append([1,xi,xi**2])
    y.append(float(yi))

#  convert to numpy
x = np.array(x)
y = np.array(y)

# plot data
plt.scatter(x[:,1],y)
plt.show()

# calculate the weights
w = np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))
yhat = np.dot(x,w)

print("poly weights:", w)

# plot everything together
plt.scatter(x[:,1],y)
plt.plot(sorted(x[:,1]),sorted(yhat))
plt.show()

# r - squared
SSres = (y-yhat).dot(y-yhat)
SStot = (y-y.mean()).dot(y-y.mean())
#  R^2 value
r_squared = 1 - SSres/SStot

print("r_squared: ",r_squared)

#check library
# import linear_model as lm
# w_coeff = lm.coeff_multi(x,y,False)
# print(w_coeff)
# print('r^2: ',lm.r_squared(y,yhat))
