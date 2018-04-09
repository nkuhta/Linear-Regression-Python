
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#load data
x = []
y = []
for line in open('data_2d.csv'):
    x1,x2,yi = line.split(',')
    x.append([1.0,float(x1),float(x2)])
    y.append(float(yi))

#  turn x and y into numpy arrays
x = np.array(x)
y = np.array(y)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x[:,0],x[:,1],y)
plt.show()

#  calculate the weights
w = np.linalg.solve(np.dot(x.T,x), np.dot(x.T,y))
#  prediction
yhat = np.dot(x,w)

#  compute R^2
import linear_model as lm
print("r_squared: ",lm.r_squared(y,yhat))
print("weight coefficients: ",w)
