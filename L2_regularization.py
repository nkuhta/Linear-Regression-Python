
import numpy as np
import matplotlib.pyplot as plt

# number of data points
N = 50
#  x is 50 data points from 0 to 10
X = np.linspace(0,10,N)
#  line plus some random noise
Y = 0.5*X + np.random.randn(N)

# set up outliers for last 2 points
Y[-1]+=30
Y[-2]+=30

#  plot raw data
plt.scatter(X,Y)
plt.show()

X = np.vstack([np.ones(N),X]).T
w_ml = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:,1],Y)
plt.scatter(X[:,1],Yhat_ml)
plt.show()

Lambda = 1000

w_map = np.linalg.solve(Lambda*np.eye(2)+X.T.dot(X),X.T.dot(Y))
Yhat_map = X.dot(w_map)

plt.scatter(X[:,1],Y,label='raw data')
plt.plot(X[:,1],Yhat_ml,label='maximum likelihood')
plt.plot(X[:,1],Yhat_map,label='L2 regularized')
plt.legend()
plt.show()

print(w_ml)
print(w_map)
