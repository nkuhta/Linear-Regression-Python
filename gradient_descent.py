
import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N,D))
X[:,0]=1
X[:5,1]=1
X[5:,2]=1
#print(X)

Y = np.array([0]*5 + [1]*5)
#print(Y)
#  closed form solution (singular matrix this time)
#w = np.linalg.solve(X.T.dot(X),X.T.dot(Y))

# Use gradient descent instead
costs=[]
np.random.seed(101)
w = np.random.randn(D)/np.sqrt(D)
#  learning rate
alpha = 0.001

#print(Y-X.dot(w))
#print(X)

for t in range(1000):
    Yhat = X.dot(w)
    delta = Yhat-Y
    w = w - alpha*X.T.dot(delta)
    mse = delta.dot(delta)/N
    costs.append(mse)

plt.plot(costs)
plt.show()

plt.plot(Yhat,label='prediction')
plt.plot(Y,label='target')
plt.legend()
plt.show()

print(w)
import linear_model as lm
w_lib = lm.grad_desc(X,Y,alpha,1000,False)
print(w_lib)
