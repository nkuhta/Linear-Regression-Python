import numpy as np

#  Input np x and y arrays
#  return output (a,b) for best-fit y = a*x + b  line
def coeff_1D(x,y):
    # sum(xi^2) - first demoninator term
    D1 = x.dot(x)
    # 1/N * [sum(xi)]^2
    D2 = x.mean()*x.sum()
    # calculate a and b (vectorized)
    a = (x.dot(y) - x.sum()*y.mean())/(D1-D2)
    b = (D1*y.mean() - x.mean()*x.dot(y))/(D1-D2)
    # linear (slope,intercept) coefficients
    return (a,b)

#  Input np x and y arrays
#  add_column == True adds the extra column of 1's
#  return output weights for best-fit of multi-dimensional
def coeff_multi(x,y,add_column):
    if add_column == True:
        #  if you need to add the first 1's columns
        x = np.concatenate((np.ones(len(y))[:, np.newaxis], x), axis=1)
    #  calculate the weights
    w = np.linalg.solve(np.dot(x.T,x), np.dot(x.T,y))
    return w


#  Input np y and y-hat (prediction) arrays
#  return r-squared quality of fit metric
def r_squared(y,yhat):
    SSres = (y-yhat).dot(y-yhat)
    SStot = (y-y.mean()).dot(y-y.mean())
    #  R^2 value
    result = 1 - SSres/SStot
    return result

#  Input numpy x/y feature/target arrays
#  alpha = learning rate (gradient stepsize)
#  steps = number of gradient descent steps
def grad_desc(x,y,alpha,steps,add_column):
    if add_column == True:
        #  if you need to add the first 1's columns (constant coeff term)
        x = np.concatenate((np.ones(len(y))[:, np.newaxis], x), axis=1)
    # initialize weights
    D = x.shape[1]
    np.random.seed(101)
    w = np.random.randn(D)/np.sqrt(D)
    for t in range(steps):
        yhat = x.dot(w)
        delta = yhat-y
        w = w - alpha*x.T.dot(delta)

    # weights calculated from gradient descent
    return w
