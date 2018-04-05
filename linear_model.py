
#  Input np x and y arrays
#  return output (a,b) for best-fit y = a*x + b  line

def lm_coeff(x,y):
    # sum(xi^2) - first demoninator term
    D1 = x.dot(x)
    # 1/N * [sum(xi)]^2
    D2 = x.mean()*x.sum()
    # calculate a and b (vectorized)
    a = (x.dot(y) - x.sum()*y.mean())/(D1-D2)
    b = (D1*y.mean() - x.mean()*x.dot(y))/(D1-D2)
    # linear (slope,intercept) coefficients
    return (a,b)

#  Input np y and y-hat (prediction) arrays
#  return r-squared quality of fit metric

def r_squared(y,yhat):
    SSres = (y-yhat).dot(y-yhat)
    SStot = (y-y.mean()).dot(y-y.mean())
    #  R^2 value
    result = 1 - SSres/SStot
    return result
