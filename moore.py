
import re
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')
    xi = int(non_decimal.sub('',r[2].split('[')[0]))
    yi = int(non_decimal.sub('',r[1].split('[')[0]))
    x.append(xi)
    y.append(yi)

x = np.array(x)
y = np.array(y)

plt.scatter(x,y)
plt.show()

y = np.log(y)
plt.scatter(x,y)
plt.show()

# sum(xi^2) - first demoninator term
D1 = x.dot(x)
# 1/N * [sum(xi)]^2
D2 = x.mean()*x.sum()
# calculate a and b (vectorized)
a = (x.dot(y) - x.sum()*y.mean())/(D1-D2)
b = (D1*y.mean() - x.mean()*x.dot(y))/(D1-D2)

# best fit y=ax+b line
yhat = a*x+b

plt.scatter(x,y)
plt.plot(x,yhat)
plt.show()

#  R^2 value
SSres = (y-yhat).dot(y-yhat)
SStot = (y-y.mean()).dot(y-y.mean())
r_squared = 1 - SSres/SStot

print('a = ', a, "b = ", b)
print('r_squared = ',r_squared)
print('time to double (years) = ',np.log(2)/a)

import linear_model as lm
print(lm.coeff_1D(x,y))
print(lm.r_squared(y,yhat))
