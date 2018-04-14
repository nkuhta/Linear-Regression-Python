
#  Data = (x1,x2,x3) for each patient
#  x1 = systolic blood pressure
#  x2 = age in years
#  x3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

# plt.scatter(X[:,1],X[:,0])
# plt.show()
# plt.scatter(X[:,2],X[:,0])
# plt.show()

df['ones']=1
y = df['X1']
x = df[['ones','X2','X3']]
x2 = df[['ones','X2']]
x3 = df[['ones','X3']]

import linear_model as lm
#  2D regression
w = lm.coeff_multi(x,y,False)
print(w)

yhat = np.dot(x,w)

# r^2 value
r2 = lm.r_squared(y,yhat)

print("r^2: ",r2)

plt.scatter(X[:,1],y)
plt.plot(sorted(X[:,1]),sorted(yhat))
plt.show()
plt.scatter(X[:,2],y)
plt.plot(sorted(X[:,2]),sorted(yhat))
plt.show()
