import cvxopt
from cvxopt import matrix
c=matrix([4.0,1.0])
G=matrix([[-4.,1.,-1.,0],[-3.,2.,0,-1.]])
h=matrix([-6.,4.,0.,0.])
A=matrix([3.,1.],(1,2))
b=matrix(3.)
sv=cvxopt.solvers.lp(c,G,h,A,b)
print(sv['x'])