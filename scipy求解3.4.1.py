import numpy as np
from scipy.optimize import linprog

c = np.array([4,1])
A_ub = np.array([[-4,-3],[1,2]])
b_ub = np.array([-6,4])
A_eq = np.array([[3,1]])
b_eq = np.array([3])
r = linprog(c,A_ub,b_ub,A_eq,b_eq,bounds=((0,None),(0,None)))
print(r)