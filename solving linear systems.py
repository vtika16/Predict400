import numpy as np
--x = [[30,18,9
import sympy


from sympy import Matrix
from sympy.abc import x,y,z
from sympy.solvers.solvers import solve_linear_system_LU

solve_linear_system_LU(Matrix([
[128,60,470,2400],
[4,0,22,90],
[32,1,4,180]]), [x,y,z])


system = Matrix(( (1,4,2), (-2,1,14)))
solve_linear_system(system,x,y)



system = Matrix((30.0,18.0,9.0,42.0),(6.0,3.0,2.0,12.0),(12.0,3.0,12.0,8.0))
solve_linear_system(system,x,y,z)



output = sympy.Matrix(x).rref()
print(output)



import sympy as sympy
from sympy import Matrix
from sympy.abc import w,x,y,z
from sympy.solvers.solvers import solve_linear_system_LU

solve_linear_system_LU(Matrix([
[24,12,10,14,7210],
[31,21,13,19,9905],
[40.8,28.8,19.2,26.4,13704],
[36.72,23.04,23.04,27.72,13705.2]]), [w,x,y,z])
