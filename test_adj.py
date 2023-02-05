import numpy as np
import dxchange
from solver_lam import SolverLam


n = 256 # sample width
nz = 32 # sample height
deth = 256 # detector height
ntheta = 20 # number of angles 
ctheta = ntheta # chunk size for angles 

# define angles
theta = np.linspace(0, 30, ntheta, endpoint=True).astype('float32')

# read [nz,n,n] part of an object [256,256,256]
u = -dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2]


with SolverLam(n, nz, deth, ntheta, ctheta, theta) as slv:
    # generate data, forward Laminography transform (data = Lu)
    data = slv.fwd_lam(u)            
    dxchange.write_tiff(data, 'data/data', overwrite=True)
    
    # adjoint Laminography transform (ur = L*data)
    ur = slv.adj_lam(data)            
    dxchange.write_tiff(ur, 'data/rec', overwrite=True)

print(np.linalg.norm(u*ur))
print(np.linalg.norm(data*data))