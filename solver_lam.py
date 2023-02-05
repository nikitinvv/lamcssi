"""Module for laminography."""

import cupy as cp
import numpy as np
from kernels import fwd,adj
from cupyx.scipy.fft import rfft, irfft

class SolverLam():
    """Base class for laminography solvers using the direct line integration with linear interpolation on GPU.
    This class is a context manager which provides the basic operators required
    to implement a laminography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    n : int
        Object size in x, detector width
    nz : int
        Object size in z
    deth : int
        Detector height
    ntheta : int
        Number of projections
    ctheta : int
        Chunk size in angles for simultaneous processing with a GPU
    theta : float32
        Angles for laminography (rotation around the axis orthogonal to the beam)     
    """
    def __init__(self, n, nz, deth, ntheta, ctheta, theta):
        self.n = n
        self.nz = nz
        self.deth = deth
        self.ntheta = ntheta
        self.ctheta = ctheta
        self.theta = (90-theta)/180*np.pi # NOTE: switching angles to the 'laminography' formulation: [0,pi], rotation from the axis orhotgonal to the beam
        
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        pass

    def fwd_lam(self, u):
        """Forward laminography operator data = Lu"""
        
        # result
        data = np.zeros([self.ntheta, self.deth, self.n], dtype='float32')    

        # GPU memory
        data_gpu = cp.zeros([self.ctheta, self.deth, self.n], dtype='float32')
        theta_gpu = cp.zeros([self.ctheta], dtype='float32')                
        u_gpu = cp.asarray(u)
        
        # processing by chunks in angles
        for it in range(int(np.ceil(self.ntheta/self.ctheta))):
            st = it*self.ctheta
            end = min(self.ntheta,(it+1)*self.ctheta)
            
            # copy a data chunk to gpu                            
            data_gpu[:end-st] = cp.asarray(data[st:end])
            data_gpu[end-st:] = 0
            theta_gpu[:end-st] = cp.asarray(self.theta[st:end])        
            
            # generate data
            fwd(data_gpu,u_gpu,theta_gpu)
            
            # copy result to CPU
            data[st:end] = data_gpu[:end-st].get()
        return data

    def adj_lam(self,data):
        """adjoint laminography operator u = L*data"""
        
        # GPU memory
        data_gpu = cp.zeros([self.ctheta, self.deth, self.n], dtype='float32')
        u_gpu = cp.zeros([self.nz, self.n, self.n], dtype='float32')
        theta_gpu = cp.zeros([self.ctheta], dtype='float32')                
        
        for it in range(int(np.ceil(self.ntheta/self.ctheta))):
            st = it*self.ctheta
            end = min(self.ntheta,(it+1)*self.ctheta)                        
            
            # copy a data chunk to gpu                            
            data_gpu[:end-st] = cp.asarray(data[st:end])
            data_gpu[end-st:] = 0
            theta_gpu[:end-st] = cp.asarray(self.theta[st:end])   
            data_gpu = self.fbp_filter_center(data_gpu)
            # bakprojection            
            adj(u_gpu,data_gpu,theta_gpu)   
        u =  u_gpu.get()
        
        return u    
    
    def fbp_filter_center(self, data, sh=0):
        """FBP filtering of projections"""
        
        ne = 3*self.n//2
        
        t = cp.fft.rfftfreq(ne).astype('float32')
        # if self.args.gridrec_filter == 'parzen':
        #     w = t * (1 - t * 2)**3  
        # elif self.args.gridrec_filter == 'shepp':
            # w = t * cp.sinc(t)  
        # elif self.args.gridrec_filter == 'ramp':
        w = t          
        # w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sh+self.det/2))  # center fix
        # w = w*cp.exp(-2*cp.pi*1j*t*(-0.5))  # center fix
        data = data.swapaxes(1,2)
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')        
        data = irfft(w*rfft(data, axis=2), axis=2)
        data = cp.ascontiguousarray(data[:, :, ne//2-self.n//2:ne//2+self.n//2])
        data[:] = data.swapaxes(1,2)
        
        return data