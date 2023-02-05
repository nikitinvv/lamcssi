"""
CUDA Raw kernels for computing back-projection to orthogonal slices
"""

import cupy as cp

source = """
extern "C" {    
    void __global__ fwd(float *data, float *u, float *theta, int n, int nz, int deth, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= deth || tz >= ntheta)
            return;
            
        float x = 0;
        float y = 0;
        float z = 0;
        int xr = 0;
        int yr = 0;
        int zr = 0;
        float data0 = 0;
        
        float ctheta = __cosf(theta[tz]);
        float stheta = __sinf(theta[tz]);
        
        for (int t = 0; t<n; t++)
        {
            x = tx;
            y = stheta*(t-n/2)-ctheta*(ty-deth/2) + n/2;
            z = ctheta*(t-n/2)+stheta*(ty-deth/2) + nz/2;      
            xr = (int)x;
            yr = (int)y;
            zr = (int)z;
            
            // linear interp            
            if ((xr >= 0) & (xr < n - 1) & (yr >= 0) & (yr < n - 1) & (zr >= 0) & (zr < nz - 1))
            {
                x = x-xr;
                y = y-yr;
                z = z-zr;
                data0 +=u[xr+0+(yr+0)*n+(zr+0)*n*n]*(1-x)*(1-y)*(1-z)+
                        u[xr+1+(yr+0)*n+(zr+0)*n*n]*(0+x)*(1-y)*(1-z)+
                        u[xr+0+(yr+1)*n+(zr+0)*n*n]*(1-x)*(0+y)*(1-z)+
                        u[xr+1+(yr+1)*n+(zr+0)*n*n]*(0+x)*(0+y)*(1-z)+
                        u[xr+0+(yr+0)*n+(zr+1)*n*n]*(1-x)*(1-y)*(0+z)+
                        u[xr+1+(yr+0)*n+(zr+1)*n*n]*(0+x)*(1-y)*(0+z)+
                        u[xr+0+(yr+1)*n+(zr+1)*n*n]*(1-x)*(0+y)*(0+z)+
                        u[xr+1+(yr+1)*n+(zr+1)*n*n]*(0+x)*(0+y)*(0+z);
            }
        }
        data[tx + ty * n + tz * n * deth] = data0*n;        
    }    

    void __global__ adj(float *u, float *data, float *theta, int n, int nz, int deth, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nz)
            return;
        float p = 0;
        float v = 0;
        int pr = 0;
        int vr = 0;        
        
        float u0 = 0;
        float ctheta = 0;
        float stheta = 0;
            
        for (int t = 0; t<ntheta; t++)
        {
            ctheta = __cosf(theta[t]);
            stheta = __sinf(theta[t]);
            
            p = tx;
            v = -ctheta*(ty-n/2)+stheta*(tz-nz/2) + deth/2;
            
            pr = (int)p;
            vr = (int)v;            
            // linear interp            
            if ((pr >= 0) & (pr < n - 1) & (vr >= 0) & (vr < deth - 1))
            {
                p = p-pr;
                v = v-vr;                
                u0 +=   data[pr+0+(vr+0)*n+t*n*deth]*(1-p)*(1-v)+
                        data[pr+1+(vr+0)*n+t*n*deth]*(0+p)*(1-v)+
                        data[pr+0+(vr+1)*n+t*n*deth]*(1-p)*(0+v)+
                        data[pr+1+(vr+1)*n+t*n*deth]*(0+p)*(0+v);
                        
            }
        }
        u[tx + ty * n + tz * n * n] += u0*n;        
    }    
}
"""

module = cp.RawModule(code=source)
fwd_kernel = module.get_function('fwd')
adj_kernel = module.get_function('adj')

def fwd(data, u, theta):
    [nz, n] = u.shape[:2]
    [ntheta, deth] = data.shape[:2]
    fwd_kernel((int(cp.ceil(n/32)), int(cp.ceil(deth/32)), ntheta), (32, 32, 1),
                  (data, u, theta, n, nz, deth, ntheta))
    return data

def adj(u, data, theta):
    [nz, n] = u.shape[:2]
    [ntheta,deth] = data.shape[:2]
    adj_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), nz), (32, 32, 1),
                  (u,data, theta, n, nz, deth, ntheta))
    return u
