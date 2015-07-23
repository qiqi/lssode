# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from numpy import *

class Wave2D:
    def __init__(self, Nx=20, Ny=10, c=1.0):
        self.Nx = Nx
        self.Ny = Ny
        self.c2 = c**2
        self.dx = 2. / Nx
        self.dy = 1. / Ny
        self.window = sin(linspace(0, pi, Nx))**2

    def size(self):
        return self.Nx * self.Ny * 2 + 3

    def unpack_vars(self, u):
        Nx, Ny = self.Nx, self.Ny
        u = u.reshape([-1, 2*Nx*Ny + 3]).T
        N = u.shape[1] 
        x, y, z = u[:3]
        v1, v2 = u[3:].reshape([2,Nx,Ny,N])
        return x, y, z, v1, v2

    def repack_vars(self, dxdt, dydt, dzdt, dv1dt, dv2dt):
        Nx, Ny, N = self.Nx, self.Ny, dv1dt.shape[2]
        dvdt = array([dv1dt, dv2dt]).reshape((2 * Nx * Ny, N))
        return transpose(vstack([[dxdt, dydt, dzdt], dvdt]))

    def extendWithBC(self, v1, v2, z):
        Nx, Ny, N = self.Nx, self.Ny, v1.shape[2]
        v1_ext = zeros([Nx + 2,Ny + 2, N], v1.dtype)
        v1_ext[1:-1,1:-1] = v1

        # bottom BC: dirchlet with lorenz z varible times a window function
        v1_ext[1:-1,0]  = z[newaxis] * self.window[:,newaxis]
        v1_ext[1:-1,-1] = v1[:,-1] - self.dy * v2[:,-1]  # top
        v1_ext[0,1:-1]  = v1[0,:]  - self.dx * v2[0,:]   # left
        v1_ext[-1,1:-1] = v1[-1,:] - self.dx * v2[-1,:]  # right
        return v1_ext

    def ddt(self, u, rho):
        shp = u.shape
        x, y, z, v1, v2 = self.unpack_vars(u)
        
        sigma, beta = 10, 8./3
        dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z

        v1_ext = self.extendWithBC(v1, v2, z)
        d2v1dx2 = (v1_ext[2:,1:-1] - 2 * v1_ext[1:-1,1:-1] + v1_ext[:-2,1:-1]) / self.dx**2
        d2v1dy2 = (v1_ext[1:-1,2:] - 2 * v1_ext[1:-1,1:-1] + v1_ext[1:-1,:-2]) / self.dy**2

        dv2dt = self.c2 * (d2v1dx2 + d2v1dy2)
        dv1dt = v2

        return self.repack_vars(dxdt, dydt, dzdt, dv1dt, dv2dt).reshape(shp)

    def obj(self, u, rho):
        x, y, z, v1, v2 = self.unpack_vars(u)
        return v1[-1, 0]

