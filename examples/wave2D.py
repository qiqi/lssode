# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
import argparse
from pylab import *
from numpy import *

sys.path.append('..')
from lssode import *

parser = argparse.ArgumentParser()
parser.add_argument('--rho', type=float, default=15.0)
parser.add_argument('--time_span', type=float, default=2.0)
parser.add_argument('--window_type', type=str, default='delta_end')
args = parser.parse_args()

print(args.time_span, args.window_type)

set_fd_step(1E-30j)

Nx = 20
Ny = 10

class VariablesUnpacker:

def wave(u, rho):
    alpha2 = 1.0  # speed of sound squared

    shp = u.shape
    u = u.reshape([-1, 2*Nx*Ny + 3]).T
    N = u.shape[1]

    x, y, z = u[:3]
    v1, v2 = u[3:].reshape([2, Nx, Ny, N])

    u = u[3:]           # shape [2*Nx*Ny by N]
    v1 = u[:Nx*Ny].reshape([Nx,Ny,N])
    v2 = u[Nx*Ny:].reshape([Nx,Ny,N])
    
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    
    dv1dt = zeros(v1.shape, v1.dtype)
    dv2dt = zeros(v1.shape, v1.dtype)

    dx,dy = 0.05,0.05

    dv1dt = v2

    v1_expanded = zeros([Nx + 2, Ny + 2])

    #left BC: dirchlet with lorenz z varible u(0,t)=z(t) over zrng, 0 otherwise
    zrng = arange(int(0.25*Ny),int(0.75*Ny)+1)
    lgt = zrng.shape[0]

    dv2dt[:,0,1:-1] = alpha2 * (v1[:,1,1:-1] - 2 * v1[:,0,1:-1]) / dx**2 \
                  + alpha2 * (v1[:,0,2:] - 2 * v1[:,0,1:-1] + v1[:,0,:-2]) / dy**2


    dv2dt[:,0,zrng] = alpha2 * (v1[:,1,zrng] - 2 * v1[:,0,zrng] + z[:,newaxis] * ones([u.shape[1],lgt])) / dx**2 \
                  + alpha2 * (v1[:,0,zrng+1] - 2 * v1[:,0,zrng] + v1[:,0,zrng-1]) / dy**2



    #interior domain
    dv2dt[:,1:-1,1:-1] = alpha2 * (v1[:,2:,1:-1] - 2 * v1[:,1:-1,1:-1] + v1[:,:-2,1:-1]) / dx**2 \
                     + alpha2 * (v1[:,1:-1,2:] - 2 * v1[:,1:-1,1:-1] + v1[:,1:-1,:-2]) / dy**2


    #right BC: 
    # robin: delta * u(N*dx,t) + gamma * ux(N*dx,t) = 0
    # delta = 0.0
    # gamma = 1.0
    #dy2dt[-1] = alpha2 * ((-1-dx*delta/gamma)*v1[-1] + v1[-2]) / dx**2 # robin
    
    #dirchlet
    #dy2dt[-1] = alpha2 * (-2*v1[-1] + v1[-2]) / dx**2     
    
    #ux(1) = -ut(1) #dissipate all KE
    dv2dt[:,-1,1:-1] = alpha2 * (-1.0*dx*v2[:,-1,1:-1]-v1[:,-1,1:-1] + v1[:,-2,1:-1]) / dx**2 \
                   + alpha2 * (v1[:,-1,2:] - 2 * v1[:,-1,1:-1] + v1[:,-1,:-2]) / dy**2

    # top
    dv2dt[:,1:-1,-1] = alpha2 * (v1[:,2:,-1] - 2 * v1[:,1:-1,-1] + v1[:,:-2,-1]) / dx**2 \
                   + alpha2 * (-1.0*dy*v2[:,1:-1,-1]-v1[:,1:-1,-1] + v1[:,1:-1,-2]) / dy**2 

    # bottom
    dv2dt[:,1:-1,0] = alpha2 * (v1[:,2:,0] - 2 * v1[:,1:-1,0] + v1[:,:-2,0]) / dx**2 \
                  + alpha2 * (-1.0*dy*v2[:,1:-1,0]-v1[:,1:-1,0] + v1[:,1:-1,1]) / dy**2
    # top right
    dv2dt[:,-1,-1] = alpha2 * (-1.0*dx*v2[:,-1,-1]-v1[:,-1,-1] + v1[:,-2,-1]) / dx**2 \
                 + alpha2 * (-1.0*dy*v2[:,-1,-1]-v1[:,-1,-1] + v1[:,-1,-2]) / dy**2

    # bottom right
    dv2dt[:,-1,0] = alpha2 * (-1.0*dx*v2[:,-1,-1]-v1[:,-1,-1] + v1[:,-2,-1]) / dx**2 \
                + alpha2 * (-1.0*dy*v2[:,-1,0]-v1[:,-1,0] + v1[:,-1,1]) / dy**2
    
    # top left
    dv2dt[:,0,-1] = alpha2 * (v1[:,1,-1] - 2 * v1[:,0,-1]) / dx**2 \
                + alpha2 * (-1.0*dy*v2[:,0,-1]-v1[:,0,-1] + v1[:,0,-2]) / dy**2

    # bottom left
    dv2dt[:,0,0] = alpha2 * (v1[:,1,-1] - 2 * v1[:,0,-1]) / dx**2 \
               + alpha2 * (-1.0*dy*v2[:,0,0]-v1[:,0,0] + v1[:,0,1]) / dy**2 


    # repack DOFs
    dv1dt = dv1dt.reshape([Nx*Ny,u.shape[1]])
    dv2dt = dv2dt.reshape([Nx*Ny,u.shape[1]])

    return transpose(vstack([[dxdt, dydt, dzdt], dv1dt, dv2dt])).reshape(shp)

def obj(u, r):
    return u[:,3+Ny*int(Nx/2) + int(Ny/2)] 


dt = 0.0025
t = 30 + dt * arange(int(args.time_span / dt)) #TODO back to 100!!

x0 = zeros(3 + 2 * Nx * Ny)
x0[:3] = random.rand(3)

adj = Adjoint(wave, x0, args.rho, t, obj, window_type=args.window_type)
wa = adj.wa
grad_l = adj.dJds()

print(adj._timing_)

print('lss: ', grad_l)

n = t.shape[0]
filename = "wave2D" + '_' + str(Nx) + '_' + str(Ny) + "_T" + str(args.time_span) + ".npz"
# savez(filename, n=n, u=u, wa_c=wa_c, wa=wa, grad_c=grad_c, grad_l=grad_l)
savez(filename, n=n, wa=wa, grad_l=grad_l)
