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

def wave(u, rho):
    
    alpha2 = 1.0

    shp = u.shape
    u = u.reshape([-1, 2*Nx*Ny + 3]).T
    x, y, z = u[:3]
    u = u[3:]
    v1 = u[:Nx*Ny].reshape([u.shape[1],Nx,Ny])
    v2 = u[Nx*Ny:].reshape([u.shape[1],Nx,Ny])
    
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    
    dv1dt = zeros(v1.shape, v1.dtype)
    dv2dt = zeros(v1.shape, v1.dtype)

    dx,dy = 0.05,0.05

    dv1dt = v2

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

def window(n, window_type='sin2'):
    if window_type == 'square':
        win = ones(n)
    elif window_type == 'sin':
        win = np.sin(np.linspace(0, np.pi, n+2)[1:-1])
    elif window_type == 'sin2':
        win = np.sin(np.linspace(0, np.pi, n+2)[1:-1])**2
    elif window_type == 'sin4':
        win = np.sin(np.linspace(0, np.pi, n+2)[1:-1])**4
    elif window_type == 'bump':
        x = np.linspace(-1, 1, n+2)[1:-1]
        win = np.exp(-1 / (1 - x**2))
    elif window_type == 'delta_end':
        win = np.zeros(n)
        win[-1] = 1
    win /= win.mean()
    return win




m = 2*Nx*Ny+3

dt = 0.0025

#T = 10

t = 30 + dt * arange(int(args.time_span / dt)) #TODO back to 100!!


print(m)
x0 = random.rand(m)
x0a = x0.copy()

'''
###############
# TEST PRIMAL #
###############

t = dt * arange(int(30/dt))


u0 = zeros(m)
u0[:3] = random.rand(3)

x0 = ((8./3.) * (args.rho-1))**0.5
y0 = ((8./3.) * (args.rho-1))**0.5
z0 = args.rho-1
u0[:3] = array([x0,y0,z0]) + random.rand(3)


from scipy.integrate import odeint



f = lambda u, t : wave(u, args.rho)
assert t[0] >= 0 and t.size > 1
N0 = int(t[0] / (t[-1] - t[0]) * t.size)
#u0 = x0.copy()
u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

# compute a trajectory
u = odeint(f, u0, t - t[0])

## plot solution at final time
#utf = u[-1]
#utf = transpose(utf[3:Nx*Ny+3].reshape([Nx,Ny]))
#
#contourf(utf, 100)
#xlabel('x')
#ylabel('y')
#title('primal')
#colorbar()
#show()


u = u[:,3:Nx*Ny+3]
u = u.reshape([t.size,Nx,Ny])

u_slice = u[:,:,int(Ny/2)] # slice of y plane

print u_slice.shape

contourf(u_slice, 100)
xlabel('space (x)')
ylabel('time')
title('primal')
colorbar()
show()



# produce video
print 'starting post pro'
import time
n = t.shape[0]
frm = 200

#u = u[:,3:]

j = 0
for i in arange(0,n,int(n/frm)):
    #uf = transpose(u[i,:Nx*Ny].reshape([Nx,Ny]))
    #contourf(uf, 100)
    contourf(transpose(u[i]),100)
    xlabel('x')
    ylabel('y')
    title('primal')
    clim(5.0,12.0)
    #colorbar()
    savefig('movie/frame' + str(j) + '.png')
    close()
    print i, '/', n ,' done'
    j = j + 1


#########################################
# COMPUTE AND PLOT CONVENTIONAL ADJOINT #
#########################################

s = array(args.rho, float).copy()
if s.ndim == 0:
    s = s[newaxis]

# compute primal
from scipy.integrate import odeint

f = lambda u, t : wave(u, s)
assert t[0] >= 0 and t.size > 1
N0 = int(t[0] / (t[-1] - t[0]) * t.size)
u0 = x0.copy()
u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

# compute a trajectory
u = odeint(f, u0, t - t[0])


# build jacobians and time integration terms
dfdu = ddu(wave)
dt = t[1:] - t[:-1]
uMid = 0.5 * (u[1:] + u[:-1])
halfJ = 0.5 * dfdu(uMid, s)
eyeDt = eye(m,m) / dt[:,newaxis,newaxis] 
E = -eyeDt - halfJ
G = eyeDt - halfJ
n = dt.shape[0]


# build forcing terms
dJdu = ddu(obj)
win = window(u.shape[0], args.window_type)
g = dJdu(u, s) * win[:,newaxis,newaxis]

# compute unsteady adjoint
wa_c = zeros(uMid.shape)

# loop BACKWARDS through everything...

# last step
wa_c[-1] = linalg.solve(G[-1].T,g[-1].T).squeeze()

for i in arange(n-2,-1,-1):
    rhs = -dot(E[i+1].T,wa_c[i+1]).T + squeeze(g[i].T)
    wa_c[i] = linalg.solve(G[i].T,rhs).squeeze()




# compute sensitvities
dfds = dds(wave)
dJds = dds(obj)
b = dfds(uMid, s)
prod = wa_c[:,:,newaxis] * b
grad1 = prod.sum(1).mean(0)
grad2 = dJds(uMid, s).mean(0)
grad_c = ravel(grad1 + grad2) 
print 'conventional: ', grad_c 

del E,G,halfJ,eyeDt
'''

# LSS

x0 = x0a
adj = Adjoint(wave, x0, args.rho, t, obj, window_type=args.window_type)
wa = adj.wa
grad_l = adj.dJds()

print(adj._timing_)

print('lss: ', grad_l)

n = t.shape[0]
filename = "wave2D" + '_' + str(Nx) + '_' + str(Ny) + "_T" + str(args.time_span) + ".npz"
# savez(filename, n=n, u=u, wa_c=wa_c, wa=wa, grad_c=grad_c, grad_l=grad_l)
savez(filename, n=n, wa=wa, grad_l=grad_l)
