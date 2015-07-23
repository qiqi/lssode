# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import os
import sys
import argparse
from pylab import *
from numpy import *

from wave2D import Wave2D

sys.path.append('../..')
from lssode import *

parser = argparse.ArgumentParser()
parser.add_argument('--rho', type=float, default=15.0)
parser.add_argument('--time_span', type=float, default=2.0)
parser.add_argument('--window_type', type=str, default='delta_end')
args = parser.parse_args()

print(args.time_span, args.window_type)

set_fd_step(1E-30j)

wave = Wave2D()
dt = 0.0025
t = 30 + dt * arange(int(args.time_span / dt))
x0 = zeros(wave.size())
x0[:3] = random.rand(3)

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


dt = 0.0025
t = 30 + dt * arange(int(args.time_span / dt))

u0 = zeros(wave.size())
u0[:3] = random.rand(3)

TEST_PRIMAL = False
if TEST_PRIMAL:
    from scipy.integrate import odeint
    
    t = dt * arange(int(20.0/dt))
    f = lambda u, t : wave.ddt(u, args.rho)
    assert t[0] >= 0 and t.size > 1

    N0 = int(t[0] / (t[-1] - t[0]) * t.size)
    print 'start run up'
    u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]
    print ' start time history'
    # compute a trajectory
    u = odeint(f, u0, t - t[0])
    
    ## plot solution at final time
    #utf = u[-1]
    #utf = transpose(utf[3:Nx*Ny+3].reshape([Nx,Ny]))
    #
    #show()
    
    Nx, Ny = wave.Nx, wave.Ny
    u = u[:,3:Nx*Ny+3]
    u = u.reshape([t.size,Nx,Ny])
    
    u_slice = u[:,:,int(Ny/2)] # slice of y plane
    
    print u_slice.shape
    
    
    figure(1)
    contourf(transpose(u[-1]), 100)
    xlabel('x')
    ylabel('y')
    title('primal')
    colorbar()
    
    
    figure(2)
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
    if not os.path.exists('movie'):
        os.mkdir('movie')
    
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
        clf()
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

f = lambda u, t : wave.ddt(u, s)
assert t[0] >= 0 and t.size > 1
N0 = int(t[0] / (t[-1] - t[0]) * t.size)
u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

# compute a trajectory
u = odeint(f, u0, t - t[0])

print "build matrices"
# build jacobians and time integration terms
dfdu = ddu_sparse(wave.ddt)
dt = t[1:] - t[:-1]
uMid = 0.5 * (u[1:] + u[:-1])
halfJ = 0.5 * dfdu(uMid, s)
print halfJ.shape, type(halfJ)
eyeDt = eye(wave.size(),wave.size()) / dt[:,newaxis,newaxis] 
E = -eyeDt - halfJ
G = eyeDt - halfJ
n = dt.shape[0]


# build forcing terms
dJdu = ddu(wave.obj)
win = window(u.shape[0], args.window_type)
g = dJdu(u, s) * win[:,newaxis,newaxis]

# compute unsteady adjoint
wa_c = zeros(uMid.shape)

# loop BACKWARDS through everything...
print "start adjoint"
# last step
wa_c[-1] = linalg.solve(G[-1].T,g[-1].T).squeeze()

for i in arange(n-2,-1,-1):
    rhs = -dot(E[i+1].T,wa_c[i+1]).T + squeeze(g[i].T)
    wa_c[i] = linalg.solve(G[i].T,rhs).squeeze()

# compute sensitvities
dfds = dds(wave.ddt)
dJds = dds(wave.obj)
b = dfds(uMid, s)
prod = wa_c[:,:,newaxis] * b
grad1 = prod.sum(1).mean(0)
grad2 = dJds(uMid, s).mean(0)
grad_c = ravel(grad1 + grad2) 
print 'conventional: ', grad_c 
