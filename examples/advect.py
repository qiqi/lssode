# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
import argparse
from pylab import *
from numpy import *

sys.path.append('..')
from lssode import *

parser = argparse.ArgumentParser()
parser.add_argument('--rho', type=float, default=15)
parser.add_argument('--time_span', type=float, default=10)
parser.add_argument('--window_type', type=str, default='delta_end')
args = parser.parse_args()

print(args.time_span, args.window_type)

set_fd_step(1E-30j)

def advect(u, rho):
    shp = u.shape
    u = u.reshape([-1, 53]).T
    x, y, z = u[:3]
    u = u[3:]
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    dudt = zeros(u.shape, u.dtype)
    dx = 0.1
    dudt[0] = (z - u[0]) / dx
    dudt[1:] = (u[:-1] - u[1:]) / dx
    return transpose(vstack([[dxdt, dydt, dzdt], dudt])).reshape(shp)

def obj(u, r):
    return u[:,-1]

dt = 0.0025

T = 10

t = 30 + dt * arange(int(args.time_span / dt))

x0 = random.rand(53)
tan = Tangent(advect, x0, args.rho, t)
print(tan.dJds(obj, window_type=args.window_type))

subplot(1,2,1)
contourf(tan.v[:,3:], 100)
xlabel('space')
ylabel('time')
title('tangent')
colorbar()

x0 = random.rand(53)
adj = Adjoint(advect, x0, args.rho, t, obj, window_type=args.window_type)
print(adj.dJds())

subplot(1,2,2)
contourf(adj.wa[:,3:], 100)
xlabel('space')
ylabel('time')
title('adjoint')
colorbar()

savefig('advect.png')
