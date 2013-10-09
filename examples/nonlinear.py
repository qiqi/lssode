# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

sys.path.append('..')
from lssode import *

def lorenz(u, rho):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    return transpose([dxdt, dydt, dzdt]).reshape(shp)


rhos = linspace(28, 33, 21)
x0 = random.rand(3)
dt, T = 0.01, 30
t = 30 + dt * arange(int(T / dt))

solver = lssSolver(lorenz, x0, rhos[0], t)
u, t = [solver.u.copy()], [solver.t.copy()]

for rho in rhos[1:]:
    print('rho = ', rho)
    solver.lss(rho)
    u.append(solver.u.copy())
    t.append(solver.t.copy())

u, t = array(u), array(t)

figure(figsize=(5,10))
contourf(rhos[:,newaxis] + t * 0, t, u[:,:,2], 501)
ylim([t.min(1).max(), t.max(1).min()])
xlabel(r'$\rho$')
ylabel(r'$t$')
title(r'$z$')
colorbar()
show()


