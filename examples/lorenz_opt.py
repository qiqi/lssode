# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

sys.path.append('..')
from lssode import *

set_fd_step(1E-30j)

def lorenz(u, rho):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    return transpose([dxdt, dydt, dzdt]).reshape(shp)

def obj(u, r):
    return (u[:,2] - 27)**2

rhos = linspace(25, 34, 25)
dt = 0.0025

J, G = [], []
for rho in rhos:
    print(rho)
    T = 50

    t = 30 + dt * arange(int(T / dt))
    x0 = random.rand(3)
    tan = Tangent(lorenz, x0, rho, t)

    J.append(tan.evaluate(obj))
    G.append(tan.dJds(obj))

J, G = array(J), ravel(array(G))
figure()
plot(rhos, J, 'sk')
axis([24,35,75,120])
xlabel('Design parameter')
ylabel('Objective function')
savefig('before.png')
dr = 0.25
plot([rhos-dr, rhos+dr], [J-G*dr, J+G*dr], '-k')
axis([24,35,75,120])
savefig('after.png')
