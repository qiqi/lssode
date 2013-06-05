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

def obj(u, r):
    return u[:,2]

rhos = linspace(25, 34, 10)
dt = 0.01

tangent = []
adjoint = []
for rho in rhos:
    print rho
    for i in range(11):
        T = 20
        if i == 10: T = 200

        t = 30 + dt * arange(int(T / dt))

        x0 = random.rand(3)
        tan = Tangent(lorenz, x0, rho, t)

        J = tan.evaluate(obj)
        dJds = tan.dJds(obj)
        tangent.append(dJds)

        x0 = random.rand(3)
        adj = Adjoint(lorenz, x0, rho, t, obj)

        J = adj.evaluate()
        dJds = adj.dJds()
        adjoint.append(dJds)

tangent = array(tangent).reshape([rhos.size, -1])
adjoint = array(adjoint).reshape([rhos.size, -1])

figure(figsize=(5,4))
plot(rhos, tangent[:,:-1], 'xr')
plot(rhos, tangent[:,-1], '-r')
plot(rhos, adjoint[:,:-1], '+b')
plot(rhos, adjoint[:,-1], '--b')
ylim([0, 1.5])
xlabel(r'$\rho$')
ylabel(r'$d\overline{J}/d\rho$')

show()


