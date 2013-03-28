# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

sys.path.append('..')
from lssode import *

def vanderpol(u, mu):
    shp = u.shape
    u = u.reshape([-1,2])
    dudt = zeros(u.shape)
    dudt[:,0] = u[:,1]
    dudt[:,1] = -u[:,0] + mu * (1 - u[:,0]**2) * u[:,1]
    return dudt.reshape(shp)

def obj(u, mu):
    return u[:,1]**8

mus = linspace(0.2, 2, 10)
dt = 0.01

tangent = []
adjoint = []
for mu in mus:
    print mu
    for i in range(11):
        T = 20
        if i == 10: T = 200

        t = 50 + dt * arange(int(T / dt))

        x0 = random.rand(2)
        tan = Tangent(vanderpol, x0, mu, t)

        J = tan.evaluate(obj)
        dJds = tan.dJds(obj)

        J = pow(J, 1./8)
        dJds = 1./8 * pow(J, -7) * dJds
        tangent.append(dJds)

        x0 = random.rand(2)
        adj = Adjoint(vanderpol, x0, mu, t, obj)

        J = adj.evaluate(obj)
        dJds = adj.dJds()

        J = pow(J, 1./8)
        dJds = 1./8 * pow(J, -7) * dJds
        adjoint.append(dJds)

tangent = array(tangent).reshape([mus.size, -1])
adjoint = array(adjoint).reshape([mus.size, -1])

figure(figsize=(5,4))
plot(mus, tangent[:,:-1], 'xr')
plot(mus, tangent[:,-1], '-r')
plot(mus, adjoint[:,:-1], '+b')
plot(mus, adjoint[:,-1], '--b')
ylim([0, 1])
xlabel(r'$\mu$')
ylabel(r'$d\overline{J}/d\mu$')

show()


