# Copyright Qiqi Wang (qiqi@mit.edu) 2013

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

adj = Adjoint(wave.ddt, x0, args.rho, t, wave.obj, window_type=args.window_type)
u = adj.u
wa = adj.wa
grad_l = adj.dJds()

print(adj._timing_)

print('lss: ', grad_l)

n = t.shape[0]
filename = "wave2D" + '_' + str(wave.Nx) + '_' + str(wave.Ny) + "_T" + str(args.time_span) + ".npz"
savez(filename, n=n, u=u, wa=wa, grad_l=grad_l)


