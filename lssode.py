# Copyright Qiqi Wang (qiqi@mit.edu) 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This module contains tools for performing tangnet sensitivity analysis
and adjoint sensitivity analysis.  The details are described in our paper
"Sensitivity computation of periodic and chaotic limit cycle oscillations"
at http://arxiv.org/abs/1204.0159

User should define two bi-variate functions, f and J

f(u, s) defines a dynamical system du/dt = f(u,s) parameterized by s
        inputs:
        u: size (m,) or size (N,m). It's the state of the m-degree-of-freedom
           dynamical system
        s: parameter of the dynamical system.
           Tangent sensitivity analysis: s must be a scalar.
           Adjoint sensitivity analysis: s may be a scalar or vector.
        return: du/dt, should be the same size as the state u.
                if u.shape == (m,): return a shape (m,) array
                if u.shape == (N,m): return a shape (N,m) array

J(u, s) defines the objective function, whose ergodic long time average
        is the quantity of interest.
        inputs: Same as in f(u,s)
        return: instantaneous objective function to be time averaged.
                Tangent sensitivity analysis:
                    J may return a scalar (single objectives)
                              or a vector (n objectives).
                    if u.shape == (m,): return a scalar or vector of shape (n,)
                    if u.shape == (N,m): return a vector of shape (N,)
                                         or vector of shape (N,n)
                Adjoint sensitivity analysis:
                    J must return a scalar (single objective).
                    if u.shape == (m,): return a scalar
                    if u.shape == (N,m): return a vector of shape (N,)

Using tangent sensitivity analysis:
        u0 = rand(m)      # initial condition of m-degree-of-freedom system
        t = linspace(T0, T1, N)    # 0-T0 is spin up time (starting from u0).
        tan = Tangent(f, u0, s, t)
        dJds = tan.dJds(J)
        # you can use the same "tan" for more "J"s ...

Using adjoint sensitivity analysis:
        adj = Adjoint(f, u0, s, t, J)
        dJds = adj.dJds()
        # you can use the same "adj" for more "s"s
        #     via adj.dJds(dfds, dJds)... See doc for the Adjoint class
"""

import sys
import pdb
import time
import numpy as np
import numpad as pad
from scipy import sparse
from scipy.integrate import odeint
import scipy.sparse.linalg as splinalg

import matplotlib.pyplot as plt

__all__ = ["ddu", "ddu_sparse", "dds", "set_fd_step", "Tangent", "Adjoint"]


def _diag(a):
    """Construct a block diagonal sparse matrix, A[i,:,:] is the ith block"""
    assert a.ndim == 1
    n = a.size
    return sparse.csr_matrix((a, np.r_[:n], np.r_[:n+1]))


EPS = 1E-7

def set_fd_step(eps):
    """Set step size in ddu and dds classess.
    set eps=1E-30j for complex derivative method."""
    assert isinstance(eps, (float, complex))
    global EPS
    EPS = eps


class ddu(object):
    """Partial derivative of a bivariate function f(u,s)
    with respect its FIRST argument u
    Usage: print(ddu(f)(u,s))
    Or: dfdu = ddu(f)
        print(dfdu(u,s))
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        global EPS
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, u.shape[1]
        dfdu = np.zeros( (N, n, m) )
        u = np.asarray(u, type(EPS))
        s = np.asarray(s, type(EPS))
        for i in range(m):
            u[:,i] += EPS
            fp = self.f(u, s).copy()
            u[:,i] -= EPS * 2
            fm = self.f(u, s).copy()
            u[:,i] += EPS
            dfdu[:,:,i] = ((fp - fm).reshape([N, n]) / (2 * EPS)).real
        return dfdu


class ddu_sparse(object):
    """Sparse Jacobian of a bivariate function f(u,s)
    with respect its FIRST argument u

    Notes:
    1. This functor returns sparse Jacobian
    2. When u is a 2D array containing a list of states,
       this functor returns a block diagonal matrix, whose
       diagonal blocks are the Jacobians of all the states.

    Usage: print(ddu(f)(u,s))
    Or: dfdu = ddu(f)
        print(dfdu(u,s))
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        return pad.diff_func(self.f, u, (s,))


class dds(object):
    """Partial derivative of a bivariate function f(u,s)
    with respect its SECOND argument s

    Usage: print(dds(f)(u,s))
    Or: dfds = dds(f)
        print(dfds(u,s))
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        global EPS
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, s.size
        dfds = np.zeros( (N, n, m) )
        u = np.asarray(u, type(EPS))
        s = np.asarray(s, type(EPS))
        for i in range(m):
            s[i] += EPS
            fp = self.f(u, s).copy()
            s[i] -= EPS * 2
            fm = self.f(u, s).copy()
            s[i] += EPS
            dfds[:,:,i] = ((fp - fm).reshape([N, n]) / (2 * EPS)).real
        return dfds

def window(n, window_type='sin2', n0=0, n1=0):
    if n1 == 0:
        n1 = n
    
    win = np.zeros(n)
    n_obj = n1-n0
    if window_type == 'square':
        win[n0:n1] = np.ones(n_obj)
    elif window_type == 'sin':
        win[n0:n1] = np.sin(np.linspace(0, np.pi, n_obj+2)[1:-1])
    elif window_type == 'sin2':
        win[n0:n1] = np.sin(np.linspace(0, np.pi, n_obj+2)[1:-1])**2
    elif window_type == 'sin4':
        win[n0:n1] = np.sin(np.linspace(0, np.pi, n_obj+2)[1:-1])**4
    elif window_type == 'bump':
        x = np.linspace(-1, 1, n_obj+2)[1:-1]
        win[n0:n1] = np.exp(-1 / (1 - x**2))
    elif window_type == 'delta_end':
        win[-1] = 1
    win /= win.mean()
    return win

class LSS(object):
    """
    Base class for both tangent and adjoint sensitivity analysis
    During __init__, a trajectory is computed,
    and the matrices used for both tangent and adjoint are built
    """
    def __init__(self, f, u0, s, t, dfdu=None):
        t0 = time.time()
        self.f = f
        self.t = np.array(t, float).copy()
        self.s = np.array(s, float).copy()

        if self.s.ndim == 0:
            self.s = self.s[np.newaxis]

        if dfdu is None:
            dfdu = ddu_sparse(f)
        self.dfdu = dfdu

        u0 = np.array(u0, float)
        if u0.ndim == 1:
            # run up to t[0]
            f = lambda u, t : self.f(u, s)
            assert t[0] >= 0 and t.size > 1
            N0 = int(t[0] / (t[-1] - t[0]) * t.size)
            u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

            # compute a trajectory
            self.u = odeint(f, u0, t - t[0])
        else:
            assert (u0.shape[0],) == t.shape
            self.u = u0.copy()

        self.dt = t[1:] - t[:-1]
        self.uMid = 0.5 * (self.u[1:] + self.u[:-1])
        self.dudt = (self.u[1:] - self.u[:-1]) / self.dt[:,np.newaxis]

        self._timing_ = {}
        self._timing_['init'] = time.time() - t0


    def Schur(self):
        """
        Builds the Schur complement of the KKT system'
        Also build B: the block-bidiagonal matrix
        """
        t0 = time.time()
        N, m = self.u.shape[0] - 1, self.u.shape[1]

        halfJ = 0.5 * self.dfdu(self.uMid, self.s).tocsr()
        eyeDt = _diag(np.kron(1./ self.dt, np.ones(m)))

        B1 = -eyeDt - halfJ
        B2 = eyeDt - halfJ
        B1 = (B1.data, B1.indices, B1.indptr)
        B2 = (B2.data, B2.indices + m, B2.indptr)
        B1 = sparse.csr_matrix(B1, (N * m, (N + 1) * m))
        B2 = sparse.csr_matrix(B2, (N * m, (N + 1) * m))
        self.B = B1 + B2
        
        # preconditioner matrix containing main block diagonal
        B1d = B1[:,:-m]
        B2d = B2[:,m:]

        self.Spre = B1d * B1d.T + B2d * B2d.T


        # the diagonal weights
        dtFrac = self.dt / (self.t[-1] - self.t[0])
        wb = 0.5 * (np.hstack([dtFrac, 0]) + np.hstack([0, dtFrac]))
        wb = np.ones(m) * wb[:,np.newaxis]
        self.wBinv = _diag(np.ravel(1./ wb))

        schur = self.B * self.wBinv * self.B.T

        self._timing_['schur'] = time.time() - t0
        return schur

    def directSolve(self,A,b):
        # solve linear system with a sparse direct solver
        return splinalg.spsolve(A, b)

    def iterSolve(self,A,b,x0,P=None,restrt=50,its=20,tol=1e-14):
        # Iterative Solver (restarted GMRES with restrt * its iterations) 
      
        # preconditioner
        if P is None:
            M = None
        else:
            ilu = splinalg.spilu(P.tocsc())
            M_x = lambda x: ilu.solve(x)
            M = splinalg.LinearOperator((x0.size,x0.size), M_x)

        for i in range(restrt):
            x, info = splinalg.gmres(A, b, x0 = x0, maxiter=its, restart=its, M=M, tol=tol)
            resnorm = np.linalg.norm(b - A * x)
            print('iter ', (i+1) * its, resnorm)
              
            x0 = x.copy()

        return x

    # Multigrid Functions:

    def SchurMG(self,N_G):
        """
        Builds the Schur complement of the KKT system for each grid level
        Returns list of matrices
        Also build B: the block-bidiagonal matrix
        """
        t0 = time.time()
        N, m = self.u.shape[0] - 1, self.u.shape[1]

        halfJ = 0.5 * self.dfdu(self.uMid, self.s).tocsr()
        eyeDt = _diag(np.kron(1./ self.dt, np.ones(m)))

        B1 = -eyeDt - halfJ
        B2 = eyeDt - halfJ
        B1 = (B1.data, B1.indices, B1.indptr)
        B2 = (B2.data, B2.indices + m, B2.indptr)
        B1 = sparse.csr_matrix(B1, (N * m, (N + 1) * m))
        B2 = sparse.csr_matrix(B2, (N * m, (N + 1) * m))
        self.B = B1 + B2 # TODO: save this only for fine grid I_G = 0
        
        # preconditioner matrix containing main block diagonal
        B1d = B1[:,:-m]
        B2d = B2[:,m:]

        self.Spre = B1d * B1d.T + B2d * B2d.T # TODO: save as list of matrices


        # the diagonal weights
        dtFrac = self.dt / (self.t[-1] - self.t[0])
        wb = 0.5 * (np.hstack([dtFrac, 0]) + np.hstack([0, dtFrac]))
        wb = np.ones(m) * wb[:,np.newaxis]
        self.wBinv = _diag(np.ravel(1./ wb))

        schur = self.B * self.wBinv * self.B.T

        self._timing_['schur'] = time.time() - t0
        return schur

    # TODO: FMG (rhs with offset definition!)
    # TODO: restriction
    # TODO: prolongation
    # TODO: recursive V cycle

    # N_G = number of grids
    # I_G = grid index, fine grid = 0
    # offset = pow(2, I_G)
    # make sure grids have 2^m - 1 time steps = t must have 2^m time steps!

    def evaluate(self, J, window_type='sin2'):
        """Evaluate a time averaged objective function"""
        win = window(self.u.shape[0], window_type)
        return (J(self.u, self.s) * win).mean(0)


class Tangent(LSS):
    """
    Tagent(f, u0, s, t, dfds=None, dfdu=None)
    f: governing equation du/dt = f(u, s)
    u0: initial condition (1d array) or the entire trajectory (2d array)
    s: parameter
    t: time (1d array).  t[0] is run up time from initial condition.
    dfds and dfdu is computed from f if left undefined.
    """
    def __init__(self, f, u0, s, t, dfds=None, dfdu=None):
        LSS.__init__(self, f, u0, s, t, dfdu)

        Smat = self.Schur()

        t0 = time.time()
        if dfds is None:
            dfds = dds(f)
        b = dfds(self.uMid, self.s)
        assert b.size == Smat.shape[0]

        w = splinalg.spsolve(Smat, np.ravel(b))
        v = self.wBinv * (self.B.T * w)

        self.v = v.reshape(self.u.shape)
        self._timing_['solve'] = time.time() - t0

    def dJds(self, J, T0skip=0, T1skip=0, window_type='sin2'):
        """Evaluate the derivative of the time averaged objective function to s
        """
        t0 = time.time()
        pJpu, pJps = ddu(J), dds(J)

        n0 = (self.t < self.t[0] + T0skip).sum()
        n1 = (self.t <= self.t[-1] - T1skip).sum()
        assert n0 < n1

        u, v = self.u[n0:n1], self.v[n0:n1]
        uMid = self.uMid[n0:n1-1]

        J0 = J(uMid, self.s)
        J0 = J0.reshape([uMid.shape[0], -1])

        win = window(self.u.shape[0], window_type)
        grad1 = ((pJpu(u, self.s) * v[:,np.newaxis,:]).sum(2) \
                * win[:,np.newaxis]).mean(0)

        grad2 = pJps(uMid, self.s)[:,:,0].mean(0)
        
        grad = np.ravel(grad1 + grad2)

        self._timing_['eval'] = time.time() - t0
        return grad


class Adjoint(LSS):
    """
    Adjoint(f, u0, s, t, J, dJdu=None, dfdu=None)
    f: governing equation du/dt = f(u, s)
    u0: initial condition (1d array) or the entire trajectory (2d array)
    s: parameter
    t: time (1d array).  t[0] is run up time from initial condition.
    J: objective function. QoI = mean(J(u))
    dJdu and dfdu is computed from f if left undefined.
    """
    def __init__(self, f, u0, s, t, J, dJdu=None, dfdu=None, window_type='sin2',T0skip=0, T1skip=0):
        LSS.__init__(self, f, u0, s, t, dfdu)


        # TODO: build separate solver functions in LSS class.  
        Smat = self.Schur()

        t0 = time.time()
        if dJdu is None:
            dJdu = ddu(J)

        n0 = (self.t < self.t[0] + T0skip).sum()
        n1 = (self.t <= self.t[-1] - T1skip).sum()
        assert n0 < n1
        
        win = window(self.u.shape[0], window_type, n0=n0, n1=n1)
        g = dJdu(self.u, self.s) * win[:,np.newaxis,np.newaxis]
        assert g.size == self.u.size

        b = self.B * (self.wBinv * np.ravel(g))
        self.rhs = b

        self.J, self.dJdu = J, dJdu

        #wa = self.directSolve(Smat,b)

        x01 = np.ones(self.uMid.shape) 
        x02 = np.random.rand(self.uMid.shape[0]) 
        
        x01 = x01 * x02[:,np.newaxis]
        x0 = x01.reshape(b.shape)

        x0 = np.random.rand(b.shape[0])
        
        b0 = Smat * x0
        x0 *= abs(b).max() / b0.std()
        wa0 = x0
        
        wa0 = 0*b
        
        wa = self.iterSolve(Smat,b,wa0,P=self.Spre,restrt=5,its=20)
        '''
        self.conv_hist = []
  
        for i in range(50):
            wa, info = splinalg.gmres(Smat, b, x0 = wa0, maxiter=20, M=M, tol=tol)
            res = b - Smat * wa
            resnorm = np.linalg.norm(res)
            self.wa = x.reshape(self.uMid.shape)
            grad = self.dJds()
            print('iter ', (i+1) * 20, resnorm, grad[0])
            self.conv_hist.append([(i+1)*20, resnorm, grad[0]])
            #
            #plt.subplot(2,1,1)
            #plt.plot(wa)
            #plt.subplot(2,1,2)
            #plt.plot(res)
            #plt.show()
            #
            wa0 = wa.copy()
        '''
       
        self.wa = wa.reshape(self.uMid.shape)
        self._timing_['solve'] = time.time() - t0

    def evaluate(self):
        """Evaluate the time averaged objective function"""
        # return self.J(self.u, self.s).mean(0)
        return LSS.evaluate(self, self.J)

    def dJds(self, dfds=None, dJds=None, T0skip=0, T1skip=0):
        """Evaluate the derivative of the time averaged objective function to s
        """
        t0 = time.time()
        if dfds is None:
            dfds = dds(self.f)
        if dJds is None:
            dJds = dds(self.J)

        n0 = (self.t < self.t[0] + T0skip).sum()
        n1 = (self.t <= self.t[-1] - T1skip).sum()


        uMid, wa = self.uMid[n0:n1-1], self.wa[n0:n1-1]

        print np.linalg.norm(uMid), np.linalg.norm(wa)

        prod = self.wa[:,:,np.newaxis] * dfds(self.uMid, self.s)
        grad1 = prod.sum(1).mean(0)
        grad2 = dJds(self.uMid, self.s).mean(0)

        grad = np.ravel(grad1 + grad2)
        self._timing_['eval'] = time.time() - t0
        return grad

class Callback:
    'convergence monitor'

    def __init__(self,lss):
        self.n = 0
        self.lss=lss
        self.hist = []


    def __call__(self,x):
        self.n += 1
        if self.n == 1 or self.n % 10 == 0:
            res = self.lss.rhs - self.lss.B * self.lss.wBinv * self.lss.B.T * x 
            resnorm = np.linalg.norm(res)
            self.lss.wa = x.reshape(self.lss.uMid.shape)
            grad = self.lss.dJds()
            print('iter ', self.n, resnorm, grad[0])
            self.hist.append([self.n, resnorm, grad[0]])
            
            #plt.subplot(2,1,1)
            #plt.plot(x.copy())
            #plt.subplot(2,1,2)
            #plt.plot(res.copy())
            #plt.show()


        sys.stdout.flush()



