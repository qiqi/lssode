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

'''This module contains tools for performing tangnet sensitivity analysis
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
'''

import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import scipy.sparse.linalg as splinalg


__all__ = ["ddu", "dds", "Tangent", "Adjoint"]


def _block_diag(A):
    'Construct a block diagonal sparse matrix, A[i,:,:] is the ith block'
    assert A.ndim == 3
    n = A.shape[0]
    return sparse.bsr_matrix((A, np.r_[:n], np.r_[:n+1]))


EPS = 1E-7

class ddu(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, u.shape[1]
        dfdu = np.zeros( (N, n, m) )
        for i in range(m):
            u[:,i] += EPS
            fp = self.f(u, s).copy()
            u[:,i] -= EPS * 2
            fm = self.f(u, s).copy()
            u[:,i] += EPS
            dfdu[:,:,i] = (fp - fm).reshape([N, n]) / (2 * EPS)
        return dfdu


class dds(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, s.size
        dfds = np.zeros( (N, n, m) )
        for i in range(m):
            s[i] += EPS
            fp = self.f(u, s).copy()
            s[i] -= EPS * 2
            fm = self.f(u, s).copy()
            s[i] += EPS
            dfds[:,:,i] = (fp - fm).reshape([N, n]) / (2 * EPS)
        return dfds


class LSS(object):
    '''
    Base class for both tangent and adjoint sensitivity analysis
    During __init__, a trajectory is computed,
    and the matrices used for both tangent and adjoint are built
    '''
    def __init__(self, f, u0, s, t, dfdu=None):
        self.f = f
        self.t = np.array(t).copy()
        self.s = np.array(s).copy()

        if self.s.ndim == 0:
            self.s = self.s[np.newaxis]

        if dfdu is None:
            dfdu = ddu(f)
        self.dfdu = dfdu

        # run up to t[0]
        f = lambda u, t : self.f(u, s)
        assert t[0] >= 0 and t.size > 1
        N0 = int(t[0] / (t[-1] - t[0]) * t.size)
        u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

        # compute a trajectory
        self.u = odeint(f, u0, t - t[0])

        self.dt = t[1:] - t[:-1]
        self.uMid = 0.5 * (self.u[1:] + self.u[:-1])
        self.dudt = (self.u[1:] - self.u[:-1]) / self.dt[:,np.newaxis]

        self.buildSparseMatrices()
    
    def buildSparseMatrices(self):
        '''
        Building B: the block-bidiagonal matrix,
             and E: the dudt matrix
        '''
        N, m = self.u.shape[0] - 1, self.u.shape[1]

        halfJ = 0.5 * self.dfdu(self.uMid, self.s)
        eyeDt = np.eye(m,m) / self.dt[:,np.newaxis,np.newaxis]
    
        L = sparse.bsr_matrix((halfJ, np.r_[1:N+1], np.r_[:N+1]) ) \
          + sparse.bsr_matrix((halfJ, np.r_[:N], np.r_[:N+1]), \
                              shape=(N*m, (N+1)*m))
    
        DDT = sparse.bsr_matrix((eyeDt, np.r_[1:N+1], np.r_[:N+1])) \
            - sparse.bsr_matrix((eyeDt, np.r_[:N], np.r_[:N+1]), \
                                shape=(N*m, (N+1)*m))
    
        self.B = DDT.tocsr() - L.tocsr()
        self.E = _block_diag(self.dudt[:,:,np.newaxis]).tocsr()

    def Schur(self, alpha):
        'Builds the Schur complement of the KKT system'
        B, E = self.B, self.E
        return (B * B.T) + (E * E.T) / alpha**2

    def evaluate(self, J):
        'Evaluate a time averaged objective function'
        return J(self.u, self.s).mean(0)


class Tangent(LSS):
    def __init__(self, f, u0, s, t, dfds=None, dfdu=None, alpha=0.1):
        LSS.__init__(self, f, u0, s, t, dfdu)

        S = self.Schur(alpha)

        if dfds is None:
            dfds = dds(f)
        b = dfds(self.uMid, self.s)
        assert b.size == S.shape[0]

        w = splinalg.spsolve(S, np.ravel(b))
        v = self.B.T * w

        self.v = v.reshape(self.u.shape)
        self.eta = self.E.T * w / alpha**2

    def dJds(self, J):
        dJdu, dJds = ddu(J), dds(J)

        J0 = J(self.uMid, self.s)
        J0 = J0.reshape([self.uMid.shape[0], -1])

        Jp = J(self.u + EPS * self.v, self.s).mean(0)
        Jm = J(self.u - EPS * self.v, self.s).mean(0)
        grad1 = (Jp - Jm) / (2*EPS) \
              - (self.eta[:,np.newaxis] * (J0 - J0.mean(0))).mean(0)

        grad2 = dJds(self.uMid, self.s).mean(0)
        return grad1 + grad2


class Adjoint(LSS):
    def __init__(self, f, u0, s, t, J, dJdu=None, dfdu=None, alpha=0.1):
        LSS.__init__(self, f, u0, s, t, dfdu)

        S = self.Schur(alpha)

        J0 = J(self.uMid, self.s)
        assert J0.ndim == 1
        h = -(J0 - J0.mean()) / (alpha**2 * J0.size)    # multiplier on eta

        if dJdu is None:
            dJdu = ddu(J)
        g = dJdu(self.u, self.s) / self.u.shape[0]      # multiplier on v
        assert g.size == self.u.size

        b = self.E * h + self.B * np.ravel(g)
        wa = splinalg.spsolve(S, b)

        self.wa = wa.reshape(self.uMid.shape)
        self.J, self.dJdu = J, dJdu

    def dJds(self, dfds=None, dJds=None):
        if dfds is None:
            dfds = dds(self.f)
        if dJds is None:
            dJds = dds(self.J)

        prod = self.wa[:,:,np.newaxis] * dfds(self.uMid, self.s)
        grad1 = prod.sum(0).sum(0)
        grad2 = dJds(self.uMid, self.s).mean(0)
        return grad1 + grad2

