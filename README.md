'''This module contains tools for performing tangnet sensitivity analysis
and adjoint sensitivity analysis.

User should define two bi-variate functions, f and J.
f(u, s) defines a dynamical system du/dt = f(u,s) parameterized by s
J(u, s) defines the objective function, whose ergodic long time average
        is the quantity of interest.

# Use:
u0 = rand(m)      # initial condition of m-degree-of-freedom system
t = linspace(T0, T1, N)    # 0-T0 is spin up time (starting from u0).

# Using tangent sensitivity analysis:
tan = Tangent(f, u0, s, t)
dJds = tan.dJds(J)

# Using tangent sensitivity analysis:
adj = Adjoint(f, u0, s, t, J)
dJds = adj.dJds()

