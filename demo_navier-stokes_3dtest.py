"""This demo program solves the incompressible Navier-Stokes equations
on an L-shaped domain using Chorin's splitting method."""

# Copyright (C) 2010-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Mikael Mortensen 2011
#
# First added:  2010-08-30
# Last changed: 2011-06-30

# Begin demo


import matplotlib.pyplot as plt
from dolfin import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
#mesh = Mesh("lshape.xml.gz")
mesh = Mesh()
with XDMFFile("pipe.xdmf") as infile:
    infile.read(mesh)
##
mvc = MeshValueCollection("size_t", mesh, 2) 
with XDMFFile("mf_pipe.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
boundary_markers=MeshFunction("size_t",mesh,mvc)
##
mvc = MeshValueCollection("size_t", mesh, 3)
with XDMFFile("cf_pipe.xdmf") as infile:
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 3)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)
in_marker=2
out_marker=1
wall_marker=3
domain_marker=4
# Set parameter values
dt = 0.001
T = 1
#nu = 0.01
nu=0.0000033018868
# Define time-dependent pressure boundary condition
#p_in = Expression("sin(3.0*t)", t=0.0, degree=2)

# Define boundary conditions
#noslip  = DirichletBC(V, (0, 0),           "on_boundary && \                       (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \                       (x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS))")
#inflow  = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
#outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
#bcu = [noslip]
#bcp = [inflow, outflow]
bcu_in=DirichletBC(V,Constant((0.182866,-0.03869,-0.071156)),boundary_markers,in_marker)

bcp_out=DirichletBC(Q,Constant(666.61),boundary_markers,out_marker)
bcu_wall=DirichletBC(V,Constant((0.0,0.0,0.0)),boundary_markers,wall_marker)
bcu=[bcu_in,bcu_wall]
bcp=[bcp_out]
# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0,0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity3d.pvd")
pfile = File("results/pressure3d.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:
    print(t)
    # Update pressure boundary condition
    #p_in.t = t
    begin("Computing tentative velocity")
    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")
    end()
    begin("Computing pressure correction")
    # Pressure correction
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)
    end()
    begin("Computing velocity correction")
    # Velocity correction
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")
    end()
    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1)
    t += dt

# Plot solution
plt.figure()
plot(p1, title="Pressure")

plt.figure()
plot(u1, title="Velocity")

plt.show()

