import gmsh
import sys
import os
import math

gmsh.initialize(sys.argv)

# merge STL, create surface patches that are reparametrizable (so we can remesh
# them) and compute the parametrizations
path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, 'xuluwen.stl'))
#gmsh.General.ScaleX=0.001
#gmsh.General.ScaleY=0.001
#gmsh.General.ScaleZ=0.001
gmsh.model.mesh.classifySurfaces(math.pi, True, True)

#classifysurface:Classify ("color") the current surface mesh based on an angle threshold (the first argument, in radians), and create new discrete surfaces, curves and points accordingly. If the second argument is set, also create discrete curves on the boundary if the surface is open. If the third argument is set, create edges and surfaces than can be reparametrized with CreateGeometry. The last optional argument sets an angle threshold to force splitting of the generated curves
gmsh.model.mesh.createGeometry()
#gmsh.option.setNumber('Mesh.ScalingFactor',0.001)
# Create a geometry for discrete entities (represented solely by a mesh, without an underlying CAD description) in the current model, i.e. create a parametrization for discrete curves and surfaces, assuming that each can be parametrized with a single map. If no entities are given, create a geometry for all discrete entities. 
extrude = True
#extrude = False

if extrude:
    # make extrusions only return "top" surfaces and volumes, not lateral
    # surfaces
    gmsh.option.setNumber('Geometry.ExtrudeReturnLateralEntities', 0)
    #gmsh.option.setNumber('Mesh.ScalingFactor',0.001)
    # extrude a boundary layer of 4 elements using mesh normals (thickness =
    # 0.5)
   # a=gmsh.model.geo.extrudeBoundaryLayer(gmsh.model.getEntities(2), [5], [0.5], True)
   # print(a)
    # extrude a second boundary layer in the opposite direction (note the
    # `second == True' argument to distinguish it from the first one)
    sur=gmsh.model.getEntities(2)
    surface = [s[1] for s in sur]
    print(surface)
    e = gmsh.model.geo.extrudeBoundaryLayer(gmsh.model.getEntities(2), [10], [-1],True,True)
    print(e)
    #print(gmsh.model.getEntities(2))
#Get all the entities in the current model. A model entity is represented by two integers: its dimension (dim == 0, 1, 2 or 3) and its tag (its unique, strictly positive identifier). If dim is >= 0, return only the entities of the specified dimension (e.g. points if dim == 0). The entities are returned as a vector of (dim, tag) pairs. 
    # get "top" surfaces created by extrusion
    
    top_ent = [s for s in e if s[0] == 2] #boundary surfaces(interior)
    top_surf = [s[1] for s in top_ent]
    print(top_surf)
    #print(top_ent)
    # get boundary of top surfaces, i.e. boundaries of holes
    gmsh.model.geo.synchronize()

#surface's boundary:curve
    bnd_ent = gmsh.model.getBoundary(top_ent) #boundary curves
# boundary curve tag
    bnd_curv = [c[1] for c in bnd_ent]
    print(bnd_curv)
    # create plane surfaces filling the holes
    loops = gmsh.model.geo.addCurveLoops(bnd_curv)
    print(loops)
    for l in loops:
        top_surf.append(gmsh.model.geo.addPlaneSurface([l]))
    print(top_surf)
    sur_boundary=gmsh.model.getEntities(2)
    surface_boundary = [s[1] for s in sur_boundary]
    print(surface_boundary)
    splenic_marker=1
    smv_marker=2
    portal_marker=3
    wall_marker=4
    domain_marker=5
    gmsh.model.geo.addPhysicalGroup(2,[211,42],splenic_marker)
    gmsh.model.setPhysicalName(2,splenic_marker,'splenic')
    gmsh.model.geo.addPhysicalGroup(2,[212,56,99],smv_marker)
    gmsh.model.setPhysicalName(2,smv_marker,'smv')
    gmsh.model.geo.addPhysicalGroup(2,[213,133,177],portal_marker)
    gmsh.model.setPhysicalName(2,portal_marker,'portal')
    gmsh.model.geo.addPhysicalGroup(2,surface,wall_marker)
    gmsh.model.setPhysicalName(2,wall_marker,'wall')
    #gmsh.model.geo.addPhysicalGroup(2,[177],6)
    #gmsh.model.setPhysicalName(2,6,'boundary')
    gmsh.model.geo.addPhysicalGroup(3,[1,2,3,4,5,6,7,8,9,10,11],domain_marker)
    gmsh.model.setPhysicalName(3,domain_marker,'domain') 
    #gmsh.model.geo.addPhysicalGroup(1,[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 44, 49, 50, 51, 52, 54, 55, 58, 63, 71, 72, 73, 75, 79, 80, 89, 91, 111, 117, 122, 123, 124, 128, 131, 147, 161, 167, 189, 195],6)
    #gmsh.model.setPhysicalName(1,6,'boundary')


    #gmsh.model.geo.addPhysicalGroup(1,[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 44, 49, 50, 51, 52, 54, 55, 58, 63, 71, 72, 73, 75, 79, 80, 89, 91, 111, 117, 122, 123, 124, 128, 131, 147, 161, 167, 189, 195],6)
    #gmsh.model.setPhysicalName(1,6,'boundary')




    # create the inner volume
    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(top_surf)])
    #print(gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(top_surf)]))
    #print(gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(top_surf)]))
    #gmsh.option.setNumber('Mesh.ScalingFactor',0.001)

    gmsh.model.geo.synchronize()

gmsh.option.setNumber('Mesh.Algorithm', 1)
gmsh.option.setNumber('Mesh.MeshSizeFactor', 0.1)
#gmsh.option.setNumber('Mesh.SaveAll', 1)
#gmsh.General.ScaleX=0.001
#gmsh.General.ScaleY=0.001
#gmsh.General.ScaleZ=0.001

#gmsh.option.setNumber('General.ScaleX',0.001)
#gmsh.option.setNumber('General.ScaleY',0.001)
#gmsh.option.setNumber('General.ScaleZ',0.001)
#gmsh.option.setNumber('Mesh.ScalingFactor',0.001)
#gmsh.option.setNumber('Geometry.MatchMeshScaleFactor',0.001)
gmsh.model.mesh.generate(3)
#gmsh.option.setNumber('Geometry.MatchMeshScaleFactor',0.001)
#gmsh.option.setNumber('Mesh.ScalingFactor',0.001)
#gmsh.model.mesh.optimize('Relocate2D',True)
#gmsh.model.mesh.optimize('Relocate3D',True)

gmsh.write('create_msh.msh')
gmsh.finalize()
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities

from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)
gdim = 3
mesh_comm = MPI.COMM_WORLD
model_rank = 0
#mesh, ct, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh,ct, ft = gmshio.read_from_msh("create_msh.msh", MPI.COMM_WORLD, 0, gdim=3)
ft.name = "Facet markers"
ct.name = "Cell markers" 
t = 0
T = 0.8                       # Final time
dt = 0.001                 # Time step size
num_steps = int(T / dt)
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(0.0035))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1060))
f = Constant(mesh, PETSc.ScalarType((0,0,-9.81)))
v_cg3 = VectorElement("Lagrange", mesh.ufl_cell(), 3)
s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, v_cg3)
Q = FunctionSpace(mesh, s_cg1)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
fdim = mesh.topology.dim - 1
u1.name="u"
p1.name="p"
#class InletVelocity():
#    def __init__(self, t):
#        self.t = t
#
#    def __call__(self, x):
#        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
#        values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
#        return values


# Inlet
#u_inlet = Function(V)
#inlet_velocity = InletVelocity(t)
#u_inlet.interpolate(inlet_velocity)
#bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
u_smv = np.array((0.1,0,-0.1), dtype=PETSc.ScalarType)
bcu_smv = dirichletbc(u_smv, locate_dofs_topological(V, fdim, ft.find(smv_marker)), V)
u_splenic = np.array((-0.1,0,0.1), dtype=PETSc.ScalarType)
bcu_splenic = dirichletbc(u_splenic, locate_dofs_topological(V, fdim, ft.find(splenic_marker)), V)
# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_wall = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
# Obstacle
#bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
#bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
bcp_portal = dirichletbc(PETSc.ScalarType(700), locate_dofs_topological(Q, fdim, ft.find(portal_marker)), Q)
bcu=[bcu_smv,bcu_splenic,bcu_wall]
bcp = [bcp_portal]

# t=n (u,p) unknown
#u_ = Function(V)
#p_ = Function(Q)

# t=n-1 known
#u_1 = Function(V)
#p_1 = Function(Q)
u0=Function(V)
u1=Function(V)
p1=Function(Q)
#normal
n  = FacetNormal(mesh)
# force
f  = Constant((0, 0,-9.81))
###chorin
#step1
#F1 = dot((u-u_1)/deltat,v)*dx+dot(nabla_grad(u_1).T*u_1,v)*dx+(mu/rho)*inner(nabla_grad(u),nabla_grad(v))*dx-dot(f,v)*dx
F1=(1/k)*dot(u - u0, v)*dx+inner(dot(nabla_grad(u0),u0),v)*dx+mu/rho*inner(grad(u),grad(v))*dx-dot(f,v)*dx
a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1=create_matrix(a1)
b1 = create_vector(L1)
# step2
a2 = form(dot(grad(p),grad(q))*dx)
L2 = form(-(rho/k)*dot(div(u1),q)*dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)
# step3
a3 = form(dot(u,v)*dx)
L3 = form(dot(u1,v)*dx-(k/rho)*dot(nabla_grad(p1),v)*dx)
A3.assemble()
b3 = create_vector(L3)
# matrix
#A1 = assemble(a1)
#A2 = assemble(a2)
#A3 = assemble(a3)

# Use amg preconditioner if available
#prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)
# boundary condition
#[bc.apply(A1) for bc in bcu]
#[bc.apply(A2) for bc in bcp]
#[bc.apply(A3) for bc in bcu]

from pathlib import Path
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, "dfg3D-3-u.bp", [u_], engine="BP4")
vtx_p = VTXWriter(mesh.comm, "dfg3D-3-p.bp", [p_], engine="BP4")
vtx_u.write(t)
vtx_p.write(t)
progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    # Update current time step
    t += dt
    # Update inlet velocity
    #inlet_velocity.t = t
    #u_inlet.interpolate(inlet_velocity)

    # Step 1: Tentative velocity step
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u1.vector)
    u1.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p1.vector)
    p1.x.scatter_forward()

    p_.vector.axpy(1, p1.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()

    # Write solutions to file
    vtx_u.write(t)
    vtx_p.write(t)
    print(f"Iteration {i}: Correction norm {p_}")
    # Update variable with solution form this time step
    with u_.vector.localForm() as loc_, u0.vector.localForm() as loc_n:
        loc_.copy(loc_n)
vtx_u.close()
vtx_p.close()
