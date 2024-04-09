import gmsh
import sys
import os
import math
import meshio
gmsh.initialize(sys.argv)

# merge STL, create surface patches that are reparametrizable (so we can remesh
# them) and compute the parametrizations
path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, 'pipe.stl'))
gmsh.model.mesh.classifySurfaces(math.pi, True, True)

#classifysurface:Classify ("color") the current surface mesh based on an angle threshold (the first argument, in radians), and create new discrete surfaces, curves and points accordingly. If the second argument is set, also create discrete curves on the boundary if the surface is open. If the third argument is set, create edges and surfaces than can be reparametrized with CreateGeometry. The last optional argument sets an angle threshold to force splitting of the generated curves
gmsh.model.mesh.createGeometry()
# Create a geometry for discrete entities (represented solely by a mesh, without an underlying CAD description) in the current model, i.e. create a parametrization for discrete curves and surfaces, assuming that each can be parametrized with a single map. If no entities are given, create a geometry for all discrete entities. 
#extrude = False
#extrude = False


#Get all the entities in the current model. A model entity is represented by two integers: its dimension (dim == 0, 1, 2 or 3) and its tag (its unique, strictly positive identifier). If dim is >= 0, return only the entities of the specified dimension (e.g. points if dim == 0). The entities are returned as a vector of (dim, tag) pairs. 
    # get "top" surfaces created by extrusion
gmsh.model.mesh.optimize('Relocate2D',True)
sur=gmsh.model.getEntities(2)
surface = [s[1] for s in sur]
top_ent = gmsh.model.getEntities(2)
top_surf = [s[1] for s in top_ent]
print(top_surf)
# get boundary of top surfaces, i.e. boundaries of holes
gmsh.model.geo.synchronize()
bnd_ent = gmsh.model.getBoundary(top_ent)
bnd_curv = [c[1] for c in bnd_ent]
print(top_surf)
print(bnd_curv)
# create plane surfaces filling the holes
loops = gmsh.model.geo.addCurveLoops(bnd_curv)
print(top_surf)
print(bnd_curv)
print(loops)
gmsh.model.mesh.optimize('Relocate2D',True)
for l in loops:
    top_surf.append(gmsh.model.geo.addPlaneSurface([l]))
sur_boundary=gmsh.model.getEntities(2)
surface_boundary = [s[1] for s in sur_boundary]
print(top_surf)
print(bnd_curv)
print(loops)
print(top_surf)
print(surface_boundary)
gmsh.model.mesh.optimize('Relocate2D',True)

in_marker=1
out_marker=2
wall_marker=3
domain_marker=4
gmsh.model.geo.addPhysicalGroup(2,[4],in_marker)
gmsh.model.setPhysicalName(2,in_marker,'in')
gmsh.model.geo.addPhysicalGroup(2,[5],out_marker)
gmsh.model.setPhysicalName(2,out_marker,'out')
gmsh.model.geo.addPhysicalGroup(2,surface,wall_marker)
gmsh.model.setPhysicalName(2,wall_marker,'wall')
#gmsh.model.geo.addPhysicalGroup(2,[177],6)
#gmsh.model.setPhysicalName(2,6,'boundary')
gmsh.model.geo.addPhysicalGroup(3,[1],domain_marker)
gmsh.model.setPhysicalName(3,domain_marker,'domain')
    # create the inner volume
gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(top_surf)])
gmsh.model.geo.synchronize()

# use MeshAdapt for the resulting not-so-smooth parametrizations
gmsh.option.setNumber('Mesh.Algorithm', 1)
gmsh.option.setNumber('Mesh.MeshSizeFactor', 0.3)
gmsh.model.mesh.generate()
gmsh.write('pipe.msh')
#Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
msh=meshio.read("tipe.msh")
meshio.write("pipe.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
#boundary face
meshio.write("mf_pipe.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells["triangle"]},cell_data={"triangle": {"name_to_read": msh.cell_data["triangle"]["gmsh:physical"]}}))
#boundary cell
meshio.write("cf_pipe.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]},cell_data={"tetra": {"name_to_read":msh.cell_data["tetra"]["gmsh:physical"]}}))
