from dolfinx import fem, io, default_scalar_type, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import meshio
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import gmsh
# from plot_result import *

def observer_operator(mesh_pos, displacement, x0, y0, dx, dy) :
    subdomain_condition_x = ((mesh_pos[:, 0] >= x0) & (mesh_pos[:, 0] <= (x0 + dx)))
    subdomain_condition_y = ((mesh_pos[:, 1] >= y0) & (mesh_pos[:, 1] <= (y0 + dy)))
    subdomain_condition = subdomain_condition_x & subdomain_condition_y
    observed_mesh_pos = mesh_pos[subdomain_condition]
    observed_mesh_pos[:, 0] -= x0
    observed_mesh_pos[:, 1] -= y0
    observed_displacement = displacement[subdomain_condition]
    return observed_mesh_pos, observed_displacement
def get_mesh(msh_data) :
    domain, _, _ = msh_data
    fdim = domain.topology.dim - 1
    node_dim = 0
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    domain.topology.create_connectivity(node_dim, domain.topology.dim)
    adj_list = domain.topology.connectivity(domain.topology.dim, node_dim)
    mesh_pos = domain.geometry.x[:, :2]
    node_connectivity = adj_list.array.reshape(-1, 3)
    return mesh_pos, node_connectivity
def solve_fem(msh_data, G, K) :
    domain, _, facet_tags = msh_data
    fdim = domain.topology.dim - 1
    node_dim = 0
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    domain.topology.create_connectivity(node_dim, domain.topology.dim)
    adj_list = domain.topology.connectivity(domain.topology.dim, node_dim)

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    # Define Dirichlet Boundary Condition (Dirichelet BC is defined at the facet 2)

    b_D = fem.locate_dofs_topological(V, fdim, facet_tags.find(2))
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc_D = fem.dirichletbc(u_D, b_D, V)

    # Define Neumann Boundary Condition
    T_neuman = fem.Constant(domain, default_scalar_type((106.26e6, 0, 0 )))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # Define Weak form
    def epsilon(u):
        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    
    def sigma(u):
        return  (-2/3 * G + K) * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * G * epsilon(u)
    
    # Define Trial and Test Function
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Include Load into the system (the facet for Neumann Boundary is set at facet 3)
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T_neuman, v) * ds(3)

    # Create Solver (Linear)
    problem = LinearProblem(a, L, bcs=[bc_D], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Solve
    uh = problem.solve()

    num_dofs = V.dofmap.index_map.size_local
    mesh_pos = domain.geometry.x[:, :2]
    node_connectivity = adj_list.array.reshape(-1, 3)

    displacement = uh.x.array[:].reshape((num_dofs, -1))[:, :2]

    return mesh_pos, node_connectivity, displacement

def main() :
    mesh_file = "/home/narupanta/ADDMM/surrogate_model/rectangle_with_hole.msh"
    domain, mesh_tags, facet_tags = io.gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0)
    msh_data = io.gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0)
    G, K = 70e9, 100e9
    observed_x_0 = 20e-3
    observed_y_0 = -10e-3
    observed_dx = 80e-3
    observed_dy = 20e-3
    mesh_pos, node_connectivity, displacement = solve_fem(msh_data, G, K)
    observed_mesh_pos, observed_displacement = observer_operator(mesh_pos, displacement, observed_x_0, observed_y_0, observed_dx, observed_dy)
    # plot_result(mesh_pos, node_connectivity, displacement, "FEM Displacement")

if __name__ == "__main__" :
    main()