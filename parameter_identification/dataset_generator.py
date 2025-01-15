from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl

import numpy as np
import pandas as pd
import json
import meshio

def FEM_Solution(domain, mesh_file, facet_tags, G, K) :
    # Import Geometry
    mesh = meshio.read(mesh_file)
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    # Define Dirichlet Boundary Condition (Dirichelet BC is defined at the facet 3)
    b_D = fem.locate_dofs_topological(V, fdim, facet_tags.find(3))
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

    # Include Load into the system (the facet for Neumann Boundary is set at facet 4)
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T_neuman, v) * ds(4)

    # Create Solver (Linear)
    problem = LinearProblem(a, L, bcs=[bc_D], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Solve
    uh = problem.solve()
    
    # Post Processing (Retrieve solution in the dataset's domain)
    x_origin_sub = [20e-3, 100e-3]
    y_origin_sub = 10e-3

    domain_arr = domain.geometry.x
    print(domain_arr.shape)
    print(mesh.points.shape)
    subdomain_condition = (domain_arr[:, 0] >= x_origin_sub[0]) & (domain_arr[:, 0] <= x_origin_sub[1]) # Extract x-coordinates and apply condition
    
    num_dofs = V.dofmap.index_map.size_local
    uh_arr = uh.x.array[:].reshape((num_dofs, -1))
    def get_uh_with_coords(positions, features, query_positions) :
        """
        Return the features corresponding to the given query positions.

        Parameters:
        - positions: A 2D NumPy array of shape (n, d) representing the positions.
        - features: A NumPy array of shape (n, m) containing the features at each position.
        - query_positions: A 2D NumPy array of shape (k, d) representing the positions to query.

        Returns:
        - A NumPy array of shape (k, m) containing the features corresponding to the query positions.
        """
        # Find indices of the query positions in the positions array
        indices = np.where((positions[:, None] == query_positions).all(axis=2))[1]
        
        # Return the corresponding features
        return features[indices]

    # Get the subdomain for using with the data
    domain_sub = domain_arr[subdomain_condition]
    for cell in mesh.cells:
        if cell.type == 'triangle' :
            triangles = cell.data
    domain_sub[:, 0] -= x_origin_sub[0]
    domain_sub[:, 1] += y_origin_sub
    u_sub = uh_arr[subdomain_condition]

    return mesh.points, triangles, get_uh_with_coords(mesh.points, uh_arr, domain_arr), uh, domain_sub, u_sub


def main() :
    G_ref, K_ref = [7.224e+10, 9.169e+10]
    G = np.linspace(6e+10, 2 * 6e+10, 30)
    K = np.linspace(8e+10, 2 * 8e+10, 30)
    mesh_file = "tensile_test_specimen2.msh"
    domain, mesh_tags, facet_tags = io.gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0)
    for i_g, g in enumerate(G) :
        for i_k, k in enumerate(K) :
            coords, triangles, disp, _, _, _= FEM_Solution(domain, mesh_file, facet_tags, g, k)
            # with open(f'/mnt/c/Users/bankp/OneDrive/Desktop/ADDMM/surrogate_model/dataset/disp_field_{str(i_g)}_{str(i_k)}.json', 'w', encoding='utf-8') as f:
            #     json.dump(dict(mesh_pos = sub_domain[:, :2].tolist(), u = disp[:, :2].tolist()), f, ensure_ascii=False, indent=4)
            np.savez(f'/mnt/c/Users/bankp/OneDrive/Desktop/ADDMM/surrogate_model/dataset/disp_field_{str(i_g)}_{str(i_k)}.npz', mesh_pos = coords[:, :2], node_connectivity = triangles, u = disp[:, :2])

    with open('/mnt/c/Users/bankp/OneDrive/Desktop/ADDMM/surrogate_model/dataset/params_desc/G_K_indices.json', 'w', encoding='utf-8') as f:
        json.dump(dict(G_list = G.tolist(), K_list = K.tolist()), f, ensure_ascii=False, indent=4)

if __name__ == "__main__" :
    main()