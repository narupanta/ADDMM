from dolfinx import io
from mpi4py import MPI
import numpy as np
from fem_solver import *

def main() :
    # G_ref, K_ref = [7.224e+10, 9.169e+10]
    G = np.linspace(6e+10, 2 * 6e+10, 30)
    K = np.linspace(8e+10, 2 * 8e+10, 30)
    mesh_file = "/home/narupanta/ADDMM/surrogate_model/rectangle_with_hole.msh"
    msh_data = io.gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0)
    index = 0
    for i_g, g in enumerate(G) :
        for i_k, k in enumerate(K) :
            params = [g, k]
            mesh_pos, node_connectivity, displacement = solve_fem(msh_data, g, k)
            np.savez(f'/home/narupanta/ADDMM/surrogate_model/dataset/disp_field_{index}.npz', mesh_pos = mesh_pos[:, :2], node_connectivity = node_connectivity, u = displacement[:, :2], params = np.array(params))
            index += 1
if __name__ == "__main__" :
    main()