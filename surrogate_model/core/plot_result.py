import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
def plot_result(mesh_pos: np.array, node_connectivity: np.array, displacement: np.array, desc: str) :

    # Plotting
    triangulation = tri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], node_connectivity)

    # Create subplots5
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))  # 2 row, 3 columns

    # First subplot
    for j in range(2) :
        im = axes[j].tripcolor(triangulation, displacement[:, j], shading='flat', cmap='viridis')
        axes[j].triplot(triangulation, color='black', linewidth=0.5)
        if j == 0 :
            axes[j].set_title(f"{desc} Displacement X (m)")
        else :
            axes[j].set_title(f"{desc} Displacement Y (m)")
        axes[j].set_xlabel("X-coordinate")
        axes[j].set_ylabel("Y-coordinate")
        fig.colorbar(im, ax=axes[j], orientation='vertical')
    fig.suptitle(desc, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_experimental_data() :
    pass

def plot_model_result() :
    pass

def plot_training_progress() :
    pass

def plot_compare_data() :
    pass