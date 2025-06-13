import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

#assigning constant values
E = 210e9  # Young's modulus for steel
b = h = 0.1  # width/height the cross-section
I = (b * h**3) / 12  # second moment of inertia
L = 3.0  # beam length
P = 10000  # load

# FEM set up
num_elements = 6
num_nodes = num_elements + 1
element_length = L / num_elements
dofs_per_node = 2
total_dofs = dofs_per_node * num_nodes

# --- Beam element stiffness matrix using cubic Hermite basis ---
def beam_element_stiffness(EI, h):
    factor = EI / h**3
    return factor * np.array([
        [12,     6*h,   -12,    6*h],
        [6*h,  4*h**2, -6*h,  2*h**2],
        [-12,   -6*h,    12,   -6*h],
        [6*h, 2*h**2, -6*h,  4*h**2]
    ])

# Sparse global stiffness matrix (use LIL for efficient assembly)
K = lil_matrix((total_dofs, total_dofs))

# Assemble global stiffness matrix
for e in range(num_elements):
    k_e = beam_element_stiffness(E * I, element_length)
    dof_map = [2*e, 2*e+1, 2*e+2, 2*e+3] # 2 here represents the dof is 2 per node, so it's +2 row when moving to the next node
    for i in range(4):
        for j in range(4):
            K[dof_map[i], dof_map[j]] += k_e[i, j]

# Convert to CSR format for efficient solving
K_csr = K.tocsr()

# Force vector stays the same
F = np.zeros(total_dofs)
F[2 * (num_nodes // 2)] = -P # corresponds to the basis function for the deflection of the center node, with the use of Dirac delta function

# Apply boundary conditions
clamped_dofs = [0, 1, total_dofs-2, total_dofs-1]
free_dofs = [i for i in range(total_dofs) if i not in clamped_dofs]

# Solve reduced system using sparse solver
K_reduced = K_csr[free_dofs, :][:, free_dofs]
F_reduced = F[free_dofs]

U = np.zeros(total_dofs)
U[free_dofs] = spsolve(K_reduced, F_reduced)

# Plot (same as before)
x_nodes = np.linspace(0, L, num_nodes)
displacements = U[::2]
scale = 1
plt.plot(x_nodes, displacements * scale, '-o')
plt.title("Deflection of Beam")
plt.xlabel("x (m)")
plt.ylabel("Deflection (scaled)")
plt.grid(True)
plt.show()