#2D System: Will need to take a lattice instead of a one-dimensional input array, and then we can flatten it in order to feed it to the same functions we've already made

def get_neighbors_2d(L):
    #With 2D, instead of a 1D array, it's a 2D lattice with periodic condiitons, and it returns a list of pairs of indicies. Unlike 1D and 2D, it's both vertical and horizontal.
    neighbors = []
    for row in range(L):
        for col in range(L):
            i = row * L + col 
            right = row * L + (col + 1) % L
            left  = row * L + (col - 1) % L
            down  = ((row + 1) % L) * L + col
            up    = ((row - 1) % L) * L + col
            neighbors.append((i, right))
            neighbors.append((i, down))
            # only need right and down to avoid double-counting
    return neighbors

#Other change is diagonal energy: have to integrate the multidimensionality of the lattice, and thereforce take one dimension as the zeroes and then their respective second dimension values

def compute_diag_energy_2d(sigma_all, J, neighbor_pairs):
    e_diag = torch.zeros(sigma_all.shape[0])
    for i, j in neighbor_pairs:
        e_diag += -J * sigma_all[:, i] * sigma_all[:, j]
    return e_diag


def _square_pbc_bonds(L: int) -> list[tuple[int, int]]:
    #2 * N unique bonds, where N is the square of L (spin size. This function gives all nearest neighbor bounds.
    bonds = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            bonds.append((i,  x * L          + (y + 1) % L)) 
            bonds.append((i, ((x + 1) % L) * L + y          ))  
    return bonds


def ed_exact_energy(L: int, h: float, J: float) -> float:
    #exact energy from _square_pbc_bonds
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    N     = L * L
    dim   = 2 ** N
    bonds = _square_pbc_bonds(L)

    assert len(bonds) == 2 * N, f"Expected {2*N} bonds, got {len(bonds)}"
