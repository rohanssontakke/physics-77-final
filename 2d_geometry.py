#2D System: Will need to take a lattice instead of a one-dimensional input array, and then we can flatten it in order to feed it to the same functions we've already made

def get_neighbors_2d(L):
    """
    For an L x L square lattice with periodic boundary conditions,
    returns a list of (i, j) pairs where i and j are neighboring site indices
    in the flat spin vector.

    Each site has 4 neighbors: up, down, left, right.
    """
    neighbors = []
    for row in range(L):
        for col in range(L):
            i = row * L + col  # flat index of this site
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
    """
    All nearest-neighbour bonds for an L×L square lattice with PBC.

    Site index: i = x * L + y,  x = row, y = column.
    Each site connects rightward (y+1) and downward (x+1), both mod L.
    This gives exactly 2*N unique bonds (N = L*L).
    """
    bonds = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            bonds.append((i,  x * L          + (y + 1) % L)) 
            bonds.append((i, ((x + 1) % L) * L + y          ))  
    return bonds


def ed_exact_energy(L: int, h: float, J: float) -> float:
    """
    Ground state energy of the 2D TFIM on an L×L lattice (PBC).

    No external dependencies beyond NumPy + SciPy (+ optional CuPy).
    TeNPy is no longer required — the NN bond list is built analytically
    for the square lattice via _square_pbc_bonds().

    H is never stored as a matrix.  It is applied on-the-fly as a
    scipy LinearOperator and ARPACK (Lanczos) finds E₀.

    Practical memory ceiling (O(2^N) per Lanczos vector):
        L=4 →  65 536 dim,  ~0.5 MB per vector   ✓
        L=5 →  33 M   dim,  ~260 MB per vector   ✓  (~30 s CPU)
        L=6 →  68 G   dim,  ~550 GB per vector   ✗  (RAM limited)

    GPU acceleration (optional):
        pip install cupy-cuda12x
      If CuPy is available the diagonal and matvec run on GPU and
      cupyx.scipy.sparse.linalg.eigsh drives the Lanczos.
      Falls back to SciPy CPU automatically if CuPy is absent.

    Install (no TeNPy needed):
        pip install scipy
        pip install cupy-cuda12x   # optional
    """
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    N     = L * L
    dim   = 2 ** N
    bonds = _square_pbc_bonds(L)

    assert len(bonds) == 2 * N, f"Expected {2*N} bonds, got {len(bonds)}"
