import math
import torch
from typing import Tuple

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NQS] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Conjugate gradient solver
# ---------------------------------------------------------------------------
def _cg(
    A_matvec,
    b: torch.Tensor,
    tol: float = 1e-5,
    max_iter: int | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Conjugate gradient solver for A x = b.

    A is supplied as a callable v ↦ A @ v (need not be materialised).

    Returns
    -------
    (x, info)
      info = 0  — converged to ||r|| < tol
      info = k  — stopped after k iterations without converging
    """
    n = b.shape[0]
    if max_iter is None:
        max_iter = 10 * n

    x = torch.zeros_like(b)
    r = b - A_matvec(x)
    p = r.clone()
    rsold = torch.dot(r, r)

    for _ in range(max_iter):
        Ap    = A_matvec(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-30)
        x     = x + alpha * p
        r     = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if math.sqrt(rsnew.item()) < tol:
            return x, 0
        p     = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, max_iter


# ---------------------------------------------------------------------------
# RBM wave function  (geometry-agnostic)
# ---------------------------------------------------------------------------
class RBM:
    def __init__(
        self,
        N:      int,
        alpha:  float = 2.0,
        seed:   int   = 0,
        device: torch.device = DEVICE,
    ):
        torch.manual_seed(seed)
        self.N      = N
        self.M      = max(1, int(alpha * N))
        self.device = device
        scale       = 0.1
        self.W = torch.randn(N, self.M, device=device) * scale
        self.a = torch.randn(N,         device=device) * scale
        self.b = torch.randn(self.M,    device=device) * scale

    def _angles(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma @ self.W + self.b

    def log_psi(self, sigma: torch.Tensor) -> torch.Tensor:
        theta = self._angles(sigma)
        return 0.5 * (sigma @ self.a + torch.sum(torch.log(2 * torch.cosh(theta)), dim=-1))

    def local_energy_batch(
        self, configs: torch.Tensor, h: float, J: float, L: int
    ) -> torch.Tensor:
        """
        2D TFIM local energy, PBC.  configs : (Ns, N),  N = L*L.

        Diagonal (ZZ) : reshape → (Ns, L, L), roll along dim=1 and dim=2.
        Off-diagonal  : single-site flip ratios, geometry-independent.
        """
        Ns  = configs.shape[0]
        c2d = configs.reshape(Ns, L, L)

        E_horiz = torch.sum(c2d * torch.roll(c2d, -1, dims=2), dim=(1, 2))
        E_vert  = torch.sum(c2d * torch.roll(c2d, -1, dims=1), dim=(1, 2))
        E_diag  = -J * (E_horiz + E_vert)

        theta     = configs @ self.W + self.b        # (Ns, M)
        new_theta = (
            theta[:, None, :]
            - 2.0 * configs[:, :, None] * self.W[None, :, :]
        )                                            # (Ns, N, M)
        log_ratios = (
            -2.0 * configs * self.a[None, :]
            + torch.sum(
                torch.log(torch.cosh(new_theta))
                - torch.log(torch.cosh(theta))[:, None, :],
                dim=2,
            )
        )                                            # (Ns, N)
        E_offdiag = -h * torch.sum(torch.exp(0.5 * log_ratios), dim=1)
        return E_diag + E_offdiag

    def grad_log_psi(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta      = self._angles(configs)           # (Ns, M)
        tanh_theta = torch.tanh(theta)
        da = 0.5 * configs                           # (Ns, N)
        db = 0.5 * tanh_theta                        # (Ns, M)
        dW = 0.5 * configs[:, :, None] * tanh_theta[:, None, :]  # (Ns, N, M)
        return dW, da, db


# ---------------------------------------------------------------------------
# Gibbs sampler  (vectorised chains + global Z₂ flip)
# ---------------------------------------------------------------------------
class GibbsSampler:
    """
    n_chains parallel Gibbs chains.

    Global Z₂ flip: each chain flips all spins with p=0.5 after every
    sweep.  Always accepted (|Ψ(σ)|² = |Ψ(-σ)|² by Z₂ symmetry).
    Restores ergodicity in the FM phase.
    """

    def __init__(self, rbm: RBM, n_chains: int = 1, seed: int = 1):
        self.rbm      = rbm
        self.n_chains = n_chains
        self.device   = rbm.device

        u = torch.rand(n_chains, rbm.N, device=self.device)
        self.sigma = torch.where(u > 0.5, torch.ones_like(u), -torch.ones_like(u))

        u = torch.rand(n_chains, rbm.M, device=self.device)
        self.h_hidden = torch.where(u > 0.5, torch.ones_like(u), -torch.ones_like(u))

    def step(self) -> torch.Tensor:
        rbm = self.rbm

        # Hidden update  p(h_j=+1 | σ) = σ(2 θ_j)
        theta     = self.sigma @ rbm.W + rbm.b     # (n_chains, M)
        prob_h1   = torch.sigmoid(2.0 * theta)
        r         = torch.rand_like(prob_h1)
        self.h_hidden = torch.where(prob_h1 > r, torch.ones_like(r), -torch.ones_like(r))

        # Visible update  p(σ_i=+1 | h) = σ(2 ξ_i)
        xi        = self.h_hidden @ rbm.W.T + rbm.a   # (n_chains, N)
        prob_s1   = torch.sigmoid(2.0 * xi)
        r         = torch.rand_like(prob_s1)
        self.sigma = torch.where(prob_s1 > r, torch.ones_like(r), -torch.ones_like(r))

        # Global Z₂ flip — (n_chains, 1) broadcasts over N
        flip_mask  = torch.rand(self.n_chains, 1, device=self.device) > 0.5
        self.sigma = torch.where(flip_mask, -self.sigma, self.sigma)

        return self.sigma

    def sample(self, n_samples: int, n_burn: int = 100) -> torch.Tensor:
        for _ in range(n_burn):
            self.step()
        steps_needed = math.ceil(n_samples / self.n_chains)
        chunks  = [self.step() for _ in range(steps_needed)]
        configs = torch.cat(chunks, dim=0)[:n_samples]
        return configs


# ---------------------------------------------------------------------------
# VMC with Stochastic Reconfiguration (SR)
# ---------------------------------------------------------------------------
class VMC:
    """
    VMC optimiser with Stochastic Reconfiguration (SR).

    SR preconditions the gradient by the quantum geometric tensor S:

      S_{kk'} = ⟨D_k D_k'⟩ − ⟨D_k⟩⟨D_k'⟩

    and solves  S · δp = g  at each step instead of using g directly.

    Regularisation:  S → S + ε·I  (ε decays exponentially from eps_start → eps_end)
    """

    def __init__(
        self,
        rbm:       RBM,
        L:         int,
        h:         float = 1.0,
        J:         float = 1.0,
        n_samples: int   = 500,
        eta:       float = 0.05,
        eps_start: float = 0.2,
        eps_end:   float = 0.001,
        max_norm:  float = 10.0,
        n_burn:    int   = 1,
    ):
        self.rbm       = rbm
        self.h_field   = h
        self.J         = J
        self.L         = L
        self.n_samples = n_samples
        self.eta       = eta
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.max_norm  = max_norm   # bugfix: was accepted but never stored in v3
        self.n_burn    = n_burn     # bugfix: same
        self.sampler   = GibbsSampler(rbm, n_chains=n_samples)

    # ------------------------------------------------------------------
    def _flatten_grads(
        self, dW: torch.Tensor, da: torch.Tensor, db: torch.Tensor
    ) -> torch.Tensor:
        """Flatten (Ns, N, M), (Ns, N), (Ns, M) → (Ns, n_params)."""
        Ns = dW.shape[0]
        return torch.cat([dW.reshape(Ns, -1), da, db], dim=1)

    def _unflatten_update(self, delta: torch.Tensor, rbm: RBM) -> None:
        """
        Split flat (n_params,) update back into W, a, b and apply in-place.
        """
        NM    = rbm.N * rbm.M
        dW    = delta[:NM].reshape(rbm.N, rbm.M)
        da    = delta[NM : NM + rbm.N]
        db    = delta[NM + rbm.N :]
        rbm.W = rbm.W - self.eta * dW
        rbm.a = rbm.a - self.eta * da
        rbm.b = rbm.b - self.eta * db

    def _eps(self, t: int, n_steps: int) -> float:
        """
        Exponential decay: eps_start → eps_end over n_steps.

        Keeps ε large for ~60 % of training; only drops sharply in the
        final 20–30 % when the RBM has settled and S_kk is better behaved.
        """
        ratio = self.eps_end / max(self.eps_start, 1e-30)
        return self.eps_start * (ratio ** (t / max(n_steps, 1)))

    # ------------------------------------------------------------------
    def step(self, eps: float) -> float:
        """
        One SR optimisation step.

        Parameters
        ----------
        eps : float
            Regularisation added to the diagonal of S this step.

        Returns
        -------
        float — estimated energy ⟨H⟩/N (per site)
        """
        rbm  = self.rbm
        configs = self.sampler.sample(self.n_samples, n_burn=50)
        N_s     = self.n_samples

        E_loc_arr              = rbm.local_energy_batch(
            configs, self.h_field, self.J, self.L   # bugfix: self.L, not bare L
        )
        dW_arr, da_arr, db_arr = rbm.grad_log_psi(configs)

        E_mean = E_loc_arr.mean()

        # Raw gradient  g_k = 2(⟨E D_k⟩ − ⟨E⟩⟨D_k⟩)
        gW = 2.0 * (
            torch.einsum("i,ijk->jk", E_loc_arr, dW_arr) / N_s
            - E_mean * dW_arr.mean(dim=0)
        )
        ga = 2.0 * (
            torch.einsum("i,ij->j", E_loc_arr, da_arr) / N_s
            - E_mean * da_arr.mean(dim=0)
        )
        gb = 2.0 * (
            torch.einsum("i,ij->j", E_loc_arr, db_arr) / N_s
            - E_mean * db_arr.mean(dim=0)
        )
        g_flat = torch.cat([gW.reshape(-1), ga, gb])       # (n_params,)

        # Global gradient norm clipping  (preserves direction)
        g_norm = torch.linalg.norm(g_flat).item()
        if g_norm > self.max_norm:
            g_flat = g_flat * (self.max_norm / g_norm)

        # Build S from flattened derivatives  O_k[s] = D_k(σ_s)
        O          = self._flatten_grads(dW_arr, da_arr, db_arr)  # (Ns, n_params)
        O_mean     = O.mean(dim=0)                                  # (n_params,)
        O_centered = O - O_mean                                     # (Ns, n_params)

        n_params = g_flat.shape[0]

        def S_matvec(v: torch.Tensor) -> torch.Tensor:
            """Matrix-free S @ v using the centred derivative matrix."""
            return (O_centered.T @ (O_centered @ v)) / N_s + eps * v

        delta, info = _cg(S_matvec, g_flat, tol=1e-5, max_iter=2 * n_params)

        if info > 0:
            print(f"  Warning: CG did not converge after {info} iterations")

        self._unflatten_update(delta, rbm)
        return (E_mean / rbm.N).item()

    # ------------------------------------------------------------------
    def run(self, n_steps: int = 300, print_every: int = 50) -> list:
        """
        Run the full VMC + SR optimisation.

        Returns
        -------
        list of (step, energy_per_site, eps) tuples
        """
        history = []
        print(f"{'Step':>6}  {'E/N':>10}  {'σ(E)':>10}  {'eps':>8}")
        print("-" * 42)

        for t in range(1, n_steps + 1):
            eps = self._eps(t - 1, n_steps - 1)

            # Measurement batch
            meas_configs = self.sampler.sample(self.n_samples, n_burn=5)
            E_loc_meas   = self.rbm.local_energy_batch(
                meas_configs, self.h_field, self.J, self.L
            )
            E_mean = (E_loc_meas.mean() / self.rbm.N).item()
            E_err  = (
                E_loc_meas.std() / (math.sqrt(self.n_samples) * self.rbm.N)
            ).item()

            # SR gradient step on a fresh independent batch
            self.step(eps)

            history.append((t, E_mean, eps))
            if t % print_every == 0 or t == 1:
                print(f"{t:>6}  {E_mean:>10.5f}  {E_err:>10.5f}  {eps:>8.5f}")

        return history


# ---------------------------------------------------------------------------
# Exact diagonalisation — matrix-free Lanczos, no TeNPy, no hard N limit
# ---------------------------------------------------------------------------
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
            bonds.append((i,  x * L          + (y + 1) % L))   # horizontal →
            bonds.append((i, ((x + 1) % L) * L + y          ))  # vertical   ↓
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

    # ------------------------------------------------------------------
    # GPU path via CuPy
    # ------------------------------------------------------------------
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.sparse.linalg import (
                LinearOperator as CuLinearOperator,
                eigsh           as cu_eigsh,
            )

            print(f"  [ed] GPU Lanczos via CuPy  (dim = {dim:,})")

            states_gpu = cp.arange(dim, dtype=cp.int64)
            diag_gpu   = cp.zeros(dim, dtype=cp.float64)
            for bi, bj in bonds:
                sz_i = 2.0 * ((states_gpu >> bi) & 1) - 1.0
                sz_j = 2.0 * ((states_gpu >> bj) & 1) - 1.0
                diag_gpu -= J * sz_i * sz_j

            def matvec_gpu(v: cp.ndarray) -> cp.ndarray:
                out = diag_gpu * v
                for k in range(N):
                    out -= h * v[states_gpu ^ (1 << k)]
                return out

            H_op = CuLinearOperator(
                (dim, dim), matvec=matvec_gpu, dtype=cp.float64
            )
            E0 = cu_eigsh(
                H_op, k=1, which="SA",
                return_eigenvectors=False,
                tol=1e-10, maxiter=3000,
            )[0]
            return float(E0)

        except ImportError:
            print("  [ed] CuPy not found — falling back to CPU Lanczos.")
        except Exception as exc:
            print(f"  [ed] CuPy error ({exc}) — falling back to CPU Lanczos.")

    # ------------------------------------------------------------------
    # CPU path via NumPy + SciPy ARPACK
    # ------------------------------------------------------------------
    print(f"  [ed] CPU Lanczos  (dim = {dim:,})")

    states = np.arange(dim, dtype=np.int64)
    diag   = np.zeros(dim, dtype=np.float64)
    for bi, bj in bonds:
        sz_i = 2.0 * ((states >> bi) & 1) - 1.0
        sz_j = 2.0 * ((states >> bj) & 1) - 1.0
        diag -= J * sz_i * sz_j

    def matvec(v: np.ndarray) -> np.ndarray:
        """H @ v without storing H.  O(N × dim) per call."""
        out = diag * v                              # diagonal ZZ part
        for k in range(N):
            out -= h * v[states ^ (1 << k)]        # σˣ flip at site k
        return out

    H_op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    E0   = eigsh(
        H_op, k=1, which="SA",
        return_eigenvectors=False,
        tol=1e-10, maxiter=3000,
    )[0]
    return float(E0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # -----------------------------------------------------------------------
    L       = 3           # lattice side → N = L×L spins
    N       = L * L
    alpha   = 4.0
    h_field = 1.0         # 2D TFIM critical point: h ≈ 3.04 J
    J_field = 1.0         # h=J=1 → deep FM phase; h=3.04 → near criticality
    n_steps = 400
    n_samp  = 2000
    eta     = 0.01
    max_norm = 10.0
    n_burn   = 100

    n_params  = N * int(alpha * N) + N + int(alpha * N)
    eps_start = 2 * n_params / n_samp
    eps_end   = eps_start * 0.01
    # -----------------------------------------------------------------------

    print("=" * 55)
    print(f"NQS-RBM 2D  |  L={L}, N={N}, α={alpha}, h={h_field}, J={J_field}")
    print(f"n_params={n_params}, Ns={n_samp}")
    print(f"eps: {eps_start:.4f} → {eps_end:.4f}  (exponential decay)")
    print(f"eta={eta}, max_norm={max_norm}, n_burn={n_burn}")
    print("=" * 55)

    print("Computing ED reference (matrix-free Lanczos)...", end=" ", flush=True)
    t_ed    = time.time()
    E_exact = ed_exact_energy(L, h_field, J_field) / N
    print(f"done ({time.time() - t_ed:.1f}s)")
    print(f"ED exact  E/N = {E_exact:.8f}\n")

    rbm = RBM(N=N, alpha=alpha, device=DEVICE)
    vmc = VMC(
        rbm, L=L, h=h_field, J=J_field,
        n_samples=n_samp, eta=eta,
        eps_start=eps_start, eps_end=eps_end,
        max_norm=max_norm, n_burn=n_burn,
    )

    t0      = time.time()
    history = vmc.run(n_steps=n_steps, print_every=50)
    elapsed = time.time() - t0

    final_E = history[-1][1]
    rel_err = abs(final_E - E_exact) / abs(E_exact) * 100
    print(f"\nFinal VMC E/N  = {final_E:.6f}")
    print(f"ED exact  E/N  = {E_exact:.6f}")
    print(f"Relative error = {rel_err:.4f}%")
    print(f"Wall time      = {elapsed:.1f}s")
