"""
Neural Network Quantum States (NQS) — RBM ansatz
Variational Monte Carlo for the 1D transverse-field Ising model

  H = -h ∑ᵢ σˣᵢ  -  J ∑ᵢ σᶻᵢ σᶻᵢ₊₁

with periodic boundary conditions.

Reference: Carleo & Troyer, Science 355, 602–606 (2017)
           Lecture notes by G. Carleo (Beijing 2017)

Architecture
------------
  Ψ(σ) = sqrt(F_rbm(σ))   [positive-definite ground state]

  F_rbm(σ) = exp(∑ᵢ aᵢ σᵢ) × ∏ⱼ 2 cosh(∑ᵢ Wᵢⱼ σᵢ + bⱼ)

Parameters: W (N×M), a (N,), b (M,)  — all real-valued here.

Algorithm
---------
1. Initialise W, a, b small random.
2. For each optimisation step:
   a. Gibbs-sample N_s spin configurations from |Ψ(σ)|² = F_rbm(σ).
   b. Compute local energy E_loc(σ) = ∑_σ' H_σσ' Ψ(σ')/Ψ(σ) for each sample.
   c. Compute variational derivatives D_k(σ) = ∂_pk ln Ψ(σ).
   d. Estimate gradient G_k = 2 Re[⟪E_loc D_k*⟫ − ⟪E_loc⟫⟪D_k*⟫].
   e. Update parameters: p ← p − η G_k.
"""

import numpy as np
from typing import Tuple

ef train_2d(L, J, h, alpha=2, n_steps=300, n_samples=300,
             eta=0.02, seed=16):
    """
    Train one RBM on the 2D L×L TFI model.

    Returns
    -------
    energy_history   : list of E/N estimates per step
    variance_history : list of var(E_loc) per step
    e_exact          : exact E/N (None if L > 4, since 2^(L²) blows up fast)
    rbm              : trained RBM
    L                : lattice side length (passed through for convenience)
    """
    N = L * L

    e_exact = None
    if N <= 16:
        print(f"    Computing exact ground state (N={N}, dim={2**N:,})...",
              end=' ', flush=True)
        e_exact = ed_exact_energy(L, h, J) / N
        print(f"E/N = {e_exact:.6f}")

    rbm     = RBM(N=N, alpha=alpha, seed=seed)
    vmc     = VMC(rbm, L, h=h, J=J, n_samples=n_samples, eta=eta)

    energy_history   = []
    variance_history = []

    for step in range(n_steps):
        with torch.no_grad():
            configs = vmc.sampler.sample(n_samples, n_burn=5)
            e_locs  = rbm.local_energy_batch(configs, h, J, L)
            e_mean  = (e_locs.mean() / N).item()
            e_var   = e_locs.var().item()

        eps = vmc._eps(step, n_steps - 1)
        vmc.step(eps)

        energy_history.append(e_mean)
        variance_history.append(e_var)

    return energy_history, variance_history, e_exact, rbm
# ---------------------------------------------------------------------------
# RBM wave function
# ---------------------------------------------------------------------------

class RBM:
    """
    Restricted Boltzmann Machine representing Ψ(σ) = sqrt(F_rbm(σ)).

    Attributes
    ----------
    N : int   — number of visible (physical spin) units
    M : int   — number of hidden units  (α = M/N is the hidden density)
    W : (N, M) float array  — visible-to-hidden weights
    a : (N,)  float array   — visible biases
    b : (M,)  float array   — hidden biases
    """

    def __init__(self, N: int, alpha: float = 1.0, seed: int = 16):
        """
        Parameters
        ----------
        N     : number of spins
        alpha : hidden unit density M/N  (paper uses α = 1–4)
        seed  : RNG seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        self.N = N
        self.M = max(1, int(alpha * N))

        # Small random initialisation — important for avoiding local minima.
        scale = 0.1
        self.W = rng.normal(0, scale, (N, self.M))
        self.a = rng.normal(0, scale, N)
        self.b = rng.normal(0, scale, self.M)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _angles(self, sigma: np.ndarray) -> np.ndarray:
        """θⱼ(σ) = ∑ᵢ Wᵢⱼ σᵢ + bⱼ  —  shape (..., M)"""
        return sigma @ self.W + self.b          # broadcast over batch

    def log_psi(self, sigma: np.ndarray) -> np.ndarray:
        """
        ln Ψ(σ) = (1/2) ln F_rbm(σ)
               = (1/2)[∑ᵢ aᵢ σᵢ  +  ∑ⱼ ln 2cosh(θⱼ)] -> turns mutiplication into summation

        Parameters
        ----------
        sigma : (..., N)  spin configuration(s) with values ±1

        Returns
        -------
        (...,) float  — log of wave-function amplitude
        """
        theta = self._angles(sigma)
        return 0.5 * (sigma @ self.a + np.sum(np.log(2 * np.cosh(theta)), axis=-1))



    def psi_ratio(self, sigma: np.ndarray, flip_site: int) -> float:
        """
        Ψ(σ')/Ψ(σ)  where σ' = σ with spin k flipped.

        From eq. (2.23) in the notes:
          Ψ(σ'(k))/Ψ(σ) = exp(2 a_k σ_k) × ∏ⱼ cosh(θⱼ − 2σ_k W_kj)/cosh(θⱼ)

        Parameters
        ----------
        sigma     : (N,)  current spin config
        flip_site : index k of spin to flip

        Returns
        -------
        float — ratio of amplitudes
        """
        theta = self._angles(sigma)                          # (M,)
        sigma_k = sigma[flip_site]
        log_ratio = (-2.0 * self.a[flip_site] * sigma_k
                     + np.sum(np.log(np.cosh(theta - 2.0 * sigma_k * self.W[flip_site])
                                     / np.cosh(theta))))
        return np.exp(0.5 * log_ratio)                       # sqrt because Ψ = sqrt(F)

    # ------------------------------------------------------------------
    # Local energy  E_loc(σ) = ∑_σ' H_σσ' Ψ(σ')/Ψ(σ)
    # ------------------------------------------------------------------
    """
    def local_energy(self, sigma: np.ndarray, h: float, J: float) -> float:

        Local energy for the 1D transverse-field Ising Hamiltonian
        H = -h ∑ᵢ σˣᵢ  -  J ∑ᵢ σᶻᵢ σᶻᵢ₊₁  (periodic BC)

        Diagonal part: -J ∑ᵢ σᵢ σᵢ₊₁
        Off-diagonal:  -h ∑ᵢ  Ψ(σ^(k))/Ψ(σ)   where σ^(k) flips spin k

        Parameters
        ----------
        sigma : (N,)  current spin config ±1
        h     : transverse field strength
        J     : Ising coupling

        Returns
        -------
        float — local energy estimator

        N = self.N
        # Diagonal: Ising ZZ term (periodic BC via modulo)
        E_diag = -J * np.sum(sigma * np.roll(sigma, -1))

        # Off-diagonal: transverse field flips each spin
        E_offdiag = 0.0
        for k in range(N):
            E_offdiag += -h * self.psi_ratio(sigma, k)

        return E_diag + E_offdiag
    """
    def _energy_diag(self, sigma: np.ndarray, J: float) -> float:
        return -J*np.sum(sigma * np.roll(sigma, -1))

    def _energy_offdiag(self, sigma: np.ndarray, h: float) -> float:
        return sum(-h * self.psi_ratio(sigma, k) for k in range(self.N))

    def local_energy(self, sigma: np.ndarray, h: float, J: float) -> float:
        return self._energy_diag(sigma, J) + self._energy_offdiag(sigma, h)
    # ------------------------------------------------------------------
    # Variational derivatives  D_k(σ) = ∂_pk ln Ψ(σ)
    # ------------------------------------------------------------------

    def grad_log_psi(self, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Variational derivatives of ln Ψ(σ) w.r.t. all parameters.

        From eqs. (2.25–2.27):
          ∂_{aᵢ} ln Ψ = (1/2) σᵢ
          ∂_{bⱼ} ln Ψ = (1/2) tanh(θⱼ)
          ∂_{Wᵢⱼ} ln Ψ = (1/2) tanh(θⱼ) σᵢ

        Returns
        -------
        dW : (N, M)   derivative w.r.t. W
        da : (N,)     derivative w.r.t. a
        db : (M,)     derivative w.r.t. b
        """
        theta = self._angles(sigma)          # (M,)
        tanh_theta = np.tanh(theta)          # (M,)

        da = 0.5 * sigma                                       # (N,)
        db = 0.5 * tanh_theta                                  # (M,)
        dW = 0.5 * np.outer(sigma, tanh_theta)                 # (N, M)

        return dW, da, db


# ---------------------------------------------------------------------------
# Gibbs sampler  (alternate Gibbs sampling — Sec 2.3.1)
# ---------------------------------------------------------------------------

class GibbsSampler:
    """
    Samples spin configs from |Ψ(σ)|² = F_rbm(σ) via alternate Gibbs steps.

    Each full sweep:
      1. Update all N spins in parallel given current h.
      2. Update all M hidden units in parallel given current σ.

    The acceptance probability is exactly 1 at every step (eq. 2.14),
    so no Metropolis rejection test is needed.
    """

    def __init__(self, rbm: RBM, seed: int = 0):
        self.rbm = rbm
        self.rng = np.random.default_rng(seed)
        # Initialise σ randomly
        self.sigma = self.rng.choice([-1, 1], size=rbm.N).astype(float)
        self.h_hidden = self.rng.choice([-1, 1], size=rbm.M).astype(float)

    def _logistic(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def step(self) -> np.ndarray:
        """One full Gibbs sweep; returns updated σ."""
        rbm = self.rbm

        # --- Update hidden units given σ (eq. 2.17) ---
        theta = self.sigma @ rbm.W + rbm.b              # (M,)
        prob_h1 = self._logistic(2.0 * theta)            # P(hⱼ=+1 | σ)
        r = self.rng.uniform(size=rbm.M)
        self.h_hidden = np.where(prob_h1 > r, 1.0, -1.0)

        # --- Update visible units given h (eq. 2.20) ---
        xi = self.h_hidden @ rbm.W.T + rbm.a            # (N,)   ξᵢ = ∑ⱼ Wᵢⱼ hⱼ + aᵢ
        prob_s1 = self._logistic(2.0 * xi)               # P(σᵢ=+1 | h)
        r = self.rng.uniform(size=rbm.N)
        self.sigma = np.where(prob_s1 > r, 1.0, -1.0)

        return self.sigma.copy()

    def sample(self, n_samples: int, n_burn: int = 100) -> np.ndarray:
        """
        Generate n_samples configurations after burning n_burn steps.

        Returns
        -------
        (n_samples, N) float array of ±1 spin configs
        """
        for _ in range(n_burn):
            self.step()
        configs = np.empty((n_samples, self.rbm.N))
        for i in range(n_samples):
            configs[i] = self.step()
        return configs


# ---------------------------------------------------------------------------
# Variational Monte Carlo optimiser
# ---------------------------------------------------------------------------

class VMC:
    """
    Stochastic gradient descent on the variational energy ⟨H⟩.

    Gradient estimator (eq. 1.15):
      G_k = 2 Re[⟪E_loc D_k*⟫ − ⟪E_loc⟫ ⟪D_k*⟫]

    For real parameters this simplifies to:
      G_k ≈ 2 (⟨E_loc D_k⟩ − ⟨E_loc⟩ ⟨D_k⟩)
    """

    def __init__(self, rbm: RBM, h: float = 1.0, J: float = 1.0,
                 n_samples: int = 500, eta: float = 0.05):
        """
        Parameters
        ----------
        rbm       : RBM ansatz to optimise
        h         : transverse field (paper's h, not hidden units)
        J         : Ising coupling
        n_samples : MC samples per gradient step
        eta       : learning rate
        """
        self.rbm = rbm
        self.h_field = h
        self.J = J
        self.n_samples = n_samples
        self.eta = eta
        self.sampler = GibbsSampler(rbm)

    def step(self) -> float:
        """
        One optimisation step.

        Returns
        -------
        float — estimated energy ⟨H⟩/N (per site)
        """
        rbm = self.rbm
        configs = self.sampler.sample(self.n_samples, n_burn=10)  # (Ns, N)

        N_s = self.n_samples
        # Accumulators
        E_loc_arr = np.zeros(N_s)
        dW_arr = np.zeros((N_s, rbm.N, rbm.M))
        da_arr = np.zeros((N_s, rbm.N))
        db_arr = np.zeros((N_s, rbm.M))

        for i, sigma in enumerate(configs):
            E_loc_arr[i] = rbm.local_energy(sigma, self.h_field, self.J)
            dW_arr[i], da_arr[i], db_arr[i] = rbm.grad_log_psi(sigma)

        E_mean = E_loc_arr.mean()

        # Gradient: G_k = 2(⟨E D_k⟩ − ⟨E⟩ ⟨D_k⟩)
        def _grad(D_arr):
            return 2.0 * ((E_loc_arr[:, None if D_arr.ndim > 1 else None] * D_arr).mean(axis=0)
                          - E_mean * D_arr.mean(axis=0))

        gW = 2.0 * (np.einsum('i,ijk->jk', E_loc_arr, dW_arr) / N_s
                    - E_mean * dW_arr.mean(axis=0))
        ga = 2.0 * (np.einsum('i,ij->j', E_loc_arr, da_arr) / N_s
                    - E_mean * da_arr.mean(axis=0))
        gb = 2.0 * (np.einsum('i,ij->j', E_loc_arr, db_arr) / N_s
                    - E_mean * db_arr.mean(axis=0))

        # Gradient descent update
        rbm.W -= self.eta * gW
        rbm.a -= self.eta * ga
        rbm.b -= self.eta * gb

        return E_mean / rbm.N     # energy per site

    def run(self, n_steps: int = 700, print_every: int = 100) -> list:
        """
        Run the full VMC optimisation.

        Returns
        -------
        list of (step, energy_per_site) tuples
        """
        history = []
        print(f"{'Step':>6}  {'E/N':>10}  {'σ(E)':>10}")
        print("-" * 32)
        for step in range(1, n_steps + 1):
            # Collect Ns samples, compute E_loc for error estimate too
            configs = self.sampler.sample(self.n_samples, n_burn=5)
            E_samples = np.array([
                self.rbm.local_energy(s, self.h_field, self.J)
                for s in configs
            ])
            E_mean = E_samples.mean() / self.rbm.N
            E_err = E_samples.std() / (np.sqrt(self.n_samples) * self.rbm.N)

            # Gradient step (uses fresh samples internally)
            self.step()

            history.append((step, E_mean))
            if step % print_every == 0 or step == 1:
                print(f"{step:>6}  {E_mean:>10.5f}  {E_err:>10.5f}")

        return history


# ---------------------------------------------------------------------------
# Exact diagonalisation (for small N, as reference)
# ---------------------------------------------------------------------------

def exact_ground_state(N: int, h: float, J: float) -> float:
    """
    Build the full 2^N × 2^N Hamiltonian and return the ground-state energy.
    Only feasible for N ≤ ~16.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim))

    # Pauli matrices
    sz = np.array([[1., 0.], [0., -1.]])
    sx = np.array([[0., 1.], [1.,  0.]])
    I2 = np.eye(2)

    def kron_op(op, site, n):
        """n-site operator with 'op' on 'site'."""
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    # Transverse field: -h ∑ᵢ σˣᵢ
    for i in range(N):
        H -= h * kron_op(sx, i, N)

    # Ising ZZ: -J ∑ᵢ σᶻᵢ σᶻᵢ₊₁  (periodic BC)
    for i in range(N):
        j = (i + 1) % N
        ops = [I2] * N
        ops[i] = sz
        ops[j] = sz
        op_ij = ops[0]
        for o in ops[1:]:
            op_ij = np.kron(op_ij, o)
        H -= J * op_ij

    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    N        = 10       # number of spins
    alpha    = 2        # hidden density M/N
    h_field  = 1.0      # transverse field (phase transition at h=J)
    J_field  = 1.0      # Ising coupling
    n_steps  = 400      # optimisation steps
    n_samp   = 300      # MC samples per step
    eta      = 0.02     # learning rate
    # -----------------------------------------------------------------------

    print("=" * 50)
    print(f"NQS-RBM  |  N={N}, α={alpha}, h={h_field}, J={J_field}")
    print("=" * 50)

    # Exact ground state energy (reference)
    if N <= 14:
        E_exact = exact_ground_state(N, h_field, J_field) / N
        print(f"Exact E/N = {E_exact:.6f}  (from full diagonalisation)\n")
    else:
        E_exact = None
        print("N too large for exact diagonalisation.\n")

    # Build RBM and run VMC
    rbm = RBM(N=N, alpha=alpha)
    vmc = VMC(rbm, h=h_field, J=J_field, n_samples=n_samp, eta=eta)

    t0 = time.time()
    history = vmc.run(n_steps=n_steps, print_every=50)
    elapsed = time.time() - t0

    final_E = history[-1][1]
    print(f"\nFinal VMC E/N = {final_E:.6f}")
    if E_exact is not None:
        rel_err = abs(final_E - E_exact) / abs(E_exact) * 100
        print(f"Exact   E/N = {E_exact:.6f}")
        print(f"Relative error = {rel_err:.3f}%")
    print(f"Wall time: {elapsed:.1f}s")
