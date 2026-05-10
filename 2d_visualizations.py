def plot_2d_convergence_vs_h(L=3, J=1.0, h_values=None,
                              alpha=2, n_steps=300, n_samples=300, eta=0.02):
    """
    Train a fresh 2D RBM at each h/J value and overlay convergence.
 
    Default L=3 (N=9 spins) — small enough for exact diag reference.
    For L=4 (N=16) exact diag is feasible but slow (~30s CPU).
    For L≥5 exact diag is disabled and you rely on variance only.
 
    Key comparison vs 1D:
    - Curves are noisier (stochastic SR vs exact enumeration)
    - Critical point plateau is more pronounced
    - Variance floor is higher — 2D states are harder to compress
    """
    if h_values is None:
        h_values = [0.3, 0.5, 1.0, 1.5, 2.5]
 
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(h_values)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'2D TFI Convergence vs h/J  (L={L}×{L}, N={L*L}, J={J}, α={alpha})',
                 fontsize=13, fontweight='bold')
 
    final_errors = []
 
    for idx, h in enumerate(h_values):
        print(f"\n  Training h/J={h/J:.1f}...", flush=True)
        energies, variances, e_exact, _ = train_2d(
            L, J, h, alpha, n_steps, n_samples, eta)
 
        color = colors[idx]
        label = f'h/J={h/J:.1f}'
        steps = np.arange(len(energies))
        w     = max(10, n_steps // 20)
 
        avg_e = np.convolve(energies, np.ones(w)/w, mode='valid')
        avg_v = np.convolve(variances, np.ones(w)/w, mode='valid')
        x_avg = np.arange(w-1, len(energies))
 
        axes[0].plot(steps, energies, alpha=0.15, color=color, lw=0.8)
        axes[0].plot(x_avg, avg_e, color=color, lw=2.0, label=label)
        if e_exact is not None:
            axes[0].axhline(e_exact, color=color, ls='--', lw=1.0, alpha=0.6)
 
        axes[1].semilogy(steps, variances, alpha=0.15, color=color, lw=0.8)
        axes[1].semilogy(x_avg, avg_v, color=color, lw=2.0, label=label)
 
        if e_exact is not None:
            final_e = np.mean(energies[-30:])
            final_errors.append((h/J, abs(final_e - e_exact) / abs(e_exact) * 100))
 
    axes[0].set_xlabel('Optimisation Step', fontsize=11)
    axes[0].set_ylabel('E/N  (energy per site)', fontsize=11)
    axes[0].set_title('Energy Convergence\n(faint = raw, solid = smoothed, dashed = exact)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
 
    axes[1].set_xlabel('Optimisation Step', fontsize=11)
    axes[1].set_ylabel('var(E_loc)  [log scale]', fontsize=11)
    axes[1].set_title('Variance Decay\n(→0 = true eigenstate; higher floor than 1D)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
 
    plt.tight_layout()
    plt.show()
 
    if final_errors:
        print(f"\n{'h/J':>8}  {'Final rel. error %':>20}")
        print("-" * 32)
        for hj, err in final_errors:
            marker = " ← critical point" if abs(hj - 1.0) < 0.05 else ""
            print(f"{hj:>8.2f}  {err:>20.4f}{marker}")
 
 
def plot_2d_residuals(L=3, J=1.0, h=1.0, alpha=2,
                      n_steps=400, n_samples=300, eta=0.02):
    """
    Signed residual E(t)/N - E_exact/N on log scale (requires L≤4).
 
    Left : residual vs step — plateau shape reveals loss landscape flatness.
    Right: residual vs variance scatter — tight diagonal = theory prediction.
 
    Compare directly to 1D residual plot:
    - Plateau should be longer in 2D (harder problem)
    - Scatter plot more spread (MC noise from sampling rather than exact enum)
    - Late-stage decay rate should be slower
    """
    N = L * L
    if N > 16:
        print(f"L={L} → N={N} > 16: exact diag not feasible, skipping residual plot.")
        return
 
    print(f"  Computing exact ground state for L={L}×{L}...", flush=True)
    e_exact = ed_exact_energy(L, h, J) / N
 
    energies, variances, _, _ = train_2d(
        L, J, h, alpha, n_steps, n_samples, eta)
 
    residuals = np.array(energies) - e_exact 
 
    w       = max(10, n_steps // 20)
    avg_res = np.convolve(residuals, np.ones(w)/w, mode='valid')
    x_avg   = np.arange(w-1, len(residuals))
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'2D RBM Signed Residuals  —  TFI  (L={L}×{L}, J={J}, h={h}, α={alpha})',
        fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.semilogy(residuals, alpha=0.2, color='steelblue', lw=0.7, label='raw')
    ax.semilogy(x_avg, avg_res, color='navy', lw=2.5, label='smoothed')
    ax.set_xlabel('Optimisation Step')
    ax.set_ylabel('E(t)/N − E_exact/N  [log scale]')
    ax.set_title('Energy Residual\n(always ≥ 0 by variational principle)')
    ax.legend()
    ax.grid(alpha=0.3)
 
    half   = len(avg_res) // 2
    log_r  = np.log(avg_res[half:] + 1e-12)
    if np.all(np.isfinite(log_r)) and len(log_r) > 5:
        coeffs = np.polyfit(np.arange(len(log_r)), log_r, 1)
        decay  = (1 - np.exp(coeffs[0] * 100)) * 100
        ax.text(0.98, 0.95,
                f'late-stage decay\n≈{decay:.1f}% per 100 steps',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
 

    ax = axes[1]
    sc = ax.scatter(variances, residuals,
                    c=np.arange(len(variances)), cmap='viridis', s=8, alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('var(E_loc)  [log]')
    ax.set_ylabel('E/N − E_exact/N  [log]')
    ax.set_title('Residual vs Variance\n(color = training step)\n'
                 'Diagonal = var ∝ error (theoretical prediction)')
    plt.colorbar(sc, ax=ax, label='step')
    ax.grid(alpha=0.3)
 
    x_ref = np.logspace(np.log10(min(variances) + 1e-12),
                        np.log10(max(variances)), 50)
    scale = np.median(np.array(residuals) / (np.array(variances) + 1e-12))
    ax.plot(x_ref, scale * x_ref, 'r--', lw=1.5, alpha=0.7,
            label='var ∝ residual')
    ax.legend(fontsize=9)
 
    plt.tight_layout()
    plt.show()
 
def plot_2d_spin_correlations(L=3, J=1.0, h_values=None,
                               alpha=2, n_steps=300, n_samples=500, eta=0.02):
    """
    2D spin-spin correlations ⟨σ₀ σᵣ⟩.
 
    Two panels per h/J value:
      Left  — 2D heatmap: correlation of each lattice site with site (0,0)
               Exact (if N≤16) and RBM side by side.
      Right — 1D slice: correlations along the top row (y=0, x=0..L-1)
               Direct comparison to 1D correlation plots.
 
    Key insight for 2D:
    - Ferromagnetic phase: entire lattice correlated, flat heatmap near +1
    - Paramagnetic phase: fast decay in both x and y directions
    - Critical point: slow decay, anisotropic if model hasn't learned 2D geometry
    """
    if h_values is None:
        h_values = [0.5, 1.0, 2.0]
 
    N        = L * L
    has_exact = N <= 16
 
    for h in h_values:
        print(f"\n  h/J={h/J:.1f} — training 2D RBM...", flush=True)
        _, _, _, rbm = train_2d(L, J, h, alpha, n_steps, n_samples, eta)
 
        sampler     = GibbsSampler(rbm, n_chains=1, seed=99)
        big_samples = sampler.sample(3000, n_burn=200).cpu().numpy()  
 
        corr_rbm_2d = np.zeros((L, L))
        corr_err_2d = np.zeros((L, L))
        for x in range(L):
            for y in range(L):
                r        = x * L + y
                products = big_samples[:, 0] * big_samples[:, r]
                corr_rbm_2d[x, y] = products.mean()
                corr_err_2d[x, y] = products.std() / np.sqrt(len(products))

        corr_exact_2d = None
        if has_exact:
            from scipy.sparse.linalg import LinearOperator, eigsh
            bonds = _square_pbc_bonds(L)
            dim   = 2 ** N
            states = np.arange(dim, dtype=np.int64)
            diag   = np.zeros(dim)
            for bi, bj in bonds:
                sz_i = 2.0 * ((states >> bi) & 1) - 1.0
                sz_j = 2.0 * ((states >> bj) & 1) - 1.0
                diag -= J * sz_i * sz_j
 
            def matvec(v):
                out = diag * v
                for k in range(N):
                    out -= h * v[states ^ (1 << k)]
                return out
 
            H_op   = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
            _, evecs = eigsh(H_op, k=1, which='SA', tol=1e-10, maxiter=3000,
                             return_eigenvectors=True)
            psi    = evecs[:, 0]
            probs  = psi ** 2
 
            all_cfg = np.array(
                [[2 * ((i >> k) & 1) - 1 for k in range(N)] for i in range(dim)],
                dtype=float)
 
            corr_exact_2d = np.zeros((L, L))
            for x in range(L):
                for y in range(L):
                    r = x * L + y
                    corr_exact_2d[x, y] = np.sum(probs * all_cfg[:, 0] * all_cfg[:, r])
 
        n_cols  = 3 if has_exact else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        title_tag = ' ← critical' if abs(h - J) < 0.05 else ''
        fig.suptitle(
            f'2D Spin Correlations ⟨σ₀₀ σₓᵧ⟩  —  h/J={h/J:.1f}{title_tag}'
            f'  (L={L}×{L}, J={J})',
            fontsize=12, fontweight='bold')
 
        vmin, vmax = -1.0, 1.0
        col = 0
 
        if has_exact and corr_exact_2d is not None:
            im = axes[col].imshow(corr_exact_2d, vmin=vmin, vmax=vmax,
                                   cmap='RdBu_r', origin='upper')
            axes[col].set_title('Exact ⟨σ₀₀ σₓᵧ⟩')
            axes[col].set_xlabel('column y')
            axes[col].set_ylabel('row x')
            plt.colorbar(im, ax=axes[col])
            col += 1
 
        im = axes[col].imshow(corr_rbm_2d, vmin=vmin, vmax=vmax,
                               cmap='RdBu_r', origin='upper')
        axes[col].set_title('RBM ⟨σ₀₀ σₓᵧ⟩')
        axes[col].set_xlabel('column y')
        axes[col].set_ylabel('row x')
        plt.colorbar(im, ax=axes[col])
        col += 1
 
        ax = axes[col]
        distances = np.arange(L)
        if has_exact and corr_exact_2d is not None:
            ax.plot(distances, corr_exact_2d[0, :], 'o-',
                    color='crimson', lw=2, ms=8, label='Exact (row 0)')
        ax.errorbar(distances, corr_rbm_2d[0, :], yerr=2 * corr_err_2d[0, :],
                    fmt='s--', color='steelblue', lw=2, ms=8,
                    capsize=4, label='RBM (±2σ, row 0)')
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_xlabel('Distance along row 0')
        ax.set_ylabel('⟨σ₀₀ σ₀ᵧ⟩')
        ax.set_title('1D slice (compare to 1D plots)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
 
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    L = 3    
    J = 1.0
 
    print("=" * 60)
    print(f"2D TFI Visualizations  —  L={L}×{L}, N={L*L}")
    print("=" * 60)
 
    print("\nFigure 1: Convergence across h/J ratios")
    plot_2d_convergence_vs_h(
        L=L, J=J,
        h_values=[0.3, 0.5, 1.0, 1.5, 2.5],
        alpha=2, n_steps=300, n_samples=300, eta=0.02
    )
 
    print("\nFigure 2: Signed residuals at critical point")
    plot_2d_residuals(
        L=L, J=J, h=1.0,
        alpha=2, n_steps=400, n_samples=300, eta=0.02
    )
 
    print("\nFigure 3: Spin-spin correlations")
    plot_2d_spin_correlations(
        L=L, J=J,
        h_values=[0.5, 1.0, 2.0],
        alpha=2, n_steps=300, n_samples=500, eta=0.02
    )
