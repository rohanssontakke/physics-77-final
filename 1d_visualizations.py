def plot_convergence_vs_h(L=6, J=1.0, h_values=None, hidden_dim=32, lr=0.01, n_iter=500):

    #Train a new model for every h/J ratio, overlay energy convergence.
    if h_values is None:
        h_values = [0.0, 0.3, 0.5, 1.0, 1.5, 2.5] #list of of h-values, with 1.0 corresponding to critical point.

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(h_values)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'1D TFI Convergence vs h/J  (L={L}, J={J})', fontsize=13, fontweight='bold')

    final_errors = []

    for idx, h in enumerate(h_values): #iterating through values, creating each individual convergence line.
        energies, variances, e_exact, _, _, _ = train_1d(L, J, h, hidden_dim, lr, n_iter)
        label = f'h/J={h/J:.1f}'
        color = colors[idx]

        w = 20
        avg = np.convolve(energies, np.ones(w)/w, mode='valid')

        axes[0].plot(energies, alpha=0.15, color=color, lw=0.7)
        axes[0].plot(range(w-1, len(energies)), avg, color=color, lw=2, label=label)
        axes[0].axhline(e_exact, color=color, ls='--', lw=0.8, alpha=0.5)

        axes[1].semilogy(variances, alpha=0.15, color=color, lw=0.7)
        avg_v = np.convolve(variances, np.ones(w)/w, mode='valid')
        axes[1].semilogy(range(w-1, len(variances)), avg_v, color=color, lw=2, label=label)

        final_errors.append(abs(np.mean(energies[-20:]) - e_exact) / abs(e_exact) * 100)

    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('⟨H⟩  (total energy)', fontsize=11)
    axes[0].set_title('Energy Convergence\n(dashed = exact ground state per h/J)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('Variance  var(E_loc)  [log scale]', fontsize=11)
    axes[1].set_title('Variance Decay\n(→0 means true eigenstate)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'h/J':>8}  {'Final rel. error %':>20}")
    print("-" * 32)
    for h, err in zip(h_values, final_errors):
        marker = " ← critical point" if abs(h - J) < 0.05 else ""
        print(f"{h/J:>8.2f}  {err:>20.4f}{marker}")


def plot_residuals(L=6, J=1.0, h=1.0, hidden_dim=32, lr=0.01, n_iter=600):
    
    #plotting residuals between E(t) - E_exact, log-transformed
    energies, variances, e_exact, _, _, _ = train_1d(L, J, h, hidden_dim, lr, n_iter)

    residuals = np.array(energies) - e_exact  

    w = 20
    #residual function
    avg_res = np.convolve(residuals, np.ones(w)/w, mode='valid')
    avg_var = np.convolve(variances, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Signed Residuals  —  1D TFI  (L={L}, J={J}, h={h})', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.semilogy(residuals, alpha=0.2, color='steelblue', lw=0.7, label='raw')
    ax.semilogy(range(w-1, len(residuals)), avg_res, color='navy', lw=2, label='smoothed')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E(t) − E_exact  [log scale]')
    ax.set_title('Energy Residual\n(variational bound: always ≥ 0)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    #fitting multiple residuals to one plot
    half = len(avg_res) // 2
    x_fit = np.arange(half, len(avg_res))
    log_res = np.log(avg_res[half:] + 1e-12)
    if np.all(np.isfinite(log_res)):
        coeffs = np.polyfit(x_fit, log_res, 1)
        decay_per_100 = (1 - np.exp(coeffs[0] * 100)) * 100
        ax.text(0.98, 0.95, f'late-stage decay\n≈{decay_per_100:.1f}% per 100 iters',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax = axes[1]
    sc = ax.scatter(variances, residuals, c=np.arange(len(variances)),
                    cmap='viridis', s=6, alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Variance  var(E_loc)  [log]')
    ax.set_ylabel('E − E_exact  [log]')
    ax.set_title('Residual vs Variance\n(color = training iteration)')
    plt.colorbar(sc, ax=ax, label='iteration')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_entanglement_spectrum(L=6, J=1.0, h_values=None):
    #Entanglement spectrum is computed by splitting the matrix into a left and right half, and then taking the SVD. 
    """
    Physical interpretation:
    - Few large singular values = simple, low-entanglement state (easy to learn)
    - Many significant singular values = highly entangled (hard to learn)
    - Near h=J the spectrum is broadest — this is WHY the network struggles there
    """
    if h_values is None:
        h_values = [0.0, 0.2, 0.5, 1.0, 1.5, 3.0]

    half = L // 2
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(h_values)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Entanglement Spectrum  —  1D TFI  (L={L}, J={J})', fontsize=13, fontweight='bold')

    entanglement_entropies = []
    #svd taken here
    for idx, h in enumerate(h_values):
        _, psi_vec = exact_diag(L, J, h)
        psi_matrix = psi_vec.reshape(2**half, 2**half)
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
        sv_norm = singular_values**2
        sv_norm /= sv_norm.sum()

        label = f'h/J={h/J:.1f}'
        color = colors[idx]
        marker = '★' if abs(h - J) < 0.05 else ''

        x = np.arange(len(singular_values))
        axes[0].bar(x + idx * 0.15, singular_values, width=0.12,
                    color=color, alpha=0.8, label=f'{label}{marker}')

        ent = -np.sum(sv_norm * np.log(sv_norm + 1e-15))
        entanglement_entropies.append(ent)

    axes[0].set_xlabel('Singular value index')
    axes[0].set_ylabel('Singular value magnitude')
    axes[0].set_title('Entanglement Spectrum by h/J\n(broader = more entangled = harder to learn)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(h_values, entanglement_entropies, 'o-', color='purple', lw=2, ms=9)
    axes[1].axvline(J, color='crimson', ls='--', lw=1.5, label=f'Critical point h=J={J}')
    axes[1].set_xlabel('h (transverse field)')
    axes[1].set_ylabel('Entanglement Entropy S')
    axes[1].set_title('Entanglement Entropy vs h/J\n(peaks at quantum phase transition)')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    peak_idx = np.argmax(entanglement_entropies)
    axes[1].annotate(
        f'  max S={entanglement_entropies[peak_idx]:.3f}\n  at h/J={h_values[peak_idx]/J:.1f}',
        xy=(h_values[peak_idx], entanglement_entropies[peak_idx]),
        xytext=(h_values[peak_idx] + 0.3, entanglement_entropies[peak_idx] - 0.05),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=9
    )

    plt.tight_layout()
    plt.show()


def plot_spin_correlations(L=6, J=1.0, h_values=None, hidden_dim=32, lr=0.01, n_iter=600):
    """
    Compute ⟨σ_0 σ_r⟩ for r = 0..L-1 using both the exact and learned wavefunctions.
    Comparing the exact and learned to see how well the model's fit understands relationships, similar to entanglement spectrum.
    """
    if h_values is None:
        h_values = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(1, len(h_values), figsize=(5 * len(h_values), 5))
    if len(h_values) == 1:
        axes = [axes]
    fig.suptitle(f'Spin-Spin Correlations ⟨σ₀ σᵣ⟩  —  1D TFI  (L={L}, J={J})', fontsize=13, fontweight='bold')

    for ax, h in zip(axes, h_values):
        _, variances, e_exact, psi_exact_vec, wf, sigma_all = train_1d(
            L, J, h, hidden_dim, lr, n_iter)

        psi_exact = psi_exact_vec / np.linalg.norm(psi_exact_vec)
        all_cfg = sigma_all.numpy()  
        probs_exact = psi_exact**2   
      
        with torch.no_grad():
            lp = wf.log_psi(sigma_all).numpy()
        psi_learned = np.exp(lp)
        psi_learned /= np.linalg.norm(psi_learned)
        probs_learned = psi_learned**2

        distances = np.arange(L)
        corr_exact = np.zeros(L)
        corr_learned = np.zeros(L)

        for r in range(L):
            spin_product = all_cfg[:, 0] * all_cfg[:, r]
            corr_exact[r] = np.sum(probs_exact * spin_product)
            corr_learned[r] = np.sum(probs_learned * spin_product)

        ax.plot(distances, corr_exact, 'o-', color='crimson', lw=2, ms=8, label='Exact')
        ax.plot(distances, corr_learned, 's--', color='steelblue', lw=2, ms=8, label='NQS learned')
        ax.fill_between(distances,
                         corr_exact - np.abs(corr_exact - corr_learned),
                         corr_exact + np.abs(corr_exact - corr_learned),
                         alpha=0.15, color='gray', label='error band')
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_xlabel('Distance r')
        ax.set_ylabel('⟨σ₀ σᵣ⟩')
        ax.set_title(f'h/J = {h/J:.1f}' + (' ← critical' if abs(h-J) < 0.05 else ''))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

        max_corr = np.max(np.abs(corr_exact[1:]))
        threshold = max_corr / np.e
        corr_length = next((r for r in range(1, L) if abs(corr_exact[r]) < threshold), L)
        ax.axvline(corr_length, color='purple', ls=':', lw=1.5,
                   label=f'ξ ≈ {corr_length}')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
