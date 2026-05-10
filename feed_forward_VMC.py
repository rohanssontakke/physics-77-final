#The code that ran our first model, pre sampling step

def compute_local_energy_1d(sigma_all, log_psi_all, L, J, h):
    #Local energy is the diagonal + off-diagonal, which is different part of hamiltonian
    n = sigma_all.shape[0]

    neighbors = sigma_all[:, [(i+1) % L for i in range(L)]]
    e_diag = -J * torch.sum(sigma_all * neighbors, dim=1)

    e_offdiag = torch.zeros(n)
    indices = torch.arange(n)
    for k in range(L):
        flipped_idx = indices ^ (1 << k)            
        log_ratio = log_psi_all[flipped_idx] - log_psi_all  
        e_offdiag += -h * torch.exp(log_ratio)

    return e_diag + e_offdiag


def vmc_step(wf, sigma_all, L, J, h, optimizer):
    optimizer.zero_grad()

    # Evaluate network on ALL configs
    log_psi = wf.log_psi(sigma_all)          # (2^L,), has grad

    log_prob = 2.0 * log_psi.detach()
    pi = torch.softmax(log_prob, dim=0)       # (2^L,), sums to 1

    # Local energy for each config
    with torch.no_grad():
        e_loc = compute_local_energy_1d(sigma_all, log_psi.detach(), L, J, h)

    # Variational energy and variance
    energy = torch.sum(pi * e_loc).item()
    variance = torch.sum(pi * (e_loc - energy)**2).item()

    e_centered = (e_loc - energy).detach()
    loss = 2.0 * torch.sum(pi.detach() * e_centered * log_psi)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(wf.parameters(), max_norm=5.0)
    optimizer.step()

    return energy, variance



def train_1d(L, J, h, hidden_dim=32, lr=0.01, n_iter=600, seed=42):
    """
    Train a WaveFunction on the 1D TFI model.
    Returns: (energies, variances, e_exact, psi_exact_vec, wf, sigma_all)
    """
    torch.manual_seed(seed)
    e_exact, psi_exact_vec = exact_diag(L, J, h)
    sigma_all = all_configs(L)
    wf = WaveFunction(L, hidden_dim)
    optimizer = torch.optim.Adam(wf.parameters(), lr=lr)

    energies, variances = [], []
    for _ in range(n_iter):
        e, v = vmc_step(wf, sigma_all, L, J, h, optimizer)
        energies.append(e)
        variances.append(v)

    return energies, variances, e_exact, psi_exact_vec, wf, sigma_all
