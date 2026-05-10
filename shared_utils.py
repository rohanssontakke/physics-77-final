#this file holds our base NN pre sampling, and some spin config functions that are used universally.

class WaveFunction(nn.Module):
    """
    Input:  σ ∈ {±1}^L   (one spin configuration)
    Output: log|Ψ(σ)|    (one scalar)
    """
    def __init__(self, L, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for p in self.parameters():
            nn.init.normal_(p, std=0.01)

    def log_psi(self, sigma):
        """sigma: (batch, L) → returns (batch,)"""
        return self.net(sigma).squeeze(-1)



def all_configs(L):
    """Returns (2^L, L) tensor of all spin configs, values ±1."""
    configs = []
    for i in range(2**L):
        spins = [2*((i >> k) & 1) - 1 for k in range(L)]
        configs.append(spins)
    return torch.tensor(configs, dtype=torch.float32)



def exact_diag(L, J, h):
    dim = 2**L
    H = np.zeros((dim, dim))
    for idx in range(dim):
        spins = [2*((idx >> k) & 1) - 1 for k in range(L)]
        for i in range(L):
            H[idx, idx] += -J * spins[i] * spins[(i+1) % L]
        for i in range(L):
            H[idx, idx ^ (1 << i)] += -h
    evals, evecs = np.linalg.eigh(H)
    return evals[0], evecs[:, 0]

