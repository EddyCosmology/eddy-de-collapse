import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

# ────────────────────────────────────────────────
#  Parameters
# ────────────────────────────────────────────────
beta   = 1.6          # nonlinear k-essence coeff
kappa  = 0.005        # hyperdiffusion damping
m      = 0.8          # mass in H0 units
phi0   = 0.5          # noise normalization scale
nu     = 0.03         # running vacuum coeff
sigma_bg = 0.002      # background noise (small)
H0     = 1.0

N_start = -8.0
N_end   = 0.0
N_points = 800
N_grid  = np.linspace(N_start, N_end, N_points)

Omega_m0 = 0.3
rho_m_today = Omega_m0

n_ensemble = 50       # number of perturbation realizations per environment
sigma_pert = 0.001    # perturbation noise amplitude (small for stability)
k_mode     = 10.0     # scale (high for sub-horizon growth)

# ────────────────────────────────────────────────
#  Hubble function
# ────────────────────────────────────────────────
def Hubble(N, rho_phi, rho_m):
    Lambda0 = 1.0 - Omega_m0
    H2 = (rho_m * np.exp(-3*N) + rho_phi + Lambda0) / (1.0 - nu)
    return np.sqrt(max(H2, 1e-12))

# ────────────────────────────────────────────────
#  Background EOM (deterministic for consistent pert base)
# ────────────────────────────────────────────────
def scalar_eom(N, y):
    phi, phi_N = y
    X = 0.5 * phi_N**2
    rho_phi = X + 0.5 * (m * phi)**2
    H = Hubble(N, rho_phi, rho_m_today * np.exp(-3*N))
    phi_NN = -3 * phi_N - beta * phi * (2*X) - (m**2 * phi)
    phi_NN -= kappa * phi_N
    return [phi_N, phi_NN]

print("Running deterministic background...")
start_bg = time.time()
sol_bg = solve_ivp(
    scalar_eom, [N_start, N_end], [0.48, 0.012],
    t_eval=N_grid, method='LSODA', rtol=1e-8, atol=1e-10, max_step=0.02
)
print(f"Background done in {time.time() - start_bg:.2f} s, success: {sol_bg.success}")

if not sol_bg.success:
    print("Background failed:", sol_bg.message)
    exit()

phi_bg_interp   = interp1d(sol_bg.t, sol_bg.y[0], kind='cubic', fill_value="extrapolate")
phi_N_bg_interp = interp1d(sol_bg.t, sol_bg.y[1], kind='cubic', fill_value="extrapolate")

# ────────────────────────────────────────────────
#  Perturbation EOM
# ────────────────────────────────────────────────
def pert_eom(N, y, k_mode=10.0):
    delta_phi, delta_phi_N, delta_m, theta_m = y

    phi   = phi_bg_interp(N)
    phi_N = phi_N_bg_interp(N)
    a     = np.exp(N)
    H     = Hubble(N, 0.5*phi_N**2 + 0.5*(m*phi)**2, rho_m_today * np.exp(-3*N))

    # Linearized KG
    delta_phi_NN = -3 * delta_phi_N \
                   - (k_mode**2 / (a**2 * H**2) + m**2 / H**2) * delta_phi \
                   - 2 * beta * phi * (phi_N * delta_phi_N + delta_phi * phi_N**2 / phi)

    # Stochastic noise in perturbation
    dN = N_grid[1] - N_grid[0]
    xi_pert = np.random.normal(0, sigma_pert * np.sqrt(dN))
    delta_phi_NN += xi_pert * (delta_phi / phi)

    # Matter continuity
    delta_m_N = theta_m

    # Matter Euler
    Omega_m = rho_m_today * np.exp(-3*N) / H**2
    theta_m_N = -theta_m + (3/2) * Omega_m * delta_m  # positive for clustering

    return [delta_phi_N, delta_phi_NN, delta_m_N, theta_m_N]

# Environments
environments = {
    'filament': {'delta_m_ini': 0.1,  'color': 'blue',  'label': 'Overdense (filament)'},
    'void':     {'delta_m_ini': -0.5, 'color': 'orange', 'label': 'Underdense (void)'},
    'mean':     {'delta_m_ini': 0.0,  'color': 'green', 'label': 'Mean cosmology'}
}

# ────────────────────────────────────────────────
# Run ensemble perturbations
# ────────────────────────────────────────────────
print(f"Running {n_ensemble} perturbation realizations per environment...")
start_pert = time.time()

delta_m_ensembles = {name: [] for name in environments}

for i in range(n_ensemble):
    if (i+1) % 5 == 0:
        print(f"  Ensemble {i+1}/{n_ensemble}")
    
    for name, env in environments.items():
        y0_pert = [1e-5, 1e-5, env['delta_m_ini'], 0.0]

        sol = solve_ivp(
            lambda N, y: pert_eom(N, y, k_mode=k_mode),
            [N_start, N_end], y0_pert,
            t_eval=N_grid, method='LSODA', rtol=1e-8, atol=1e-10
        )

        if sol.success:
            delta_m_ensembles[name].append(sol.y[2])
        else:
            print(f"    Failed realization {i+1} for {name}")

print(f"Ensemble perturbations done in {time.time() - start_pert:.2f} s")

# ────────────────────────────────────────────────
# Compute mean and std for each environment
# ────────────────────────────────────────────────
pert_results = {}
for name in environments:
    if delta_m_ensembles[name]:
        dm_array = np.array(delta_m_ensembles[name])
        dm_mean = np.mean(dm_array, axis=0)
        dm_std  = np.std(dm_array, axis=0)
        pert_results[name] = {
            'z': np.exp(-N_grid) - 1,
            'delta_m_mean': dm_mean,
            'delta_m_std': dm_std
        }
    else:
        print(f"No successful realizations for {name}")

# ────────────────────────────────────────────────
# Plot ensemble mean ±1σ (with scaling if scatter tiny)
# ────────────────────────────────────────────────
if pert_results:
    scale_factor = 5000.0  # adjust to make band visible if std very small
    
    plt.figure(figsize=(9,6))
    for name, data in pert_results.items():
        env = environments[name]
        plt.plot(data['z'], data['delta_m_mean'], color=env['color'], lw=2.5, label=env['label'])
        plt.fill_between(data['z'],
                         data['delta_m_mean'] - scale_factor * data['delta_m_std'],
                         data['delta_m_mean'] + scale_factor * data['delta_m_std'],
                         color=env['color'], alpha=0.25, label=f'{env["label"]} ±{scale_factor}σ')

    plt.xscale('log')
    plt.xlim(1e-3, 1e3)
    plt.ylim(-0.8, 0.4)
    plt.xlabel('1 + z (log scale)')
    plt.ylabel('Matter density contrast δ_m')
    plt.title(f'Ensemble-averaged Linear δ_m Evolution\n({n_ensemble} realizations, k = {k_mode} H₀)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('delta_m_ensemble.png', dpi=350)
    plt.show()

# ────────────────────────────────────────────────
# Proxy calculation (using mean at z≈0)
# ────────────────────────────────────────────────
if 'filament' in pert_results and 'void' in pert_results:
    z_target = 0.0
    idx_low = np.argmin(np.abs(pert_results['filament']['z'] - z_target))
    
    delta_fil = pert_results['filament']['delta_m_mean'][idx_low]
    delta_void = pert_results['void']['delta_m_mean'][idx_low]
    
    delta_diff = delta_fil - delta_void
    
    print(f"Ensemble mean δ_m (filament) at z ≈ 0: {delta_fil:.4f}")
    print(f"Ensemble mean δ_m (void) at z ≈ 0: {delta_void:.4f}")
    print(f"Δδ_m = δ_fil - δ_void: {delta_diff:.4f}")
    print(f"Rough Δz/z proxy (ΔH/H ≈ Δδ_m / 3): {abs(delta_diff)/3:.4f}")
else:
    print("Could not compute proxy — missing filament or void ensemble.")
