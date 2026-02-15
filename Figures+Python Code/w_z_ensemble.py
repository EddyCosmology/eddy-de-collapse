# Simple scalar field + running vacuum evolution in cosmology
# Run this in a Python environment with numpy, scipy, matplotlib installed
# (e.g., Anaconda, Google Colab, or plain Python + pip install numpy scipy matplotlib)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------
#  Parameters from your paper (feel free to change)
# -------------------------------
beta   = 1.6          # nonlinear k-essence coeff
kappa  = 0.005        # hyperdiffusion (we approximate mild damping here)
m      = 0.8          # mass in units of H0 (m ≈ 0.8 H0)
phi0   = 0.5          # normalization scale for noise
nu     = 0.03         # running vacuum coeff (3 nu H^2 term)
sigma  = 0.002        # noise amplitude -- starting SMALL to avoid blow-up!
H0     = 1.0          # set H0 = 1 in natural units (rescale later if needed)

# Initial conditions at high redshift (z_init ≈ 3000 → N_init small)
N_start = -8.0        # N = ln(a), N=0 today, N negative in past
N_end   = 0.0
N_points = 600
N_grid  = np.linspace(N_start, N_end, N_points)

# Today: Omega_m ≈ 0.3, so H(N=0) = H0 = 1
Omega_m0 = 0.3
rho_m_today = Omega_m0                     # in units H0^2

# -------------------------------
# Hubble as function of N (approximate, iterative if needed)
# -------------------------------
def Hubble(N, rho_phi, rho_m):
    # Friedmann: H^2 = rho_total + Lambda(H)/3   (in units 8piG/3 = 1)
    # Running: Lambda(H) = Lambda0 + 3 nu H^2
    # Effective: H^2 ≈ (rho_m + rho_phi + Lambda0) / (1 - nu)
    Lambda0 = 1.0 - Omega_m0   # approx, tuned so H(N=0)=1 when rho_phi≈0.7
    H2 = (rho_m * np.exp(-3*N) + rho_phi + Lambda0) / (1.0 - nu)
    return np.sqrt(max(H2, 1e-12))  # prevent negative/zero

# -------------------------------
# Scalar field background equations in e-folds (d/dN)
# y = [phi, phi_N] where phi_N = d phi / dN = dot(phi)/H
# -------------------------------
def scalar_eom(N, y, beta, kappa, sigma, phi0):
    phi, phi_N = y
    X = 0.5 * phi_N**2                  # kinetic term


    # Approximate H(N) -- we update it outside in loop for consistency
    # For first approx we use a placeholder; in full code we'd couple better
    H = Hubble(N, 0.5*phi_N**2 + 0.5*(m*phi)**2, rho_m_today)  # rough
    
    # Multiplicative noise (discrete white noise approximation)
    # In one step ΔN, variance sigma^2 ΔN
    dN = N_grid[1] - N_grid[0] if len(N_grid)>1 else 0.01
    xi = np.random.normal(0, sigma * np.sqrt(dN))   # scaled Wiener increment
    noise = xi * (phi**2 / phi0**2)
    
    # Klein-Gordon in e-folds
    phi_NN = -3 * phi_N - beta * phi * (2*X) - (m**2 * phi) + noise
    
    # Very crude hyperdiffusion damping (higher deriv → extra friction on velocity)
    phi_NN -= kappa * (phi_N**3)   # phenomenological damping term
    
    return [phi_N, phi_NN]

# -------------------------------
# Run one realization
# -------------------------------
def run_one_realization():
    y0 = [0.48, 0.012]   # from your paper
    
    sol = solve_ivp(
        fun=lambda N, y: scalar_eom(N, y, beta, kappa, sigma, phi0),
        t_span=[N_start, N_end],
        y0=y0,
        t_eval=N_grid,
        method='LSODA',       # good for stiff + noise
        rtol=1e-8, atol=1e-10,
        max_step=0.02         # prevent noise from making huge jumps
    )
    
    if not sol.success:
        print("Integration failed:", sol.message)
        return None, None, None
    
    phi     = sol.y[0]
    phi_N   = sol.y[1]
    X       = 0.5 * phi_N**2
    V       = 0.5 * (m * phi)**2 
    rho_phi = X + V
    p_phi   = X - V - beta * phi * (2 * X * phi_N**2)  # k-essence pressure term approx
    
    # Effective w = p_DE / rho_DE  (including running part indirectly via H)
    w = np.where(rho_phi > 1e-12, p_phi / rho_phi, -1.0)
    
    z = np.exp(-N_grid) - 1
    return z, w, rho_phi

# -------------------------------
# Run ensemble and plot mean ±1σ
# -------------------------------
n_realizations = 50   # increase to 50+ for smoother bands
w_all = []

print("Running ensemble... (may take a minute)")
for i in range(n_realizations):
    z, w, _ = run_one_realization()
    if w is not None:
        w_all.append(w)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{n_realizations} done")

if len(w_all) > 0:
    w_all = np.array(w_all)
    w_mean = np.mean(w_all, axis=0)
    w_std  = np.std(w_all, axis=0)

    print("Ensemble statistics at z=0:")
    idx0 = np.argmin(np.abs(z-0))
    print(f" w(z=0) mean = {w_mean[idx0]:.6f}")
    print(f" w(z=0) std = {w_std[idx0]:.6f}")
    print(f" Min w_std across all z = {np.nanmin(w_std):.6f}")
    print(f" Max w_std across all z = {np.nanmax(w_std):.6f}")
    print(f" Median w_std = {np.nanmedian(w_std):.6f}")


    plt.figure(figsize=(8,5.5))
    plt.plot(z, w_mean, 'b-', lw=2.5, label='Mean w(z)')
    plt.fill_between(z, w_mean - w_std, w_mean + w_std, color='lightblue', alpha=0.45,edgecolor='blue', linewidth=0.8, label='±1σ ensemble')
    plt.axhline(-1.0, color='k', ls='--', lw=1, label='Λ constant')
    plt.axhline(-0.862, color='r', ls=':', lw=1.5, label='Desi DR2-inspired target w(0) ≈ -0.862')
    
    plt.xscale('log')
    plt.xlim(1e-3, 1e3)
    plt.ylim(-1.05, -0.70)
    plt.xlabel('1 + z  (log scale)')
    plt.ylabel('w(z)')
    plt.title('Scalar field w(z) — ensemble average')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print some values
    idx_z0 = np.argmin(np.abs(z - 0))
    print(f"w(z=0) mean = {w_mean[idx_z0]:.3f} ± {w_std[idx_z0]:.3f}")
else:
    print("All integrations failed — try smaller sigma or tighter tolerances.")
