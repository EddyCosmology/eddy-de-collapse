import numpy as np
import matplotlib.pyplot as plt

# Parameters
Omega_m0 = 0.3
w0 = -0.862  # your target local w(0)

# Approximate growth index γ for constant w
gamma = 0.55 + 0.05 * (1 + w0)  # Linder approximation

# Ω_m(z) = Omega_m0 * (1+z)^3 / E(z)^2
def E_z(z, w=w0):
    return np.sqrt(Omega_m0 * (1+z)**3 + (1 - Omega_m0) * (1+z)**(3*(1+w)))

def Omega_m_z(z):
    return Omega_m0 * (1+z)**3 / E_z(z)**2

def f_z(z):
    return Omega_m_z(z)**gamma

# ΛCDM reference (w = -1, γ = 0.55)
f_lcdm = Omega_m0**0.55  # ≈ 0.5157

# Compute at z=0
f_model_z0 = f_z(0)
suppression_factor = f_model_z0 / f_lcdm
sigma8_model = 0.811 * suppression_factor

print(f"Approximate f(z=0) in model (w = {w0}): {f_model_z0:.4f}")
print(f"ΛCDM f(z=0): {f_lcdm:.4f}")
print(f"σ₈ suppression factor (model / ΛCDM): {suppression_factor:.4f}")
print(f"Estimated σ₈ in model: {sigma8_model:.3f} (vs Planck 0.811)")

# Plot f(z) for illustration
z_plot = np.logspace(-3, 3, 1000)
f_plot = f_z(z_plot)
plt.figure(figsize=(8,5))
plt.plot(z_plot, f_plot, 'b-', lw=2, label=f'Model f(z) (w = {w0})')
plt.axhline(f_lcdm, color='k', ls='--', label='ΛCDM reference')
plt.xscale('log')
plt.xlabel('1 + z (log scale)')
plt.ylabel('Growth rate f(z)')
plt.title('Approximate Linear Growth Rate f(z)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('growth_rate_approx.png', dpi=300)
plt.show()
