import numpy as np
import matplotlib.pyplot as plt

def metric_equator(x, M, gamma):
    """ Returns g_tt and g_phi at the equator (y=0) for ZV spacetime """
    # Avoid division by zero exactly at x=1
    x = np.where(x == 1.0, 1.0001, x)
    
    g_tt = ((x - 1) / (x + 1))**gamma
    # At y=0:
    sigma_sq = x**2 - 1
    g_phi = (M**2) * sigma_sq / g_tt
    
    return g_tt, g_phi

def effective_potential(x, L, M, gamma):
    """ V_eff for a particle with angular momentum L """
    g_tt, g_phi = metric_equator(x, M, gamma)
    # The relativistic effective potential squared
    V_eff_sq = g_tt * (1 + (L**2 / g_phi))
    return np.sqrt(V_eff_sq)

# --- SETUP ---
M = 1.0
x_vals = np.linspace(1.5, 15, 500)

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')

# 1. Plot Schwarzschild (gamma = 1.0)
# The ISCO for Schwarzschild is at x=6, which corresponds to L=sqrt(12) ~ 3.46
L_isco_sph = np.sqrt(12)
V_sph = effective_potential(x_vals, L_isco_sph, M, gamma=1.0)
plt.plot(x_vals, V_sph, color='blue', linewidth=2, label=f'Schwarzschild ($\gamma=1.0$), L={L_isco_sph:.2f}')

# Mark the Schwarzschild ISCO
plt.scatter(6.0, effective_potential(6.0, L_isco_sph, M, 1.0), color='cyan', s=100, zorder=5, label='ISCO (x = 6.0M)')

# 2. Plot Zipoy-Voorhees Oblate (gamma = 0.5)
# We test a similar angular momentum to see how the potential well changes
V_zv = effective_potential(x_vals, L_isco_sph, M, gamma=0.5)
plt.plot(x_vals, V_zv, color='red', linewidth=2, linestyle='--', label=f'Zipoy-Voorhees ($\gamma=0.5$), L={L_isco_sph:.2f}')

plt.title("Effective Potential ($V_{eff}$) and the ISCO", color='white', fontsize=14)
plt.xlabel("Radial Distance (x)", fontsize=12)
plt.ylabel("Effective Potential Energy", fontsize=12)
plt.axhline(1.0, color='gray', linestyle=':', linewidth=1) # Rest mass energy

# 

ax = plt.gca()
ax.tick_params(colors='white')
ax.grid(color='gray', linestyle=':', linewidth=0.3)
plt.legend(loc='lower right')

plt.show()
