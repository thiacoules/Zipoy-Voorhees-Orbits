import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from core_physics import ZVEngine

def run_precession_orbit(M, gamma, init_state, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    def system_dynamics(t, state):
        x, y, phi, px, py = state
        q_vec = np.array([x, y])
        p_vec = np.array([-0.95, px, py, 3.0]) 
        
        dHdp = dh_dp(q_vec, p_vec)
        dHdq = dh_dq(q_vec, p_vec)
        
        return [dHdp[1], dHdp[2], dHdp[3], -dHdq[0], -dHdq[1]]

    # High precision integration to find exact angles
    sol = solve_ivp(system_dynamics, [0, total_time], init_state, 
                    t_eval=np.linspace(0, total_time, 5000), rtol=1e-8, atol=1e-8)
    return sol

def calculate_precession(sol, name):
    x_vals = sol.y[0]
    phi_vals = sol.y[2]
    
    # Find the "Periapsis" (closest approach points). 
    # We invert x (-x) to find the minima using find_peaks
    periapsis_indices, _ = find_peaks(-x_vals)
    
    if len(periapsis_indices) < 2:
        return 0, []

    # Get the angles (phi) at the first two closest approaches
    phi_1 = phi_vals[periapsis_indices[0]]
    phi_2 = phi_vals[periapsis_indices[1]]
    
    # Calculate the shift (Total angle traveled between two periapsis minus 360 degrees)
    delta_phi_rad = (phi_2 - phi_1) - (2 * np.pi)
    delta_phi_deg = np.degrees(delta_phi_rad)
    
    print(f"\n--- {name} ---")
    print(f"Angle at First Periapsis:  {np.degrees(phi_1):.2f}°")
    print(f"Angle at Second Periapsis: {np.degrees(phi_2):.2f}°")
    print(f"Total Orbital Precession Shift per revolution: {delta_phi_deg:+.4f} degrees")
    
    return delta_phi_deg, periapsis_indices

# --- SIMULATION SETUP ---
M = 1.0
time_span = 400
# Initial state: [x, y, phi, px, py] -> Starting from an apoapsis (x=10)
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Running High-Precision Orbital Precession Study...")

# 1. Run Schwarzschild (Spherical)
sol_sph = run_precession_orbit(M, 1.0, init_state, time_span)
shift_sph, idx_sph = calculate_precession(sol_sph, "Schwarzschild (γ=1.0)")

# 2. Run Zipoy-Voorhees (Oblate)
sol_zv = run_precession_orbit(M, 0.5, init_state, time_span)
shift_zv, idx_zv = calculate_precession(sol_zv, "Zipoy-Voorhees (γ=0.5)")

# --- PLOTTING IN POLAR COORDINATES (Top-Down View) ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(8, 8))
# Using a polar plot shows the true rosette orbital shape
ax = fig.add_subplot(111, projection='polar')

# Plot Schwarzschild
ax.plot(sol_sph.y[2], sol_sph.y[0], color='blue', linewidth=1.5, label=f'γ=1.0 (Shift: {shift_sph:+.1f}°)')
# Mark its periapsis points
ax.scatter(sol_sph.y[2][idx_sph], sol_sph.y[0][idx_sph], color='cyan', zorder=5)

# Plot Zipoy-Voorhees
ax.plot(sol_zv.y[2], sol_zv.y[0], color='red', linewidth=1.5, linestyle='--', label=f'γ=0.5 (Shift: {shift_zv:+.1f}°)')
# Mark its periapsis points
ax.scatter(sol_zv.y[2][idx_zv], sol_zv.y[0][idx_zv], color='yellow', zorder=5)

ax.set_title("Top-Down Orbital Precession (Rosette Orbits)", color='white', fontsize=14, pad=20)
ax.set_rmax(11)
ax.set_rticks([2, 4, 6, 8, 10])  # Radial ticks
ax.grid(color='gray', linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()
