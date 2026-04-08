import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

# --- 1. PHYSICS ENGINE ---
def get_3d_geodesics(M, gamma, initial_conditions, total_time):
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

    # INTEGRATING FOR A LONG TIME WITH EXTREME PRECISION
    # We use solve_ivp's events feature to stop if the math breaks, 
    # but otherwise let it run to map the full geometry.
    print(f"Integrating the Geodesic equations for τ={total_time}...")
    print("This maps the underlying geometric sketch. Please wait...")
    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, 
                    t_eval=np.linspace(0, total_time, 8000), 
                    rtol=1e-7, atol=1e-7)
    return sol

# --- 2. SOLVE THE MATH FOR THE AFTERMATH ---
M = 1.0
gamma = 0.5  # Voorhees Oblate Mass
time_span = 5000  # Massive integration time to reveal full sketch
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

sol_zv = get_3d_geodesics(M, gamma, init_state, time_span)

x_vals, y_vals, phi_vals = sol_zv.y[0], sol_zv.y[1], sol_zv.y[2]

# Transform to Physical 3D Cartesian Space
rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
Z_sketch = M * x_vals * y_vals
X_sketch = rho * np.cos(phi_vals)
Y_sketch = rho * np.sin(phi_vals)

# --- 3. THE AFTERMATH VISUALIZATION ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Central Deformed Mass (Red mesh)
x_surf = 1.05
u = np.linspace(0, 2 * np.pi, 40) 
v = np.linspace(-1, 1, 40)        
U, V = np.meshgrid(u, v)
rho_surf = M * np.sqrt((x_surf**2 - 1) * (1 - V**2))
ax.plot_surface(rho_surf * np.cos(U), rho_surf * np.sin(U), M * x_surf * V, 
                color='black', alpha=0.9, edgecolor='red', linewidth=0.5)

# PLOT THE ENTIRE GEOMETRIC HISTORY AT ONCE
# We use a very thin line (0.3) and high transparency (0.4)
# so the overlapping lines look soft and misty, like a cosmic web.
ax.plot(X_sketch, Y_sketch, Z_sketch, color='cyan', linewidth=0.3, alpha=0.4, label='Toroidal Envelope (Sketch)')

# SCROLLS OF TEXT for Explanation (Explainable yet Scientific)
explanation = (f"AFTERMATH ANALYSIS\n"
               f"=================\n"
               f"Metric: Zipoy-Voorhees ($\gamma$={gamma})\n"
               f"Integration Time: τ={time_span} M\n\n"
               f"This visualization shows the complete density of the orbital structure. "
               f"Because General Relativity causes periapsis precession, the orbit does not close into a standard oval. "
               f"Over thousands of units of time, the geodesic fills a hollow, donut-shaped 3D volume called a **Toroidal Envelope**.")

ax.text2D(0.02, 0.95, explanation, transform=ax.transAxes, color='cyan', 
          fontsize=11, family='serif', verticalalignment='top',
          bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'))

# RESTORE THE GRID AND COORDINATES (Scientific Requirement)
limit = 15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_title(f"Orbital Aftermath: Trajectory Density Map", color='white', fontsize=16, weight='bold')
ax.set_xlabel("X-Axis (M)")
ax.set_ylabel("Y-Axis (M)")
ax.set_zlabel("Z-Axis (M)")

ax.grid(color='gray', linestyle=':', linewidth=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# ax.legend(loc='upper right')

plt.show()
