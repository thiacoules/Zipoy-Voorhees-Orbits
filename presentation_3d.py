import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

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

    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, 
                    t_eval=np.linspace(0, total_time, 2000))
    return sol

# --- 1. SIMULATION SETUP ---
M = 1.0
gamma = 0.5 # Oblate (squished) mass
time_span = 800
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Calculating 3D trajectories and Central Mass shape...")
sol_zv = get_3d_geodesics(M, gamma, init_state, time_span)

x_vals, y_vals, phi_vals = sol_zv.y[0], sol_zv.y[1], sol_zv.y[2]

# Transform Orbit to 3D Cartesian
rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
X_orbit = rho * np.cos(phi_vals)
Y_orbit = rho * np.sin(phi_vals)
Z_orbit = M * x_vals * y_vals

# --- 2. GENERATE THE "BLACK HOLE" SURFACE ---
# We map a surface very close to the singularity (x = 1.05) to represent the central body
x_surf = 1.05
u = np.linspace(0, 2 * np.pi, 50) # phi angle
v = np.linspace(-1, 1, 50)        # y coordinate
U, V = np.meshgrid(u, v)

rho_surf = M * np.sqrt((x_surf**2 - 1) * (1 - V**2))
Z_surf = M * x_surf * V
X_surf = rho_surf * np.cos(U)
Y_surf = rho_surf * np.sin(U)

# --- 3. DRAW THE PLOT ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Orbit
ax.plot(X_orbit, Y_orbit, Z_orbit, color='cyan', linewidth=1.5, label='Geodesic Orbit')

# Plot the Central Deformed Mass (The "Black Hole")
ax.plot_surface(X_surf, Y_surf, Z_surf, color='black', alpha=0.9, edgecolor='red', linewidth=0.5)

# Visual settings
limit = 15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

ax.set_title(f"3D Orbital Study of Zipoy-Voorhees Spacetime ($\gamma$={gamma})", fontsize=14, color='white')
ax.set_xlabel("X (M)")
ax.set_ylabel("Y (M)")
ax.set_zlabel("Z (M)")

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(color='gray', linestyle=':', linewidth=0.5)

# Fake legend entry for the surface
import matplotlib.lines as mlines
red_patch = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Central Mass Surface')
cyan_line = mlines.Line2D([], [], color='cyan', label='Orbital Trajectory')
ax.legend(handles=[cyan_line, red_patch])

plt.show()
