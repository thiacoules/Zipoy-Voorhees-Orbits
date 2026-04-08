import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, 
                    t_eval=np.linspace(0, total_time, 3000))
    return sol

# --- 2. SOLVE THE MATH ---
M = 1.0
gamma = 0.5  # Voorhees Oblate Mass
time_span = 800
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Calculating spacetime geodesics. Please wait...")
sol_zv = get_3d_geodesics(M, gamma, init_state, time_span)

x_vals, y_vals, phi_vals = sol_zv.y[0], sol_zv.y[1], sol_zv.y[2]
t_vals = sol_zv.t

# Transform to Physical 3D Space
rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
Z_orbit = M * x_vals * y_vals
X_orbit = rho * np.cos(phi_vals)
Y_orbit = rho * np.sin(phi_vals)

# --- 3. SCIENTIFIC VISUAL SETUP (WITH GRID) ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the Central Mass (Clear surface with grid edges)
x_surf = 1.05
u = np.linspace(0, 2 * np.pi, 40) 
v = np.linspace(-1, 1, 40)        
U, V = np.meshgrid(u, v)
rho_surf = M * np.sqrt((x_surf**2 - 1) * (1 - V**2))
X_surf = rho_surf * np.cos(U)
Y_surf = rho_surf * np.sin(U)
Z_surf = M * x_surf * V
ax.plot_surface(X_surf, Y_surf, Z_surf, color='black', alpha=0.9, edgecolor='red', linewidth=0.5)

# Setup Orbit Trail and Planet
trail_line, = ax.plot([], [], [], color='cyan', linewidth=1.5, alpha=0.8, label='Geodesic Trail')
planet, = ax.plot([], [], [], marker='o', color='white', markersize=8, label='Planet / Particle')

# HUD Text (Live Telemetry)
hud_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color='cyan', 
                     fontsize=11, family='monospace', verticalalignment='top',
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'))

# RESTORE THE AXES AND GRID
limit = 15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_title(f"Dynamic Orbital Simulation ($\gamma$={gamma})", color='white', fontsize=14)
ax.set_xlabel("X-Axis (M)")
ax.set_ylabel("Y-Axis (M)")
ax.set_zlabel("Z-Axis (M)")

ax.grid(color='gray', linestyle=':', linewidth=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.legend(loc='upper right')

# --- 4. ANIMATION LOGIC ---
step_size = 12 

def update(frame):
    idx = frame * step_size 
    
    # Update visual trail and planet
    trail_line.set_data(X_orbit[:idx], Y_orbit[:idx])
    trail_line.set_3d_properties(Z_orbit[:idx])
    planet.set_data([X_orbit[idx]], [Y_orbit[idx]])
    planet.set_3d_properties([Z_orbit[idx]])
    
    # Update Telemetry Math
    hud_text.set_text(f"Telemetry Data:\n"
                      f"TIME: {t_vals[idx]:06.1f} M\n"
                      f"RHO:  {rho[idx]:05.2f} M\n"
                      f"Z:    {Z_orbit[idx]:+05.2f} M")
    
    return trail_line, planet, hud_text

total_frames = len(X_orbit) // step_size
ani = FuncAnimation(fig, update, frames=total_frames, interval=25, blit=False)

plt.show()
