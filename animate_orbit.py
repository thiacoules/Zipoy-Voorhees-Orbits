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

    # We use more points for a smooth animation
    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, 
                    t_eval=np.linspace(0, total_time, 2500))
    return sol

# --- 2. RUN MATH SETUP ---
M = 1.0
gamma = 0.5  # Oblate central mass
time_span = 800
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Calculating spacetime data for the animation. Please wait...")
sol_zv = get_3d_geodesics(M, gamma, init_state, time_span)

x_vals, y_vals, phi_vals = sol_zv.y[0], sol_zv.y[1], sol_zv.y[2]

# Transform to 3D Cartesian Space
rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
X_orbit = rho * np.cos(phi_vals)
Y_orbit = rho * np.sin(phi_vals)
Z_orbit = M * x_vals * y_vals

# --- 3. ANIMATION SETUP ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the Central Mass (Static)
x_surf = 1.05
u = np.linspace(0, 2 * np.pi, 40) 
v = np.linspace(-1, 1, 40)        
U, V = np.meshgrid(u, v)
rho_surf = M * np.sqrt((x_surf**2 - 1) * (1 - V**2))
X_surf = rho_surf * np.cos(U)
Y_surf = rho_surf * np.sin(U)
Z_surf = M * x_surf * V
ax.plot_surface(X_surf, Y_surf, Z_surf, color='black', alpha=0.9, edgecolor='red', linewidth=0.3)

# Setup empty objects for the moving planet and its trail
trail_line, = ax.plot([], [], [], color='cyan', linewidth=1.5, alpha=0.6, label='Orbital Trail')
planet, = ax.plot([], [], [], marker='o', color='white', markersize=8, label='Particle')

limit = 15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_title(f"Dynamic Orbital Simulation in ZV Spacetime ($\gamma$={gamma})", color='white', fontsize=14)
ax.set_xlabel("X (M)")
ax.set_ylabel("Y (M)")
ax.set_zlabel("Z (M)")
ax.grid(color='gray', linestyle=':', linewidth=0.5)

# Fix panes for dark background
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.legend(loc='upper right')

# --- 4. ANIMATION LOGIC ---
# We step by 10 frames at a time so the animation plays at a good speed
step_size = 10 

def update(frame):
    # Calculate the current index based on the frame
    idx = frame * step_size 
    
    # Update the trail (draw from beginning up to current frame)
    trail_line.set_data(X_orbit[:idx], Y_orbit[:idx])
    trail_line.set_3d_properties(Z_orbit[:idx])
    
    # Update the planet's exact current position
    planet.set_data([X_orbit[idx]], [Y_orbit[idx]])
    planet.set_3d_properties([Z_orbit[idx]])
    
    return trail_line, planet

# Calculate total frames
total_frames = len(X_orbit) // step_size

# Create the animation loop
ani = FuncAnimation(fig, update, frames=total_frames, interval=20, blit=False)

plt.show()
