import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core_physics import ZVEngine

# --- 1. SCIENTIFIC PHYSICS ENGINE ---
def get_3d_geodesics(M, gamma, initial_conditions, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    def system_dynamics(t, state):
        x, y, phi, px, py = state
        q_vec = np.array([x, y])
        p_vec = np.array([-0.95, px, py, 3.0]) 
        return [dh_dp(q_vec, p_vec)[1], dh_dp(q_vec, p_vec)[2], dh_dp(q_vec, p_vec)[3], 
                -dh_dq(q_vec, p_vec)[0], -dh_dq(q_vec, p_vec)[1]]

    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, 
                    t_eval=np.linspace(0, total_time, 3000))
    return sol

# --- 2. SOLVE THE MATH ---
M = 1.0
gamma = 0.5  # Voorhees Oblate Mass
time_span = 800
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Calculating spacetime geodesics for the Director's Cut. Please wait...")
sol_zv = get_3d_geodesics(M, gamma, init_state, time_span)

x_vals, y_vals, phi_vals = sol_zv.y[0], sol_zv.y[1], sol_zv.y[2]
t_vals = sol_zv.t

# Transform to Physical 3D Space
rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
Z_orbit = M * x_vals * y_vals
X_orbit = rho * np.cos(phi_vals)
Y_orbit = rho * np.sin(phi_vals)

# --- 3. CINEMATIC SETUP ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Remove the ugly grid for a "deep space" look
ax.set_axis_off() 

# Draw the Deformed Voorhees Singularity
x_surf = 1.05
u = np.linspace(0, 2 * np.pi, 40) 
v = np.linspace(-1, 1, 40)        
U, V = np.meshgrid(u, v)
rho_surf = M * np.sqrt((x_surf**2 - 1) * (1 - V**2))
ax.plot_surface(rho_surf * np.cos(U), rho_surf * np.sin(U), M * x_surf * V, 
                color='black', alpha=1.0, edgecolor='darkred', linewidth=0.5)

# Setup Orbit Trail and Planet
trail_line, = ax.plot([], [], [], color='cyan', linewidth=1.5, alpha=0.8)
planet, = ax.plot([], [], [], marker='o', color='white', markersize=6, 
                  markeredgecolor='cyan', markeredgewidth=2)

# --- 4. EXPLAINABLE LIVE HUD (Telemetry) ---
# This text box will update live on the screen!
hud_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, color='white', 
                     fontsize=12, family='monospace', 
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='cyan'))

title_text = ax.text2D(0.05, 0.95, f"ZIPOY-VOORHEES SPACETIME ($\gamma$={gamma})\nGeodesic Orbital Simulation", 
                       transform=ax.transAxes, color='cyan', fontsize=14, weight='bold')

# Setup limits
limit = 15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# --- 5. ANIMATION & CAMERA LOGIC ---
step_size = 12 

def update(frame):
    idx = frame * step_size 
    
    # 1. Update the orbit visual
    trail_line.set_data(X_orbit[:idx], Y_orbit[:idx])
    trail_line.set_3d_properties(Z_orbit[:idx])
    planet.set_data([X_orbit[idx]], [Y_orbit[idx]])
    planet.set_3d_properties([Z_orbit[idx]])
    
    # 2. Update the Explainable HUD with scientific data
    current_t = t_vals[idx]
    current_rho = rho[idx]
    current_z = Z_orbit[idx]
    current_phi = np.degrees(phi_vals[idx]) % 360
    
    hud_text.set_text(f"TIME (τ):  {current_t:06.1f} M\n"
                      f"RADIAL(ρ): {current_rho:05.2f} M\n"
                      f"HEIGHT(z): {current_z:+05.2f} M\n"
                      f"ANGLE (φ): {current_phi:05.1f}°")
    
    # 3. Cinematic Camera Movement!
    # Slowly rotate the camera around the z-axis, and gently bob it up and down
    azimuth_angle = (frame * 0.4) % 360
    elevation_angle = 15 + 10 * np.sin(frame * 0.02)
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    
    return trail_line, planet, hud_text

total_frames = len(X_orbit) // step_size
ani = FuncAnimation(fig, update, frames=total_frames, interval=25, blit=False)

plt.show()
