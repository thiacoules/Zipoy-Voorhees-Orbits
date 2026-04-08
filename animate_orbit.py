import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
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
                    t_eval=np.linspace(0, total_time, 3500))
    return sol

# --- 2. SOLVE THE MATH ---
M = 1.0
gamma = 0.5  
time_span = 1000  # Slightly longer to make a beautiful, dense sketch
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Calculating spacetime geodesics. Please wait...")
sol_zv = get_3d_geodesics(M, gamma, init_state, time_span)

x_vals, y_vals, phi_vals = sol_zv.y[0], sol_zv.y[1], sol_zv.y[2]
t_vals = sol_zv.t

rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
Z_orbit = M * x_vals * y_vals
X_orbit = rho * np.cos(phi_vals)
Y_orbit = rho * np.sin(phi_vals)

# --- 3. VISUAL SETUP ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.15) # Make room for the button

# Central Mass
x_surf = 1.05
u = np.linspace(0, 2 * np.pi, 40) 
v = np.linspace(-1, 1, 40)        
U, V = np.meshgrid(u, v)
rho_surf = M * np.sqrt((x_surf**2 - 1) * (1 - V**2))
ax.plot_surface(rho_surf * np.cos(U), rho_surf * np.sin(U), M * x_surf * V, 
                color='black', alpha=0.9, edgecolor='red', linewidth=0.5)

trail_line, = ax.plot([], [], [], color='cyan', linewidth=1.5, alpha=0.8, label='Geodesic Trail')
planet, = ax.plot([], [], [], marker='o', color='white', markersize=8, label='Planet')

hud_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color='cyan', 
                     fontsize=11, family='monospace', verticalalignment='top',
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'))

limit = 15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])
ax.set_title(f"Interactive Orbital Simulation ($\gamma$={gamma})", color='white', fontsize=14)
ax.set_xlabel("X-Axis (M)")
ax.set_ylabel("Y-Axis (M)")
ax.set_zlabel("Z-Axis (M)")

ax.grid(color='gray', linestyle=':', linewidth=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.legend(loc='upper right')

# --- 4. INTERACTIVE CONTROLS ---
current_idx = [0]       # Using a list so it's mutable inside functions
speed_multiplier = [10] # Initial normal speed

def update(frame):
    idx = current_idx[0]
    
    # Stop condition so it doesn't crash at the end
    if idx >= len(X_orbit):
        idx = len(X_orbit) - 1
        ani.event_source.stop()
        
    trail_line.set_data(X_orbit[:idx], Y_orbit[:idx])
    trail_line.set_3d_properties(Z_orbit[:idx])
    planet.set_data([X_orbit[idx]], [Y_orbit[idx]])
    planet.set_3d_properties([Z_orbit[idx]])
    
    hud_text.set_text(f"Telemetry Data:\n"
                      f"TIME: {t_vals[idx]:06.1f} M\n"
                      f"RHO:  {rho[idx]:05.2f} M\n"
                      f"Z:    {Z_orbit[idx]:+05.2f} M\n"
                      f"SPEED MULTIPLIER: {speed_multiplier[0]}x")
    
    # Move forward by the current speed multiplier
    current_idx[0] += speed_multiplier[0]
    
    return trail_line, planet, hud_text

# Create the UI Button
button_ax = plt.axes([0.75, 0.05, 0.2, 0.06]) # [left, bottom, width, height]
fast_forward_btn = Button(button_ax, 'FAST FORWARD ⏭️', color='darkslategray', hovercolor='red')
fast_forward_btn.label.set_color('white')
fast_forward_btn.label.set_weight('bold')

def on_click(event):
    # When clicked, multiply the speed by 10!
    speed_multiplier[0] = 100
    fast_forward_btn.color = 'red'
    fast_forward_btn.label.set_text('WARP SPEED! 🚀')
    fig.canvas.draw_idle() # Force UI update

fast_forward_btn.on_clicked(on_click)

# Start animation (frames = extremely high number so it keeps running until our custom stop condition)
ani = FuncAnimation(fig, update, frames=10000, interval=25, blit=False)

plt.show()
