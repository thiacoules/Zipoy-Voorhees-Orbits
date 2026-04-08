import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

# ==========================================
# 1. THE 3D INTEGRATOR
# ==========================================
def get_3d_geodesics(M, gamma, initial_conditions, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    def system_dynamics(t, state):
        x, y, phi, px, py = state  # We added phi to the state!
        q_vec = np.array([x, y])
        
        # p_phi = 3.0 gives the particle its spin around the z-axis
        p_vec = np.array([-0.95, px, py, 3.0]) 
        
        dHdp = dh_dp(q_vec, p_vec)
        dHdq = dh_dq(q_vec, p_vec)
        
        dx_dt = dHdp[1]
        dy_dt = dHdp[2]
        dphi_dt = dHdp[3] # The rotational velocity!
        
        dpx_dt = -dHdq[0]
        dpy_dt = -dHdq[1]
        
        return [dx_dt, dy_dt, dphi_dt, dpx_dt, dpy_dt]

    # We use more points (2000) so the 3D line is perfectly smooth
    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, 
                    t_eval=np.linspace(0, total_time, 2000))
    return sol

# ==========================================
# 2. RUN THE SIMULATION
# ==========================================
M = 1.0
time_span = 800  # Longer time to see a full wrapping orbit
# [x, y, phi, px, py] -> Starting at equator (y=0), angle (phi=0)
init_state = [10.0, 0.0, 0.0, 0.0, 0.1] 

print("Calculating 3D trajectories. This might take a few seconds...")
sol_zv = get_3d_geodesics(M, gamma=0.5, initial_conditions=init_state, total_time=time_span)

# Extract coordinates
x_vals = sol_zv.y[0]
y_vals = sol_zv.y[1]
phi_vals = sol_zv.y[2]

# ==========================================
# 3. TRANSFORM TO 3D CARTESIAN SPACE (X, Y, Z)
# ==========================================
# First to cylindrical...
rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
z_cyl = M * x_vals * y_vals

# Then to full 3D Cartesian...
X = rho * np.cos(phi_vals)
Y = rho * np.sin(phi_vals)
Z = z_cyl

# ==========================================
# 4. DRAW THE PRESENTATION PLOT
# ==========================================
# Use a dark background to make it look like space
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(X, Y, Z, color='cyan', linewidth=1.5, label='ZV Orbit ($\gamma=0.5$)')

# Draw the Singularity Rod in the center
ax.plot([0, 0], [0, 0], [-M, M], color='red', linewidth=5, label='Singularity Rod')

# Set visual limits and labels
limit = 20
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

ax.set_title("3D Orbital Plane in Zipoy-Voorhees Spacetime", fontsize=14, color='white')
ax.set_xlabel("X (M)")
ax.set_ylabel("Y (M)")
ax.set_zlabel("Z (M)")

# Remove the default grid panes for a cleaner space look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(color='gray', linestyle=':', linewidth=0.5)

ax.legend()
plt.show()
