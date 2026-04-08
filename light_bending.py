import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

def get_null_geodesic(M, gamma, init_state, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    x0, y0, px0, py0, p_phi0 = init_state
    g_tt, g_xx, g_yy, g_phi = engine.metric_components(x0, y0)
    
    spatial_momentum = (px0**2 / g_xx) + (py0**2 / g_yy) + (p_phi0**2 / g_phi)
    pt0 = -np.sqrt(spatial_momentum * g_tt)

    def system_dynamics(t, state):
        x, y, px, py = state
        
        # SAFEGUARD: If the solver accidentally overshoots into the singularity,
        # return zero derivatives to freeze the calculation and avoid math errors.
        if x <= 1.01:
            return [0.0, 0.0, 0.0, 0.0]
            
        q_vec = np.array([x, y])
        p_vec = np.array([pt0, px, py, p_phi0])
        
        dHdp = dh_dp(q_vec, p_vec)
        dHdq = dh_dq(q_vec, p_vec)
        
        return [dHdp[1], dHdp[2], -dHdq[0], -dHdq[1]]

    def hit_singularity(t, state):
        return state[0] - 1.05 # Stop the ray slightly further away
    hit_singularity.terminal = True

    ode_init = [x0, y0, px0, py0]
    # Reduced max_step to 0.1 so the solver is more careful near the singularity
    sol = solve_ivp(system_dynamics, [0, total_time], ode_init, 
                    events=hit_singularity, max_step=0.1)
    return sol

# --- SIMULATION SETUP ---
M = 1.0
gamma = 0.5  
time_span = 100

plt.figure(figsize=(10, 8))
plt.style.use('dark_background')

print("Shooting laser beams at the Zipoy-Voorhees singularity...")

starting_heights = np.linspace(-0.8, 0.8, 7)
colors = plt.cm.rainbow(np.linspace(0, 1, len(starting_heights)))

for i, y_start in enumerate(starting_heights):
    init = [15.0, y_start, -1.0, 0.0, 0.5] 
    sol = get_null_geodesic(M, gamma, init, time_span)
    
    x_vals, y_vals = sol.y[0], sol.y[1]
    rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
    z = M * x_vals * y_vals
    
    plt.plot(rho, z, color=colors[i], linewidth=2, label=f'Ray {i+1}')

plt.plot([0, 0], [-M, M], color='white', linewidth=6, label='Naked Singularity')

plt.title("Gravitational Lensing: Light Bending in ZV Spacetime ($\gamma=0.5$)", fontsize=14, color='white')
plt.xlabel("Radial Distance ρ (M)")
plt.ylabel("Vertical Distance z (M)")
plt.xlim([-1, 16])
plt.ylim([-10, 10])

# Keep grid and axes visible on dark background
ax = plt.gca()
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.grid(color='gray', linestyle=':', linewidth=0.5)

plt.legend(loc='upper right')
plt.show()
