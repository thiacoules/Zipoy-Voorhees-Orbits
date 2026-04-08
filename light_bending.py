import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

def get_null_geodesic(M, gamma, init_state, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    # For light, H must be 0. We calculate the required p_t to satisfy this.
    x0, y0, px0, py0, p_phi0 = init_state
    g_tt, g_xx, g_yy, g_phi = engine.metric_components(x0, y0)
    
    # Solving 0 = - (pt^2 / g_tt) + (px^2 / g_xx) + (py^2 / g_yy) + (p_phi^2 / g_phi) for pt
    spatial_momentum = (px0**2 / g_xx) + (py0**2 / g_yy) + (p_phi0**2 / g_phi)
    pt0 = -np.sqrt(spatial_momentum * g_tt) # Negative because pt = -Energy

    def system_dynamics(t, state):
        x, y, px, py = state
        q_vec = np.array([x, y])
        p_vec = np.array([pt0, px, py, p_phi0])
        
        dHdp = dh_dp(q_vec, p_vec)
        dHdq = dh_dq(q_vec, p_vec)
        
        return [dHdp[1], dHdp[2], -dHdq[0], -dHdq[1]]

    # For light, we stop the integration if it hits the singularity (x -> 1.01)
    def hit_singularity(t, state):
        return state[0] - 1.02
    hit_singularity.terminal = True

    # We only need [x, y, px, py] for the ODE solver
    ode_init = [x0, y0, px0, py0]
    sol = solve_ivp(system_dynamics, [0, total_time], ode_init, 
                    events=hit_singularity, max_step=0.5)
    return sol

# --- SIMULATION SETUP ---
M = 1.0
gamma = 0.5  # Zipoy-Voorhees Oblate mass
time_span = 100

plt.figure(figsize=(10, 8))
plt.style.use('dark_background')

print("Shooting laser beams at the Zipoy-Voorhees singularity...")

# We will shoot 7 light rays from the right (x=15) moving left (px = -1.0)
# We vary their starting height (y) to see how the lens bends them
starting_heights = np.linspace(-0.8, 0.8, 7)

colors = plt.cm.rainbow(np.linspace(0, 1, len(starting_heights)))

for i, y_start in enumerate(starting_heights):
    # init_state = [x, y, px, py, p_phi]
    init = [15.0, y_start, -1.0, 0.0, 0.5] 
    sol = get_null_geodesic(M, gamma, init, time_span)
    
    # Convert Weyl to Cylindrical for plotting
    x_vals, y_vals = sol.y[0], sol.y[1]
    rho = M * np.sqrt(np.abs((x_vals**2 - 1) * (1 - y_vals**2)))
    z = M * x_vals * y_vals
    
    # Because they start from the right and move left, we plot Rho as positive
    plt.plot(rho, z, color=colors[i], linewidth=2, label=f'Ray {i+1}')

# Draw the Singularity
plt.plot([0, 0], [-M, M], color='white', linewidth=6, label='Naked Singularity')

plt.title("Gravitational Lensing: Light Bending in ZV Spacetime ($\gamma=0.5$)", fontsize=14)
plt.xlabel("Radial Distance ρ (M)")
plt.ylabel("Vertical Distance z (M)")
plt.xlim([-1, 16])
plt.ylim([-10, 10])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.legend(loc='upper right')
plt.show()
