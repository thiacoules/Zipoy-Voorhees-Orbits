import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

def get_poincare_crossings(M, gamma, init_state, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    # Constants of motion
    pt0 = -0.96   # Energy related
    p_phi0 = 3.2  # Angular momentum

    def system_dynamics(t, state):
        x, y, px, py = state
        if x <= 1.05: # Safeguard
            return [0.0, 0.0, 0.0, 0.0]
            
        q_vec = np.array([x, y])
        p_vec = np.array([pt0, px, py, p_phi0])
        
        dHdp = dh_dp(q_vec, p_vec)
        dHdq = dh_dq(q_vec, p_vec)
        
        return [dHdp[1], dHdp[2], -dHdq[0], -dHdq[1]]

    # EVENT DETECTOR: Trigger exactly when the particle crosses the equator (y=0)
    def equator_crossing(t, state):
        return state[1] # state[1] is y
    
    # We only care when it crosses from North to South to avoid double-counting
    equator_crossing.direction = -1 

    # We need a very long time to get enough "punches" through the paper
    sol = solve_ivp(system_dynamics, [0, total_time], init_state, 
                    events=equator_crossing, max_step=1.0)
    
    # sol.y_events[0] contains the [x, y, px, py] exact state at every crossing!
    return sol.y_events[0]

# --- SIMULATION SETUP ---
M = 1.0
gamma = 2.0  # We use a Prolate (cigar-shaped) mass here, as it induces extreme chaos!
time_span = 50000  # Massive integration time to generate thousands of dots

plt.figure(figsize=(9, 7))
plt.style.use('dark_background')
print("Calculating Chaos via Poincaré Section. This requires heavy computing...")
print("Please wait ~15-30 seconds...")

# Launching particles from different starting distances
# x_starts: 6.0, 7.0, 8.0, 9.0, 10.0
starting_x = np.linspace(6.0, 10.0, 5)
colors = plt.cm.spring(np.linspace(0, 1, len(starting_x)))

for i, x_start in enumerate(starting_x):
    # init_state = [x, y, px, py]
    init = [x_start, 0.0, 0.0, 0.2] 
    
    crossings = get_poincare_crossings(M, gamma, init, time_span)
    
    if len(crossings) > 0:
        # Extract x and px at the crossings
        x_cross = crossings[:, 0]
        px_cross = crossings[:, 2]
        
        # Scatter plot the dots
        plt.scatter(x_cross, px_cross, s=2, color=colors[i], label=f'Initial x = {x_start}')

plt.title("Poincaré Section: Deterministic Chaos in ZV Spacetime ($\gamma=2.0$)", fontsize=14, color='white')
plt.xlabel("Radial Distance x at Equator")
plt.ylabel("Radial Momentum $p_x$ at Equator")
plt.axhline(0, color='white', linewidth=0.5, linestyle='--')

# Styling
ax = plt.gca()
ax.tick_params(colors='white')
ax.grid(color='gray', linestyle=':', linewidth=0.3)
plt.legend(loc='upper right', markerscale=5)
plt.show()
