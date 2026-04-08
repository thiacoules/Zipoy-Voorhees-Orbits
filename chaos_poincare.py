import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from core_physics import ZVEngine

def get_poincare_crossings(M, gamma, init_state, total_time):
    engine = ZVEngine(M, gamma)
    dh_dq = grad(engine.hamiltonian, 0)
    dh_dp = grad(engine.hamiltonian, 1)

    # The "Goldilocks" momentum to keep orbits trapped
    pt0 = -0.95   
    p_phi0 = 3.0  

    def system_dynamics(t, state):
        x, y, px, py = state
        if x <= 1.05: 
            return [0.0, 0.0, 0.0, 0.0]
            
        q_vec = np.array([x, y])
        p_vec = np.array([pt0, px, py, p_phi0])
        
        dHdp = dh_dp(q_vec, p_vec)
        dHdq = dh_dq(q_vec, p_vec)
        
        return [dHdp[1], dHdp[2], -dHdq[0], -dHdq[1]]

    # Trigger exactly when y crosses 0
    def equator_crossing(t, state):
        return state[1] 
    
    # Catching crossings going in one direction to map the section cleanly
    equator_crossing.direction = 1 

    sol = solve_ivp(system_dynamics, [0, total_time], init_state, 
                    events=equator_crossing, max_step=1.0)
    
    return sol.y_events[0]

# --- SIMULATION SETUP ---
M = 1.0
gamma = 0.5  # Oblate mass
time_span = 10000  # 10,000 units of time to gather dots

plt.figure(figsize=(9, 7))
plt.style.use('dark_background')
print("Calculating Chaos via Poincaré Section...")
print("Finding trapped chaotic orbits. Please wait ~15 seconds...")

# Launching particles from a safe distance
starting_x = np.linspace(8.0, 15.0, 6)
colors = plt.cm.spring(np.linspace(0, 1, len(starting_x)))

plotted_anything = False

for i, x_start in enumerate(starting_x):
    # init_state = [x, y, px, py]
    # We start them slightly off the equator (y=-0.1) so they fall towards it
    init = [x_start, -0.1, 0.0, 0.1] 
    
    crossings = get_poincare_crossings(M, gamma, init, time_span)
    
    if len(crossings) > 0:
        plotted_anything = True
        x_cross = crossings[:, 0]
        px_cross = crossings[:, 2]
        plt.scatter(x_cross, px_cross, s=4, color=colors[i], label=f'Initial x = {x_start:.1f}')

if not plotted_anything:
    print("WARNING: Particles still escaped. Try adjusting initial conditions.")

plt.title("Poincaré Section: Orbital Dynamics in ZV Spacetime ($\gamma=0.5$)", fontsize=14, color='white')
plt.xlabel("Radial Distance x at Equator")
plt.ylabel("Radial Momentum $p_x$ at Equator")
plt.axhline(0, color='white', linewidth=0.5, linestyle='--')

ax = plt.gca()
ax.tick_params(colors='white')
ax.grid(color='gray', linestyle=':', linewidth=0.3)
if plotted_anything:
    plt.legend(loc='upper right', markerscale=3)
plt.show()
