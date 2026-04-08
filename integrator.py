import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp
from core_physics import ZVEngine

def get_geodesics(M, gamma, initial_conditions, total_time):
    engine = ZVEngine(M, gamma)
    
    # We define the Hamiltonian gradient (The 'Forces')
    # This is where autograd shines: it derives the physics automatically!
    def hamiltonian_func(state_coords, momentum):
        return engine.hamiltonian(state_coords, momentum)

    # Gradient of Hamiltonian with respect to coordinates
    dh_dq = grad(engine.hamiltonian, 0)
    # Gradient of Hamiltonian with respect to momenta
    dh_dp = grad(engine.hamiltonian, 1)

    def system_dynamics(t, y):
        q_vec = np.array([y[0], y[1]])
        p_vec = np.array([-0.95, y[2], y[3], 3.0]) # pt, px, py, p_phi
        
        dq_dt = dh_dp(q_vec, p_vec)[1:3] 
        dp_dt = -dh_dq(q_vec, p_vec)
        
        return np.concatenate([dq_dt, dp_dt])

    # Solving the differential equations
    sol = solve_ivp(system_dynamics, [0, total_time], initial_conditions, t_eval=np.linspace(0, total_time, 1000))
    return sol
