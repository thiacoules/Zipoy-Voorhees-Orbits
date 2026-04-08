import numpy as np
import matplotlib.pyplot as plt
from integrator import get_geodesics

# 1. System Parameters
M = 1.0
time_span = 500
init_state = [10.0, 0.0, 0.0, 0.1] # [x, y, px, py]

print("--- NUMERICAL RESULTS: ZIPOY-VOORHEES SPACETIME ---")

def process_and_print_results(sol, gamma, name):
    x = sol.y[0]
    y = sol.y[1]
    
    # Numerical Data: Periapsis (Min x) and Apoapsis (Max x)
    min_x = np.min(x)
    max_x = np.max(x)
    final_x = x[-1]
    
    print(f"\n[{name} (gamma = {gamma})]")
    print(f"Initial Distance (x0): {x[0]:.4f} M")
    print(f"Closest Approach (Periapsis): {min_x:.4f} M")
    print(f"Maximum Distance (Apoapsis): {max_x:.4f} M")
    print(f"Final Position at t={time_span}: {final_x:.4f} M")
    
    # Convert to Cylindrical Coordinates (rho, z)
    # Based on the Voorhees (1970) coordinate transformation
    rho = M * np.sqrt(np.abs((x**2 - 1) * (1 - y**2)))
    z = M * x * y
    return rho, z

# 2. Execute Integration
sol_sph = get_geodesics(M, gamma=1.0, initial_conditions=init_state, total_time=time_span)
sol_zv  = get_geodesics(M, gamma=0.5, initial_conditions=init_state, total_time=time_span)

# 3. Extract Numerical Data & Transform Coordinates
rho_sph, z_sph = process_and_print_results(sol_sph, 1.0, "Schwarzschild")
rho_zv, z_zv = process_and_print_results(sol_zv, 0.5, "Zipoy-Voorhees Oblate")

# 4. Plotting in Physical Space (rho, z)
plt.figure(figsize=(10, 6))
plt.plot(rho_sph, z_sph, label='Schwarzschild ($\gamma=1.0$)', color='blue')
plt.plot(rho_zv, z_zv, label='Zipoy-Voorhees ($\gamma=0.5$)', color='red', linestyle='--')

# Draw the Singularity Rod
plt.plot([0, 0], [-M, M], color='black', linewidth=4, label='Singularity Rod (x=1)')

plt.title("Orbital Study (Cylindrical Coordinates ρ-z)")
plt.xlabel("Radial Distance ρ (M)")
plt.ylabel("Vertical Distance z (M)")
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensures the scale is 1:1 so orbits don't look artificially stretched
plt.show()
