import matplotlib.pyplot as plt
import numpy as np
from integrator import get_geodesics

# 1. Setup Parameters
M = 1.0
time_span = 500
# Initial conditions: [x, y, px, py]
# x=10 (distance), y=0 (equator), px=0 (radial mom), py=0.1 (latitudinal mom)
init_state = [10.0, 0.0, 0.0, 0.1]

print("Launching geodesics in ZV spacetime...")

# 2. Run Simulations
# Case A: Spherical (Schwarzschild limit)
sol_spherical = get_geodesics(M, gamma=1.0, initial_conditions=init_state, total_time=time_span)

# Case B: Deformed (Zipoy-Voorhees)
sol_deformed = get_geodesics(M, gamma=0.5, initial_conditions=init_state, total_time=time_span)

# 3. Plotting
plt.figure(figsize=(10, 5))

# Plot X-Y plane (Weyl coordinates)
plt.plot(sol_spherical.y[0], sol_spherical.y[1], label='Schwarzschild ($\gamma=1.0$)', color='blue')
plt.plot(sol_deformed.y[0], sol_deformed.y[1], label='Zipoy-Voorhees ($\gamma=0.5$)', color='red', linestyle='--')

plt.title("Geodesic Motion: Schwarzschild vs. Zipoy-Voorhees")
plt.xlabel("Coordinate x")
plt.ylabel("Coordinate y")
plt.legend()
plt.grid(True)
plt.show()
