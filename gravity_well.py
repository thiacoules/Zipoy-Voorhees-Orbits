import numpy as np
import matplotlib.pyplot as plt

# Ορισμός του δυναμικού (g_tt component)
def gravitational_potential(x, gamma):
    # Αποφυγή διαίρεσης με το μηδέν ακριβώς πάνω στην ιδιομορφία (x=1)
    x = np.where(x <= 1.01, 1.01, x) 
    return ((x - 1) / (x + 1))**gamma

# Δημιουργία πλέγματος συντεταγμένων (Grid)
M = 1.0
x_vals = np.linspace(1.01, 15, 400)
y_vals = np.linspace(-1, 1, 400)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# Μετατροπή σε Κυλινδρικές Συντεταγμένες (για να φαίνεται στο φυσικό χώρο)
Rho = M * np.sqrt((X_grid**2 - 1) * (1 - Y_grid**2))
Z = M * X_grid * Y_grid

# Υπολογισμός του πηγαδιού βαρύτητας για δύο περιπτώσεις
V_spherical = gravitational_potential(X_grid, gamma=1.0)
V_deformed = gravitational_potential(X_grid, gamma=0.5)

# --- ΣΧΕΔΙΑΣΜΟΣ ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Χάρτης 1: Schwarzschild
contour1 = ax1.contourf(Rho, Z, V_spherical, levels=50, cmap='magma')
ax1.plot([0, 0], [-M, M], color='cyan', linewidth=3, label='Singularity')
ax1.set_title("Gravity Well: Schwarzschild ($\gamma=1.0$)")
ax1.set_xlabel("Radial Distance ρ (M)")
ax1.set_ylabel("Vertical Distance z (M)")
ax1.axis('equal')
fig.colorbar(contour1, ax=ax1, label='Spacetime Warping ($g_{tt}$)')

# Χάρτης 2: Zipoy-Voorhees
contour2 = ax2.contourf(Rho, Z, V_deformed, levels=50, cmap='magma')
ax2.plot([0, 0], [-M, M], color='cyan', linewidth=3, label='Singularity Rod')
ax2.set_title("Gravity Well: Zipoy-Voorhees ($\gamma=0.5$)")
ax2.set_xlabel("Radial Distance ρ (M)")
ax2.axis('equal')
fig.colorbar(contour2, ax=ax2, label='Spacetime Warping ($g_{tt}$)')

plt.suptitle("Topographical Map of the Gravitational Field", fontsize=16)
plt.tight_layout()
plt.show()
