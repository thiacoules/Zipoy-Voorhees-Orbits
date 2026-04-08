# Orbital Study of the Zipoy-Voorhees Spacetime

This project provides a numerical integration framework to compute and analyze geodesic trajectories in the Zipoy-Voorhees (ZV) spacetime. The ZV metric (also known as the $\gamma$-metric) is a static, axially symmetric vacuum solution to Einstein's field equations.

## 1. Theoretical Background
The project is heavily based on the foundational work by *[Voorhees, B. H. (1970). Static Axially Symmetric Gravitational Fields. Physical Review D 2, 2119–2122]*. 

Unlike the Schwarzschild metric, which describes a perfectly spherical mass, the Zipoy-Voorhees spacetime allows for a continuous deformation of the central mass via the parameter $\gamma$. The multipole moments of the source are given by:
* **Monopole (Total Mass):** $M_0 = M \gamma$
* **Quadrupole Moment:** $Q = \frac{1}{3} M^3 \gamma (1 - \gamma^2)$

* If **$\gamma = 1$**: $Q = 0$ (Recovers the spherically symmetric Schwarzschild limit).
* If **$\gamma < 1$**: $Q > 0$ (Oblate source, structurally similar to a flattened disk).
* If **$\gamma > 1$**: $Q < 0$ (Prolate source, structurally similar to a cigar).

According to Voorhees (1970), the singularity in this spacetime is not a central point, but rather a "rod" of length $2M$ lying on the axis of symmetry. Furthermore, for $\gamma \neq 1$, this singularity is "naked" (it possesses no event horizon).

## 2. Methodology: Hamiltonian Dynamics
To compute the geodesic orbits without calculating dozens of Christoffel symbols, this project utilizes the **Hamiltonian formalism**. For a particle of unit mass ($\mu=1$), the Hamiltonian is:
$$H = \frac{1}{2} g^{\mu\nu} p_\mu p_\nu$$

The equations of motion are generated via automatic differentiation (using the `autograd` library) solving Hamilton's equations:
$$\frac{dx^\mu}{d\tau} = \frac{\partial H}{\partial p_\mu}$$
$$\frac{dp_\mu}{d\tau} = -\frac{\partial H}{\partial x^\mu}$$

## 3. Coordinate Systems
The numerical integration is performed in **Prolate Spheroidal / Weyl coordinates** $(x, y)$, where $x \ge 1$ is a radial-like coordinate and $-1 \le y \le 1$ is an angular-like coordinate ($y = \cos\theta$).

To visualize the orbits in a physically intuitive space, we transform the results into **Weyl Cylindrical Coordinates** $(\rho, z)$ using the standard transformations:
$$\rho = M \sqrt{(x^2 - 1)(1 - y^2)}$$
$$z = M x y$$

## 4. Numerical Results
Running the simulation (`analyze_orbits.py`) yields numerical comparisons between a spherical and a deformed mass with the exact same initial kinematic conditions ($x_0 = 10.0M, y_0 = 0$).

| Spacetime Model | $\gamma$ Parameter | Initial Distance | Periapsis (Min) | Apoapsis (Max) | Orbital Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Schwarzschild** | $\gamma = 1.0$ | $10.0 M$ | $\sim 0.99 M$ | $10.0 M$ | Plunging / Highly elliptical |
| **Zipoy-Voorhees** | $\gamma = 0.5$ | $10.0 M$ | $10.0 M$ | $\sim 48.2 M$ | Scattering / Strong equatorial repulsion |

**Conclusion of the Numerical Study:**
The introduction of a positive quadrupole moment ($\gamma = 0.5$, oblate mass) drastically alters the effective potential. A particle that would typically fall into a spherical black hole is instead repelled outward by the deformed gravitational field, demonstrating the profound orbital mechanics unique to the Zipoy-Voorhees geometry.

## 5. Repository Structure
* `core_physics.py`: Defines the ZV metric tensor and the Hamiltonian. Uses `autograd` for gradients.
* `integrator.py`: Solves the ODEs using SciPy's `solve_ivp`.
* `analyze_orbits.py`: Executes the simulation, extracts numerical boundaries, transforms coordinates to $(\rho, z)$, and plots the physical trajectories.

## Requirements
* Python 3.8+
* `numpy`
* `scipy`
* `matplotlib`
* `autograd`
