
# ============================================================
# CENTRAL-FORCE BRACHISTOCHRONE
# Minimum-time path in a central gravitational field
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss


# ============================================================
# CELL B1: SETUP CONSTANTS AND GEOMETRY
# ============================================================

# Solar gravitational parameter
GM = 1.3271244e20  # m^3/s^2

# Astronomical unit
AU = 1.495978707e11  # meters

# Starting and ending radial coordinates (AU)
r_start = 1.0
r_end = 0.4

# Starting and ending angular coordinates (degrees)
phi_start = 0.0
phi_end = 90.0

# Convert angles to radians
phi_start = np.radians(phi_start)
phi_end = np.radians(phi_end)

# Minimum permitted radius (AU)
# Adjustable constraint to prevent the path approaching r = 0
r_min = 0.0465

# Number of discretization points (including endpoints)
n_points = 30

# Number of Gaussian quadrature points per segment
n_quad = 64


# Validate endpoint configuration
if r_end > r_start:
    raise ValueError(
        "Starting from rest, the endpoint must be at or "
        "inside the starting radius for this gravitational model."
    )

if r_min <= 0 or r_min > r_end:
    raise ValueError("Invalid minimum radius.")

# Fixed angular and radial endpoints are specified above.


# ============================================================
# CELL B2: INITIAL PATH
# ============================================================

# Parameter along the path
u_loc = np.linspace(0.0, 1.0, n_points)

# Initial guess: linearly interpolate radius and angle
r_initial = np.linspace(r_start, r_end, n_points)
phi_initial = np.linspace(phi_start, phi_end, n_points)

# If both endpoints have the same radius, introduce an
# inward initial deviation so the object can start accelerating.
if np.isclose(r_start, r_end):
    r_initial = (
        r_start
        - 0.1 * (r_start - r_min)
        * np.sin(np.pi * u_loc)
    )

# Convert initial path into Cartesian coordinates
x_initial = r_initial * np.cos(phi_initial)
y_initial = r_initial * np.sin(phi_initial)

# Plot initial configuration
plt.figure(figsize=(7, 6), dpi=120)

plt.plot(
    x_initial, y_initial,
    'o-', label='Initial Path'
)

plt.scatter(
    0, 0,
    color='orange',
    s=100,
    label='Central Mass'
)

plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Initial Path')
plt.axis('equal')
plt.legend()
plt.grid()
plt.show()


# ============================================================
# CELL B3: GAUSSIAN QUADRATURE
# ============================================================

# Generate Gaussian quadrature points and weights
# Transform standard interval [-1, 1] to [0, 1]

quad_nodes, quad_weights = leggauss(n_quad)

quad_nodes = (quad_nodes + 1.0) / 2.0
quad_weights = quad_weights / 2.0


# ============================================================
# CELL B4: DEFINE OBJECTIVE FUNCTION
# ============================================================

def travel_time(variables):

    # First half of variables contains internal radii
    # Second half contains internal angular coordinates

    n_internal = n_points - 2

    r_internal = variables[:n_internal]
    phi_internal = variables[n_internal:]

    # Reconstruct complete path
    r_full = np.concatenate((
        [r_start],
        r_internal,
        [r_end]
    ))

    phi_full = np.concatenate((
        [phi_start],
        phi_internal,
        [phi_end]
    ))

    total_time = 0.0

    # Calculate travel time for each segment
    for i in range(n_points - 1):

        # Segment coordinate differences
        dr = r_full[i+1] - r_full[i]
        dphi = phi_full[i+1] - phi_full[i]

        # Evaluate positions along the segment
        u = quad_nodes

        r = r_full[i] + u * dr

        # Convert radius to meters
        r_m = r * AU
        r_start_m = r_start * AU

        # Conservation of energy:
        # v^2 = 2GM(1/r - 1/r_start)

        v_squared = 2.0 * GM * (
            1.0 / r_m - 1.0 / r_start_m
        )

        # Invalid path: zero or imaginary velocity
        if np.any(v_squared <= 0):
            return 1e100

        velocity = np.sqrt(v_squared)

        # Differential arc length with respect to u
        # ds/du = sqrt((dr/du)^2 + r^2(dphi/du)^2)

        ds_du = AU * np.sqrt(
            dr**2 + (r * dphi)**2
        )

        # Integrate dt = ds/v using Gaussian quadrature
        delta_t = np.sum(
            quad_weights * ds_du / velocity
        )

        total_time += delta_t

    return total_time


# ============================================================
# CELL B5: INITIALIZE OPTIMIZATION
# ============================================================

# Internal coordinates only
r_internal_initial = r_initial[1:-1]
phi_internal_initial = phi_initial[1:-1]

# Combine optimization variables
initial_guess = np.concatenate((
    r_internal_initial,
    phi_internal_initial
))

# Radial constraints
r_bounds = [
    (r_min, r_start * (1.0 - 1e-10))
    for _ in range(n_points - 2)
]

# Angular constraints
# Allow up to one additional revolution in either direction
phi_low = min(phi_start, phi_end) - 2*np.pi
phi_high = max(phi_start, phi_end) + 2*np.pi

phi_bounds = [
    (phi_low, phi_high)
    for _ in range(n_points - 2)
]

bounds = r_bounds + phi_bounds


# ============================================================
# CELL B6: CARRY OUT MINIMIZATION
# ============================================================

result = minimize(
    travel_time,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={
        'maxiter': 10000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 50
    }
)

print(result)

print("\nOptimization successful:", result.success)
print("Optimization message:", result.message)
print("Minimum travel time:", result.fun, "seconds")
print("Minimum travel time:", result.fun / 3600, "hours")


# ============================================================
# CELL B7: RECONSTRUCT OPTIMIZED PATH
# ============================================================

n_internal = n_points - 2

# Extract optimized coordinates
r_opt_internal = result.x[:n_internal]
phi_opt_internal = result.x[n_internal:]

# Reconstruct full path
r_opt = np.concatenate((
    [r_start],
    r_opt_internal,
    [r_end]
))

phi_opt = np.concatenate((
    [phi_start],
    phi_opt_internal,
    [phi_end]
))

# Convert to Cartesian coordinates
x_opt = r_opt * np.cos(phi_opt)
y_opt = r_opt * np.sin(phi_opt)

# Initial path in Cartesian coordinates
x_initial = r_initial * np.cos(phi_initial)
y_initial = r_initial * np.sin(phi_initial)

# Straight-line reference
x_start = r_start * np.cos(phi_start)
y_start = r_start * np.sin(phi_start)

x_end = r_end * np.cos(phi_end)
y_end = r_end * np.sin(phi_end)

x_straight = np.linspace(x_start, x_end, n_points)
y_straight = np.linspace(y_start, y_end, n_points)


# ============================================================
# CELL B8: PLOT RESULTS
# ============================================================

plt.figure(figsize=(8, 7), dpi=120)

# Initial path
plt.plot(
    x_initial, y_initial,
    'o-', alpha=0.5,
    label='Initial Path'
)

# Optimized path
plt.plot(
    x_opt, y_opt,
    's-', linewidth=2,
    label='Optimized Path'
)

# Straight-line reference
plt.plot(
    x_straight, y_straight,
    'k--',
    label='Straight Line'
)

# Central mass
plt.scatter(
    0, 0,
    color='orange',
    s=120,
    label='Central Mass',
    zorder=5
)

# Minimum-radius boundary
circle = plt.Circle(
    (0, 0),
    r_min,
    fill=False,
    linestyle=':',
    color='red',
    label='Minimum Radius'
)

plt.gca().add_patch(circle)

# Mark endpoints
plt.scatter(
    [x_start, x_end],
    [y_start, y_end],
    color='black',
    s=40,
    zorder=6
)

plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Central Force Brachistochrone')

plt.axis('equal')
plt.legend()
plt.grid()
plt.show()


# ============================================================
# CELL B9: REPORT OPTIMIZED COORDINATES
# ============================================================

print("\nOptimized path coordinates:")
print("--------------------------------")

for i in range(n_points):
    print(
        f"Point {i:2d}: "
        f"r = {r_opt[i]:.6f} AU, "
        f"phi = {np.degrees(phi_opt[i]):.3f} degrees"
    )

print("--------------------------------")
print("Closest approach:", np.min(r_opt), "AU")
print("Initial travel time:",
      travel_time(initial_guess), "seconds")
print("Optimized travel time:",
      result.fun, "seconds")
