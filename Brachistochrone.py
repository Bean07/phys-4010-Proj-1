# Cell B1: Setup the geometry for the brachistochrone problem
import numpy as np
import matplotlib.pyplot as plt

##TODO gravity
g = 9.8  # gravitational acceleration in m/s^2

# Define the endpoint coordinates, in meters
x_end, y_end = 5, 5

# number of discretization points (*including* endpoints)
n_points = 15

x_loc = np.linspace(0, x_end, n_points)
Delta_x = x_end / (n_points - 1)  # Delta_x is the same for all segments

# Straight line for reference
y_straight = np.linspace(0, y_end, n_points)

plt.figure(figsize=(7,4), dpi=120)
plt.plot(x_loc, y_straight, 'o-', label='Straight Line')
plt.ylim(plt.gca().get_ylim()[::-1])  # Reverse y-limits
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid();

# Cell B2: Define the objective function (travel time)

def travel_time(y_internal, Delta_x, y_end, g):
    # `y_internal` has the y locations excluding endpoints
    # `y_full` stores the full y locations including endpoints
    y_full = np.concatenate(([0], y_internal, [y_end]))

    total_time = 0.0
    for i in range(len(y_full) - 1):  # Calculate each segment travel time
        if y_full[i+1] < 0:
            return np.inf
        Delta_y = y_full[i+1] - y_full[i]
        Delta_s = np.sqrt(Delta_x**2 + Delta_y**2)
        Delta_t = Delta_s / (np.sqrt(2.0 * g) * 0.5 * (np.sqrt(y_full[i+1]) + np.sqrt(y_full[i])))
        # Delta_t = Delta_s / (np.sqrt(2.0 * g * y_full[i+1])) # Uncomment to see what happens if we use the worse approximation
        total_time += Delta_t
    return total_time

# Test the function with the initial straight line
# The first argument that we pass to `travel_time` needs to exclude the endpoints
y_internal_initial = y_straight[1:-1]
print('Travel time (straight path):', travel_time(y_internal_initial, Delta_x, y_end, g))   

# Cell B3: Carry out the minimization and show the results
from scipy.optimize import minimize
from IPython.display import display

result = minimize(travel_time, y_internal_initial, args=(Delta_x, y_end, g), method='BFGS')
display(result)

# Cell B4: Compute the analytic solution of the brachistochrone curve (see Worksheet 12)
from scipy.optimize import brentq

theta_max = brentq(
    lambda theta: (1 - np.cos(theta)) / (theta - np.sin(theta)) - y_end / x_end,
    1e-6, np.pi*2.0 - 1e-6
)
a = x_end / (theta_max - np.sin(theta_max))
theta = np.linspace(0, theta_max, 101)
x_brachi = (theta - np.sin(theta)) * a
y_brachi = (1 - np.cos(theta)) * a

# Cell B5: Plot the results

# Note that result.x contains the optimized y-coordinates (excluding endpoints)
y_opt = np.concatenate(([0], result.x, [y_end]))

plt.figure(figsize=(7,4), dpi=120)

plt.plot(x_loc, y_straight, 'o-', label='Straight Line')
plt.plot(x_loc, y_opt, 's-', label='Optimized path')
plt.plot(x_brachi, y_brachi, 'k--', lw=2, label='Brachistochrone curve')
plt.ylim(plt.gca().get_ylim()[::-1])  # Reverse y-limits
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid();